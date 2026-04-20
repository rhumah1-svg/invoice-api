"""
Invoice/Quote PDF Extraction API - V16.2 HYBRID (Extract Only)
===============================================================
Mode HYBRIDE : pdfplumber extrait le TEXTE du PDF, envoyé à gpt-5.4-mini.
Plus besoin d'images → 5-8s au lieu de 30-40s, coût /10.

  - extract_with_text() : fonction principale (texte pdfplumber → LLM)
  - extract_with_vision() : conservée en FALLBACK si pdfplumber échoue
  - Endpoints : /health + /extract uniquement
  - Modèle unique : gpt-5.4-mini (override possible via OPENAI_MODEL env)
"""

import os, io, logging, base64, re, gc, json, time
import httpx
from typing import Optional

import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"


def get_model() -> str:
    """Retourne le modèle à utiliser. Override possible via env OPENAI_MODEL."""
    override = os.getenv("OPENAI_MODEL", "").strip()
    if override:
        logger.info(f"[model] override env → {override}")
        return override
    return DEFAULT_MODEL


# ═══════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════

class Metadata(BaseModel):
    vendor_name:    str = Field(..., description="Nom de l'entreprise cliente")
    project_name:   str = Field(..., description="Nom du projet / chantier")
    invoice_number: str = Field(..., description="Référence devis_dexxxxxx")
    date:           str = Field(..., description="Date YYYY-MM-DD")
    currency:       str = Field(..., description="Code devise ISO")

class LineItem(BaseModel):
    designation: str   = Field(..., description="Nom court du produit/service (5-8 mots max)")
    description: str   = Field(
        default="",
        description=(
            "Texte COMPLET de la cellule Description pour cet item. "
            "Inclure: sous-sections, tirets, détails techniques, normes. "
            "EXCLURE CGV: 'Travaux réalisés en semaine...', "
            "'Fourniture eau/électricité...', 'QUALIDAL n est pas responsable...'"
        )
    )
    quantity:    float = Field(..., description="Quantité")
    unite:       str   = Field(..., description="Unité (ML, M2, FORF, U, etc.)")
    unit_price:  float = Field(..., description="Prix unitaire HT")

class Totals(BaseModel):
    subtotal_ht: float = Field(..., description="Total HT")
    total_tax:   float = Field(..., description="TVA")
    total_ttc:   float = Field(..., description="Total TTC")

class InvoiceData(BaseModel):
    metadata:   Metadata
    line_items: list[LineItem]
    totals:     Totals

class ExtractionResponse(BaseModel):
    success:    bool
    data:       Optional[InvoiceData] = None
    error:      Optional[str]         = None
    file_url:   Optional[str]         = None
    model_used: Optional[str]         = None


# ═══════════════════════════════════════════════════════
# PDF → IMAGES BASE64 (pour fallback vision)
# ═══════════════════════════════════════════════════════

def pdf_bytes_to_images_b64(pdf_bytes: bytes, dpi: int = 200) -> list[str]:
    """Chaque page du PDF → JPEG base64. DPI 200 pour meilleure lisibilité."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            buf = io.BytesIO(pix.tobytes("jpeg", jpg_quality=75))
            out.append(base64.b64encode(buf.getvalue()).decode())
        doc.close()
        logger.info(f"[vision] pymupdf {len(out)} page(s) @ {dpi} DPI")
        return out
    except ImportError:
        logger.warning("[vision] pymupdf absent, essai pdf2image")

    try:
        import pdf2image
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi, fmt="jpeg")
        out = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            out.append(base64.b64encode(buf.getvalue()).decode())
        logger.info(f"[vision] pdf2image {len(out)} page(s)")
        return out
    except ImportError:
        pass

    raise RuntimeError(
        "Aucune librairie de rendu PDF.\n"
        "Solution : pip install pymupdf"
    )


# ═══════════════════════════════════════════════════════
# EXTRACTION TEXTE PDFPLUMBER
# ═══════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extrait le texte de toutes les pages du PDF via pdfplumber."""
    all_text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                all_text += (page.extract_text() or "") + "\n"
    except Exception as e:
        logger.warning(f"[text] pdfplumber échoué: {e}")
        return ""
    return all_text.strip()


# ═══════════════════════════════════════════════════════
# PROMPT TEXTE — V16.1 (optimisé pour texte pdfplumber)
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT_TEXT = """\
Tu es un expert en extraction de données de devis de travaux BTP français.
Ces devis sont émis par QUALIDAL (13 avenue du Parc Alata, 60100 Creil).
Tu reçois le TEXTE BRUT extrait d'un PDF de devis (via pdfplumber).
Retourne UNIQUEMENT un JSON valide, sans texte ni markdown autour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMENT LIRE LE TEXTE EXTRAIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Le texte provient d'un tableau PDF avec colonnes : Description | Qté | U | P.U. HT | Montant HT | TVA

Chaque ITEM est identifiable par une ligne contenant des CHIFFRES à la fin :
  "Réparation épaufrures. 28,00 ML 120,00 3 360,00 20,00"
  → C'est un item : designation="Réparation épaufrures", Qté=28, U=ML, P.U.HT=120, Montant=3360

Les lignes SANS chiffres qui suivent = la DESCRIPTION de l'item précédent :
  "Sciage de part et d'autre de l'épaufrure sur largeur requise..."
  → Description de "Réparation épaufrures"

Les lignes avec "0,00 0,00 0,00 0,00" = séparateurs, titres de section, ou CGV → PAS d'items.

ATTENTION aux espaces dans les nombres :
  "1 022,00" = 1022.0 (espace = séparateur milliers)
  "16 750,00" = 16750.0
  "1 500,00" = 1500.0
  "1 700" = 1700.0 (pas de décimale → .0)
  "156 878" = 156878.0 (pas de décimale → .0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 1 — vendor_name
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Entreprise CLIENTE (pas Qualidal). Chercher après "Monsieur/Madame" → ligne suivante.
Sinon : première ligne en majuscules après le bloc Qualidal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 2 — project_name
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Valeur de la ligne "Chantier" dans le tableau récapitulatif.
Si courte, compléter avec "Adresse du chantier" ou "Ref Cde".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 3 — invoice_number
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Chercher "DE" + 4-10 chiffres → format "devis_de" + 8 chiffres min.
DE00005612 → "devis_de00005612"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 4 — date
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Colonne "Date" du tableau récapitulatif → AAAA-MM-JJ. Si absente : "".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 5 — LINE ITEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Un item valide = une ligne avec P.U. HT ≠ 0 OU Montant HT ≠ 0.

NE PAS créer d'item pour :
  • Lignes "0,00 0,00 0,00 0,00" (séparateurs/titres/CGV)
  • Sous-totaux, totaux, CGV, "BON POUR ACCORD"
  • Titres de section sans prix (ex: "MISSION CAPACITÉ DE CHARGE..." avec 0,00)

── designation ──
NOM COURT (4-8 mots). Première ligne de l'item, sans la description technique.

CELLULES : si le texte contient "Cellule X - NOM :" avec 0,00 (titre de section),
les items suivants doivent inclure " - Cellule X NOM" dans leur designation.
Ex: sous "Cellule 3 -ZIEGLER :" → "Reprise ancrages - Cellule 3 ZIEGLER"

── description ──
La description = le TEXTE COMPLET de la cellule pour cet item, y compris la ligne du titre.

MÉTHODE :
  1. Prendre la ligne qui contient les chiffres (Qté/P.U. HT) → extraire le texte AVANT les chiffres
  2. Prendre toutes les lignes en dessous JUSQU'À la prochaine ligne avec des chiffres
  3. Concaténer le tout = description complète

Exemple :
  "Rectification de planéité par ponçage Laser Grinder 1.00 FORF 192 940,00 ..."
  "sur 3 voies de roulement (3 x 400 mm)"
  "858 mètres linéaires d'allées étroites (13 allées de 66 m)"
  "Conformité selon norme EN15620 DM2"
  → designation = "Rectification de planéité par ponçage Laser Grinder"
  → description = "Rectification de planéité par ponçage Laser Grinder sur 3 voies de roulement (3 x 400 mm) 858 mètres linéaires d'allées étroites (13 allées de 66 m) Conformité selon norme EN15620 DM2"

Autre exemple :
  "Réparation épaufrures. 134,00 ML 125,00 16 750,00 20,00"
  "Sciage de part et d'autre de l'épaufrure..."
  → designation = "Réparation épaufrures"
  → description = "Réparation épaufrures. Sciage de part et d'autre de l'épaufrure..."

Si AUCUN texte entre deux lignes chiffrées → description = "" (vide).
Cas typiques avec description vide : "AMENÉ ET REPLI DU MATÉRIEL", "Trie des déchets",
"Traitements des déchets", "TRAVAIL VENDREDI - SAMEDI - DIMANCHE".
Ne JAMAIS prendre le texte d'un item voisin.
Copier la description ENTIÈRE sans tronquer.

── quantity, unite, unit_price ──
Lire les chiffres sur la ligne de l'item.
Espaces = séparateurs de milliers : "1 022,00" = 1022.0 | "1 700" = 1700.0 | "156 878" = 156878.0
Virgule = décimal : "4,60" = 4.6
Unités : FORF/ML/M2/UNIT→U/Heures/Jours/Semaine

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 6 — totals
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Chercher la ligne "Total HT ... Total TVA ... Total TTC" → extraire les valeurs.
Appliquer les mêmes règles de format numérique.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DE SORTIE — JSON STRICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "metadata": {
    "vendor_name": "...",
    "project_name": "...",
    "invoice_number": "devis_deXXXXXXXX",
    "date": "YYYY-MM-DD",
    "currency": "EUR"
  },
  "line_items": [
    {
      "designation": "...",
      "description": "...",
      "quantity": 0.0,
      "unite": "...",
      "unit_price": 0.0
    }
  ],
  "totals": {
    "subtotal_ht": 0.0,
    "total_tax": 0.0,
    "total_ttc": 0.0
  }
}"""


# ═══════════════════════════════════════════════════════
# EXTRACTION HYBRIDE — V16.2 : gpt-5.4-mini partout
# ═══════════════════════════════════════════════════════

async def extract_with_text(
    pdf_bytes: bytes,
    openai_client: AsyncOpenAI
) -> tuple[InvoiceData, str]:
    """
    V16.2 HYBRIDE : extrait le texte via pdfplumber puis l'envoie à gpt-5.4-mini.
    Fallback sur extract_with_vision si pdfplumber échoue.
    """
    t0 = time.time()

    # Étape 1 : extraire le texte
    pdf_text = extract_text_from_pdf(pdf_bytes)
    t1 = time.time()
    logger.info(f"[perf] pdfplumber: {t1-t0:.1f}s | {len(pdf_text)} chars")

    # Si pdfplumber retourne peu de texte → fallback vision
    if len(pdf_text) < 100:
        logger.warning("[text] Texte trop court, fallback vision")
        return await extract_with_vision(pdf_bytes, openai_client)

    model = get_model()

    # max_tokens adaptatif : gros devis → plus de place pour la réponse
    max_tok = 8000 if len(pdf_text) > 5000 else 4000
    logger.info(f"[text] Mode texte → {model} | {len(pdf_text)} chars (~{len(pdf_text)//4} tokens) | max_tokens={max_tok}")

    last_error = None
    for attempt in range(1, 3):
        try:
            resp = await openai_client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tok,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                    {"role": "user",   "content": f"Voici le texte extrait du devis Qualidal :\n\n{pdf_text}"}
                ]
            )
        except Exception as e:
            logger.exception(f"[text] Erreur OpenAI tentative {attempt}")
            last_error = e
            continue

        finish_reason = resp.choices[0].finish_reason
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE).strip()

        if finish_reason == "length":
            logger.warning(f"[text] Tentative {attempt} tronquée. Retry...")
            last_error = Exception("Réponse tronquée")
            continue

        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            logger.error(f"[text] Tentative {attempt} JSON invalide: {e}\n{raw[:400]}")
            last_error = e
            continue
    else:
        raise HTTPException(status_code=500, detail=f"Extraction texte échouée après 2 tentatives: {last_error}")

    # Sécurité invoice_number
    inv = data.get("metadata", {}).get("invoice_number", "")
    m = re.search(r'DE(\d{4,10})', inv, re.IGNORECASE)
    if m:
        data["metadata"]["invoice_number"] = f"devis_de{m.group(1).zfill(8)}"

    n_items = len(data.get("line_items", []))
    t2 = time.time()
    logger.info(f"[perf] OpenAI: {t2-t1:.1f}s | {model} | {n_items} item(s)")
    logger.info(f"[perf] TOTAL: {t2-t0:.1f}s (mode texte)")

    if n_items == 0:
        logger.warning("[text] ⚠ 0 items — tentative fallback vision")
        return await extract_with_vision(pdf_bytes, openai_client)

    try:
        return InvoiceData(**data), f"{model} (text)"
    except Exception as e:
        logger.error(f"[text] Pydantic: {e} — fallback vision")
        return await extract_with_vision(pdf_bytes, openai_client)


# ═══════════════════════════════════════════════════════
# PROMPT VISION  — V15 (conservé pour fallback)
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT_VISION = """\
Tu es un expert en extraction de données de devis de travaux BTP français.
Ces devis sont toujours émis par la société QUALIDAL (13 avenue du Parc Alata, 60100 Creil).
Tu reçois une ou plusieurs images de pages d'un même devis.
Retourne UNIQUEMENT un JSON valide, sans texte ni markdown autour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛔ RÈGLE ABSOLUE N°1 — NE PAS TRONQUER LES ITEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AVANT de commencer, compte visuellement TOUTES les lignes du tableau des
prestations qui ont un P.U. HT ≠ 0 ou un Montant HT ≠ 0.
Ce nombre est ton OBJECTIF. Tu dois produire EXACTEMENT ce nombre d'items.

Si le devis fait 3 pages, parcours les 3 pages entièrement.
Un item en bas de page 2 ou en haut de page 3 est AUSSI important qu'un item page 1.
Ne jamais s'arrêter avant la dernière ligne du tableau.

Items fréquemment manqués (sois EXTRA vigilant) :
  • Les portes et seuils de portes (souvent en fin de tableau)
  • Les bennes et évacuation de déchets
  • Les remises et ristournes (prix négatifs)
  • Les items en texte normal (non gras) sur un devis où les autres sont en gras
  • Les items de fin de zone (dernier item avant un séparateur ou sous-total)
  • Les "Amené et repli du matériel" et autres lignes courtes
  • Les impacts et petites réparations ponctuelles (1-2 unités)
  • Les éprouvettes et contrôles (souvent en fin de devis)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛔ RÈGLE ABSOLUE N°2 — ZÉRO HALLUCINATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tu dois UNIQUEMENT retranscrire ce qui est VISUELLEMENT PRÉSENT dans les images.
Il est STRICTEMENT INTERDIT d'inventer, compléter, déduire ou paraphraser du contenu.
Si un texte est partiellement illisible : recopie ce que tu vois, ne complète pas.
Ne jamais fusionner le contenu de deux cellules différentes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛔ RÈGLE ABSOLUE N°3 — FRONTIÈRE STRICTE ENTRE ITEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Le tableau des prestations a des colonnes : Description | Qté | U | P.U. HT | Montant HT | TVA
Chaque item = UNE RANGÉE (ou groupe de rangées) avec des CHIFFRES dans les colonnes Qté/P.U. HT.

⚠️ LA CLÉ : regarde les colonnes CHIFFRÉES (Qté, P.U. HT, Montant HT) pour délimiter les items.
  → Chaque fois que tu vois des chiffres ALIGNÉS dans les colonnes Qté + U + P.U. HT,
    c'est une NOUVELLE PRESTATION = un nouvel item.
  → Le texte dans la colonne Description qui est sur LA MÊME RANGÉE que ces chiffres
    (ou juste en dessous avant les prochains chiffres) = l'item avec sa description.

MÉTHODE DE LECTURE EN 3 PASSES :

PASSE 1 — INVENTAIRE : Parcours tout le tableau de haut en bas.
  À chaque fois que tu vois des chiffres dans les colonnes Qté + P.U. HT, note :
    "Item N : [texte de la ligne] — Qté=X, U=Y, P.U.HT=Z, Montant=W"
  C'est ta LISTE DE RÉFÉRENCE. Compte-les = nombre d'items attendus.

PASSE 2 — DESCRIPTION : Pour chaque item de ta liste :
  - Le TITRE = la ligne (souvent en gras) qui est sur la même rangée que les chiffres
  - La DESCRIPTION = le texte qui suit EN DESSOUS, JUSQU'À la rangée de l'item suivant
  - Si le prochain item (chiffres suivants) commence IMMÉDIATEMENT après le titre
    sans texte intermédiaire → description = "" (chaîne vide)

PASSE 3 — VÉRIFICATION : Ton nombre d'items JSON doit correspondre au comptage de la passe 1.

EXEMPLE VISUEL — Comment lire ce tableau :
  ┌─────────────────────────────────────┬───────┬──────┬─────────┬──────────┐
  │ AMENÉ ET REPLI DU MATÉRIEL - Zone 1 │  1,00 │ FORF │ 1 150,00│ 1 150,00 │ ← ITEM 1 (chiffres)
  │                                     │       │      │         │          │
  │ Préparation du support :            │  1,00 │ FORF │ 2 580,00│ 2 580,00 │ ← ITEM 2 (chiffres)
  │ Ponçage diamant de la surface...    │       │      │         │          │   (description item 2)
  │ Ponçage diamant manuel des bords... │       │      │         │          │   (suite description)
  └─────────────────────────────────────┴───────┴──────┴─────────┴──────────┘

  → Item 1 : "Amené et repli du matériel - Zone 1", description = "" car AUCUN texte
    entre cette ligne et la suivante qui a des chiffres (Préparation du support).
  → Item 2 : "Préparation du support", description = "Ponçage diamant de la surface..."
    car ce texte est SOUS la rangée de l'item 2 et AVANT l'item 3.

  ⚠️ ERREUR TYPIQUE À ÉVITER : attribuer "Ponçage diamant..." à "Amené et repli"
     simplement parce que le texte apparaît visuellement en dessous. NON !
     Il faut regarder les CHIFFRES : "Ponçage diamant" est dans la cellule
     de "Préparation du support" (même rangée de chiffres), pas celle d'Amené et repli.

INTERDIT :
  • Prendre le texte descriptif d'un item voisin pour remplir un item qui n'en a pas
  • Fusionner deux cellules de description adjacentes
  • Inventer une description quand la cellule n'en contient pas
  • Attribuer du texte à un item en se basant sur la proximité visuelle verticale
    SANS vérifier l'alignement avec les colonnes chiffrées

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛔ RÈGLE ABSOLUE N°4 — NOMBRES ET FORMATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Les devis français utilisent :
  • L'ESPACE comme séparateur de milliers : "1 560,00" = 1560.00, "10 000,00" = 10000.00
  • La VIRGULE comme séparateur décimal : "7,70" = 7.7
  • Parfois un point pour les milliers sur certains devis : "1.560,00" = 1560.00
  • Nombres SANS décimale : "1 700" = 1700.0, "156 878" = 156878.0

MÉTHODE pour convertir un prix ou quantité :
  1. Supprimer tous les espaces dans le nombre
  2. Remplacer la virgule par un point décimal
  3. Résultat = float
  Exemples : "1 560,00" → 1560.0 | "28,50" → 28.5 | "1 760,00" → 1760.0 | "1 700" → 1700.0 | "156 878" → 156878.0

VÉRIFICATION CROISÉE obligatoire :
  Si Qté × P.U. HT ≠ Montant HT (à 1€ près) → relire les chiffres plus attentivement.
  Exemples : 12 × 130 = 1560 ✓ | 18 × 63 = 1134 ✓ | 94 × 7.70 = 723.80 ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURE GÉNÉRALE D'UN DEVIS QUALIDAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Page 1 en-tête (haut gauche) : logo + adresse Qualidal
Page 1 en-tête (haut droite) : destinataire → "Monsieur/Madame Prénom NOM" puis nom entreprise
Tableau récapitulatif       : colonnes Chantier | Date | Date validité | Condition règlement | N° TVA
Tableau des prestations     : colonnes Description | Qté | U | P.U. HT | (R%) | Montant HT | TVA
Dernière page bas           : Totaux + "Adresse du chantier" + "BON POUR ACCORD"

Le tableau des prestations peut s'étendre sur plusieurs pages.
La colonne R% (remise) peut être absente.
Il peut y avoir des sous-totaux ou des séparateurs de zones (ex: "Zone 1", "Cellule A").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 1 — vendor_name (entreprise CLIENTE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C'est l'entreprise À QUI le devis est adressé. Jamais Qualidal.

Méthode principale :
  Repère "Monsieur", "Madame", "M.", "Mme" suivi d'un prénom et nom.
  La ligne SUIVANTE est le nom de l'entreprise cliente.
  Ex: "Monsieur Jean-Eudes Gohard" → ligne suivante → "IDEC"

Méthode fallback (si pas de civilité) :
  Après le bloc Qualidal (après "Email :" ou "Fax :"),
  chercher la première ligne en majuscules qui n'est pas :
  une adresse, un numéro de téléphone, "DEVIS", "FACTURE", ou une ville connue.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 2 — project_name (chantier)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Source primaire : valeur de la colonne "Chantier" dans le tableau récapitulatif.

Cas 1 — valeur explicite et complète :
  "AREFIM - REIMS (51)" → garder tel quel
  "LOZENNES (59)" → garder tel quel

Cas 2 — valeur courte ou nom propre seul (ex: "Autostore", "Amazon", "Lidl") :
  Chercher "Adresse du chantier" ou "Ref Cde" en bas de page pour compléter.
  Ex: Chantier="Autostore" + Adresse="Ussel" → "Autostore Ussel (19)"

Cas 3 — valeur vide :
  Utiliser "Ref Cde" si présent, sinon "INCONNU"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 3 — invoice_number
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cherche la référence commençant par "DE" suivie de 4 à 10 chiffres.
Format de sortie OBLIGATOIRE : "devis_de" + numéro en minuscules, 8 chiffres minimum.
  DE00004001 → "devis_de00004001"
  DE1898     → "devis_de00001898"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 4 — date
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Colonne "Date" du tableau récapitulatif (date d'émission, pas la date de validité).
Convertir JJ/MM/AAAA → AAAA-MM-JJ. Si absente : "".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 5 — LINE ITEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── Qu'est-ce qu'un item valide ? ──────────────────

Un item valide est une ligne du tableau qui remplit AU MOINS UNE condition :
  • P.U. HT est un nombre différent de 0 (positif ou négatif)
  • Montant HT est un nombre différent de 0 (positif ou négatif)
  • Qté est renseignée ET l'unité est renseignée (même si prix=0)

Ne PAS créer d'item pour :
  • Lignes entièrement à 0,00 (séparateurs visuels vides)
  • Sous-totaux ou totaux intermédiaires de zone
  • Titres de sous-section sans quantité ET sans prix
  • Conditions générales de vente
  • La ligne de total général (Total HT, Total TVA, Total TTC)

RÈGLE ABSOLUE — Le prix prime sur TOUT :
  Si P.U. HT ≠ 0 OU Montant HT ≠ 0 → item obligatoire, sans exception.
  • Même si la ligne commence par un tiret "- "
  • Même si le texte est court ("Impact", "Amené et repli du matériel")
  • Même si ce n'est PAS en gras
  • Même si c'est la dernière ligne avant les totaux
  Le style typographique (gras/normal/italique) n'est JAMAIS un critère d'exclusion.

── designation ────────────────────────────────────

NOM COURT de la prestation. 4 à 8 mots maximum.
- Texte court (une ligne) → prendre tel quel
- Texte long avec première ligne en gras → prendre uniquement cette première ligne
- Remise → "Remise exceptionnelle" ou libellé exact court
- Conserver l'identifiant de zone : "Grenaillage surface - Zone 1"
- Designation sans deux-points (ex: "Impact", "Benne") → valide comme les autres
- Parenthèse informative → designation = texte AVANT la parenthèse

RÈGLE CELLULES — quand le devis contient des sections "Cellule X - NOM" :
  Les lignes "Cellule 1 - THEBAULT :", "Cellule 2 - MARTIN :" etc. sont des
  TITRES DE SECTION (pas d'items). Les items qui suivent DOIVENT inclure le nom
  de la cellule dans leur designation.
  Format : "{prestation} - {Cellule X NOM}"

RÈGLE SPÉCIALE devis AMO / contrôle qualité :
  Ces devis ont souvent un titre en GRAS ou MAJUSCULES suivi d'un long bloc descriptif.
  Si le titre de section (ex: "CONTRÔLE QUALITÉ DALLAGE DE 6 600M²") a Qté=0 ET prix=0
  → NE PAS créer d'item pour ce titre.
  Les items SUIVANTS avec Qté + prix sont les vrais items à extraire.

── description ────────────────────────────────────

La description est le texte qui se trouve DANS LA MÊME RANGÉE du tableau
que les chiffres de l'item, ou dans les lignes ENTRE cet item et le SUIVANT.

MÉTHODE POUR DÉTERMINER LA DESCRIPTION D'UN ITEM :
  1. Identifie la rangée de l'item (celle avec Qté + U + P.U. HT)
  2. Regarde si le titre occupe toute la ligne ou s'il y a du texte descriptif
     sur cette même rangée (après le titre, avant les chiffres)
  3. Regarde les lignes EN DESSOUS : tout le texte JUSQU'À la prochaine rangée
     qui a des chiffres dans Qté/P.U. HT = description de CET item
  4. S'il n'y a AUCUN texte entre cette rangée et la prochaine rangée chiffrée
     → description = "" (chaîne vide)

PIÈGE PRINCIPAL — Items courts sans description :
  Certaines lignes n'ont QUE le titre + les chiffres, sans aucun texte descriptif.
  Le texte qui apparaît visuellement "en dessous" appartient en fait à l'ITEM SUIVANT.

RÈGLES STRICTES :
  1. Si la cellule contient UNIQUEMENT le titre SANS texte descriptif → description = ""
  2. Ne JAMAIS emprunter le texte d'une cellule voisine pour remplir une description vide
  3. La description se termine là où commence la RANGÉE SUIVANTE avec des chiffres
  4. Inclure : texte principal, sous-sections, tirets, normes, dimensions
  5. Exclure (CGV) : horaires de travail, fourniture eau/électricité, non-responsabilité
     Qualidal, mentions légales, conditions de paiement, délais, "BON POUR ACCORD"
  6. Pour une remise : description = "" (chaîne vide)
  7. Copier la description ENTIÈRE et FIDÈLEMENT — ne pas tronquer à mi-phrase

── quantity ───────────────────────────────────────

Valeur numérique de la colonne Qté. Toujours un float.
Si vide : 0.0
ATTENTION parenthèses : "Réparation seuil de porte : (2 unités à 4ml)   2,00"
→ quantity = 2.0 (la colonne Qté après ")"), PAS le "2" dans la parenthèse.

ATTENTION FORMAT : les espaces sont des séparateurs de milliers.
  "1 000,00" dans la colonne Qté → quantity = 1000.0 (pas 1.0)
  "28,50" → 28.5 | "1 700" → 1700.0 | "156 878" → 156878.0

── unite ──────────────────────────────────────────

  FORF / Forfait / FF / Ens / Ensemble  → "FORF"
  M2 / m² / M²                         → "M2"
  ML / ml / m / Lin                    → "ML"
  H / Heure / Heures / HR              → "Heures"
  J / Jour / Jours                     → "Jours"
  Sem / Semaine                        → "Semaine"
  U / unité / pce / pièce / UNIT       → "U"
  Vide ou non reconnu                  → "U"

── unit_price ─────────────────────────────────────

Valeur colonne P.U. HT. Peut être négatif (remise). Si vide : 0.0

ATTENTION FORMAT : les espaces sont des séparateurs de milliers.
  "1 760,00" → 1760.0 (pas 1.0 ni 760.0)
  "7,70" → 7.7 | "1 700" → 1700.0 | "156 878" → 156878.0

── Zones et sous-sections ─────────────────────────

Titre de zone = pas de valeur dans Qté/U/P.U. HT/Montant HT → NE PAS créer d'item.
Prestations dans une zone → items normaux avec nom de zone dans designation.
Sous-totaux de zone → NE PAS créer d'item.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 6 — totals
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  subtotal_ht : ligne "Total HT"
  total_tax   : ligne "Total TVA"
  total_ttc   : ligne "Total TTC" ou "Net à payer"
Si absent : 0.0

ATTENTION : appliquer les mêmes règles de format numérique (espaces = milliers, "1 700" = 1700.0).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DE SORTIE — JSON STRICT, SANS MARKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "metadata": {
    "vendor_name": "...",
    "project_name": "...",
    "invoice_number": "devis_deXXXXXXXX",
    "date": "YYYY-MM-DD",
    "currency": "EUR"
  },
  "line_items": [
    {
      "designation": "...",
      "description": "...",
      "quantity": 0.0,
      "unite": "...",
      "unit_price": 0.0
    }
  ],
  "totals": {
    "subtotal_ht": 0.0,
    "total_tax": 0.0,
    "total_ttc": 0.0
  }
}"""


# ═══════════════════════════════════════════════════════
# EXTRACTION VISION  — fallback si pdfplumber échoue
# ═══════════════════════════════════════════════════════

async def extract_with_vision(
    pdf_bytes: bytes,
    openai_client: AsyncOpenAI
) -> tuple[InvoiceData, str]:
    """
    Retourne (InvoiceData, model_used).
    Fallback vision : convertit le PDF en images puis envoie à gpt-5.4-mini.
    """
    t0 = time.time()

    try:
        images_b64 = pdf_bytes_to_images_b64(pdf_bytes, dpi=150)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    t1 = time.time()
    n_pages = len(images_b64)
    img_sizes_kb = [len(b) * 3 / 4 / 1024 for b in images_b64]
    logger.info(f"[perf] PDF→images: {t1-t0:.1f}s | {n_pages} page(s) | {sum(img_sizes_kb):.0f} KB total")

    model = get_model()
    logger.info(f"[vision] fallback vision → {model}")

    content = [{
        "type": "text",
        "text": (
            f"Voici les {n_pages} page(s) d'un devis Qualidal. "
            "IMPORTANT : parcours TOUTES les pages et extrait TOUS les items avec P.U. HT ≠ 0. "
            "Retourne UNIQUEMENT le JSON complet."
        )
    }]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}
        })

    last_error = None
    for attempt in range(1, 3):
        try:
            resp = await openai_client.chat.completions.create(
                model=model,
                max_completion_tokens=8000,,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_VISION},
                    {"role": "user",   "content": content}
                ]
            )
        except Exception as e:
            logger.exception(f"[vision] Erreur OpenAI tentative {attempt}")
            last_error = e
            continue

        finish_reason = resp.choices[0].finish_reason
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE).strip()

        if finish_reason == "length":
            logger.warning(f"[vision] Tentative {attempt} tronquée (finish_reason=length). Retry...")
            last_error = Exception("Réponse tronquée par max_tokens")
            continue

        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            logger.error(f"[vision] Tentative {attempt} JSON invalide: {e}\n{raw[:400]}")
            last_error = e
            continue
    else:
        raise HTTPException(status_code=500, detail=f"Extraction échouée après 2 tentatives: {last_error}")

    # Sécurité invoice_number
    inv = data.get("metadata", {}).get("invoice_number", "")
    m   = re.search(r'DE(\d{4,10})', inv, re.IGNORECASE)
    if m:
        data["metadata"]["invoice_number"] = f"devis_de{m.group(1).zfill(8)}"

    n_items = len(data.get("line_items", []))
    t2 = time.time()
    logger.info(f"[perf] OpenAI API: {t2-t1:.1f}s | {model} | {n_items} item(s)")
    logger.info(f"[perf] TOTAL extract_with_vision: {t2-t0:.1f}s")
    logger.info(f"[vision] ✓ {model} — {n_items} item(s) extraits")
    if n_items == 0:
        logger.warning("[vision] ⚠ AUCUN item extrait — vérifier le PDF")

    try:
        return InvoiceData(**data), f"{model} (vision)"
    except Exception as e:
        logger.error(f"[vision] Pydantic: {e}")
        raise HTTPException(status_code=500, detail=f"Structure IA invalide: {e}")


# ═══════════════════════════════════════════════════════
# APP & ROUTES
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Invoice Extraction API", version="16.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_openai_client() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY manquant")
    return AsyncOpenAI(api_key=key)


async def upload_pdf_to_bubble(pdf_bytes: bytes, filename: str) -> str:
    bubble_token = os.getenv("BUBBLE_API_KEY", "")
    bubble_base  = os.getenv("BUBBLE_BASE_URL", "https://www.portail-qualidal.com/version-test")
    upload_url   = f"{bubble_base}/api/1.1/files/uploadprivate"

    if not bubble_token:
        logger.warning("BUBBLE_API_KEY manquant — upload PDF ignoré")
        return ""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                upload_url,
                headers={"Authorization": f"Bearer {bubble_token}"},
                files={"file": (filename, pdf_bytes, "application/pdf")},
            )
            resp.raise_for_status()
            file_url = resp.json().get("fileUrl", "")
            logger.info(f"PDF uploadé vers Bubble : {file_url[:80]}...")
            return file_url
    except Exception as e:
        logger.warning(f"Upload PDF Bubble échoué : {e}")
        return ""


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "16.2.0",
        "mode": "hybrid (text first, vision fallback)",
        "model_default": DEFAULT_MODEL,
        "model_override": os.getenv("OPENAI_MODEL", "non défini — gpt-5.4-mini par défaut"),
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        filename  = file.filename or "devis.pdf"
        data, model_used = await extract_with_text(pdf_bytes, get_openai_client())
        file_url  = await upload_pdf_to_bubble(pdf_bytes, filename)
        del pdf_bytes; gc.collect()
        return ExtractionResponse(success=True, data=data, file_url=file_url, model_used=model_used)
    except HTTPException as e:
        return ExtractionResponse(success=False, error=e.detail)
    except Exception as e:
        logger.exception("Error /extract")
        return ExtractionResponse(success=False, error=str(e))
