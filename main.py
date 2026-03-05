"""
Invoice/Quote PDF Extraction API - V14.0
=========================================
Modifications vs V13 :
  1. Remplacement de get_model(n_pages) par choose_model(pdf_bytes) :
       - Analyse le CONTENU du PDF (ratio texte/prix, mots-clés AMO, longueur lignes)
       - Détecte automatiquement les devis complexes (AMO, contrôle qualité, prestations intellectuelles)
       - Score sur 7 points → gpt-4o si score ≥ 3, sinon gpt-4o-mini
       - Fallback sur gpt-4o-mini si erreur d'analyse
       - Override toujours possible via OPENAI_MODEL sur Render
  2. Log détaillé du score et des raisons de sélection (diagnostic)
  3. Tous les autres paramètres V13 conservés :
       - max_tokens=8000, dpi=200, detail=high, retry si tronqué
       - model_used retourné dans la réponse

Logique choose_model (score sur 8 points) :
  • nb pages ≥ 3             → +2 pts  (devis long, risque items manqués en fin de tableau)
  • ratio texte/prix > 1500  → +2 pts  (beaucoup de texte pour peu de lignes tarifées)
  • longueur moy. ligne > 60 → +1 pt   (lignes descriptives longues)
  • ≥ 2 mots-clés AMO        → +2 pts  (AMO, DOE, OPR, CCTP, EN15620, etc.)
  • ≤ 4 prix ET > 2000 chars → +1 pt   (peu d'items mais gros texte)
  Score ≥ 3 → gpt-4o | Score < 3 → gpt-4o-mini
"""

import os, io, logging, base64, re, gc, json
import httpx
from typing import Optional

import pdfplumber
from pypdf import PdfReader, PdfWriter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# SÉLECTION DU MODÈLE  — V14 : analyse contenu PDF
# ═══════════════════════════════════════════════════════

# Mots-clés caractéristiques des devis AMO / contrôle qualité
# (prestations intellectuelles avec gros blocs texte descriptif)
_AMO_KEYWORDS = [
    "AMO",
    "phase exe",
    "phase exé",
    "OPR",
    "DOE",
    "CCTP",
    "EN15620",
    "profilographe",
    "planéité",
    "réunion préparatoire",
    "dossier béton",
    "contrôle qualité",
    "bureau de contrôle",
    "pièces marchés",
    "prédimensionnement",
    "pré-dimensionnement",
    "étude préliminaire",
    "analyse des offres",
    "rapport journalier",
    "visite après coulage",
    "analyse CCTP",
]


# ═══════════════════════════════════════════════════════
# PATCH choose_model() — V15
# ═══════════════════════════════════════════════════════
#
# CHANGEMENTS vs V14 :
#   1. Nouveau critère 6 : "densité descriptive" — détecte les devis avec
#      des blocs de texte longs entre les lignes de prix (comme le devis
#      XPO Aéroparc "Système Semi Lisse"). Ces devis sont difficiles pour
#      gpt-4o-mini car les frontières entre items ne sont pas visuellement
#      claires.
#   2. Score max passe de 8 à 10 points, seuil reste à ≥ 2
#   3. Log amélioré avec toutes les métriques
#
# Pour appliquer : remplacer la fonction choose_model() dans main.py
# ═══════════════════════════════════════════════════════

def choose_model(pdf_bytes: bytes) -> str:
    """
    Sélection intelligente du modèle OpenAI selon la COMPLEXITÉ du devis.

    Analyse 6 métriques objectives extraites du texte pdfplumber :
      1. nb_pages          : nombre de pages
      2. ratio_texte_prix  : caractères totaux / nb lignes avec prix
      3. avg_line_length   : longueur moyenne des lignes
      4. keyword_hits      : nb de mots-clés AMO détectés
      5. lignes_vs_chars   : peu de prix pour beaucoup de texte
      6. desc_density      : nb moyen de lignes de texte entre deux lignes de prix
                             (détecte les devis avec descriptions denses)

    Score sur 10 → gpt-4o si score ≥ 2, sinon gpt-4o-mini.
    Override via env var OPENAI_MODEL.
    """
    # Override manuel toujours prioritaire
    override = os.getenv("OPENAI_MODEL", "").strip()
    if override in ("gpt-4o", "gpt-4o-mini"):
        logger.info(f"[choose_model] override env → {override}")
        return override

    try:
        full_text    = ""
        total_chars  = 0
        total_lines  = 0
        lines_with_prices = 0
        # V15 : comptage des lignes de texte entre les lignes de prix
        text_lines_between_prices = []  # liste des gaps entre lignes prix
        current_gap = 0

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                total_lines += len(lines)

                for line in lines:
                    total_chars += len(line)
                    # Ligne avec prix = contient un nombre type "1 500,00" ou "27 500,00"
                    if re.search(r'\d[\d\s]{2,},\d{2}', line):
                        lines_with_prices += 1
                        # V15 : enregistrer le gap depuis la dernière ligne prix
                        if current_gap > 0:
                            text_lines_between_prices.append(current_gap)
                        current_gap = 0
                    else:
                        current_gap += 1

        # ── Métriques ──────────────────────────────────────────
        ratio_texte_prix = total_chars / max(lines_with_prices, 1)
        avg_line_length  = total_chars / max(total_lines, 1)

        # V15 : densité descriptive = nombre moyen de lignes de texte entre les lignes de prix
        # Devis simple : ~0-1 (juste titre + chiffres)
        # Devis dense  : ~3-6 (titre + description multi-lignes + chiffres)
        avg_desc_density = (
            sum(text_lines_between_prices) / len(text_lines_between_prices)
            if text_lines_between_prices else 0
        )
        max_desc_gap = max(text_lines_between_prices) if text_lines_between_prices else 0

        # Comptage mots-clés AMO
        full_text_lower = full_text.lower()
        keyword_hits = sum(
            1 for kw in _AMO_KEYWORDS
            if kw.lower() in full_text_lower
        )

        # ── Score ───────────────────────────────────────────────
        score   = 0
        reasons = []

        # Critère 1 : 3 pages ou plus → risque d'items manqués
        if n_pages >= 3:
            score += 2
            reasons.append(f"nb_pages={n_pages}>=3 (+2)")

        # Critère 2 : beaucoup de texte pour peu de lignes tarifées
        if ratio_texte_prix > 1500:
            score += 2
            reasons.append(f"ratio_texte_prix={ratio_texte_prix:.0f}>1500 (+2)")

        # Critère 3 : lignes longues = descriptions denses
        if avg_line_length > 60:
            score += 1
            reasons.append(f"avg_line_length={avg_line_length:.0f}>60 (+1)")

        # Critère 4 : mots-clés AMO
        if keyword_hits >= 2:
            score += 2
            reasons.append(f"keyword_hits={keyword_hits}>=2 (+2)")
        elif keyword_hits == 1:
            score += 1
            reasons.append(f"keyword_hits={keyword_hits}=1 (+1)")

        # Critère 5 : très peu d'items pour un PDF dense
        if lines_with_prices <= 4 and total_chars > 2000:
            score += 1
            reasons.append(f"peu_de_prix={lines_with_prices}<=4 (+1)")

        # Critère 6 (V15) : densité descriptive élevée
        # Si en moyenne il y a 3+ lignes de texte entre chaque ligne de prix,
        # OU si un bloc dépasse 5 lignes, c'est un devis avec descriptions denses
        # que gpt-4o-mini risque de mal découper.
        if avg_desc_density >= 3:
            score += 2
            reasons.append(f"avg_desc_density={avg_desc_density:.1f}>=3 (+2)")
        elif max_desc_gap >= 5:
            score += 1
            reasons.append(f"max_desc_gap={max_desc_gap}>=5 (+1)")

        # ── Décision ────────────────────────────────────────────
        model = "gpt-4o" if score >= 2 else "gpt-4o-mini"

        logger.info(
            f"[choose_model] score={score}/10 → {model} | "
            f"pages={n_pages} lines_prix={lines_with_prices} "
            f"avg_desc_density={avg_desc_density:.1f} max_gap={max_desc_gap} | "
            f"raisons: {' | '.join(reasons) if reasons else 'devis standard'}"
        )

        return model

    except Exception as e:
        logger.warning(f"[choose_model] erreur analyse PDF → fallback gpt-4o-mini : {e}")
        return "gpt-4o-mini"


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
    description:  str  = Field(
        default="",
        description=(
            "Texte COMPLET de la cellule Description pour cet item. "
            "Inclure: sous-sections, tirets, détails techniques, normes. "
            "EXCLURE CGV: 'Travaux réalisés en semaine...', "
            "'Fourniture eau/électricité...', 'QUALIDAL n est pas responsable...'"
        )
    )
    quantity:     float = Field(..., description="Quantité")
    unite:        str   = Field(..., description="Unité (ML, M2, FORF, U, etc.)")
    unit_price:   float = Field(..., description="Prix unitaire HT")

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

class SplitResult(BaseModel):
    file_name:  str
    pdf_base64: str

class SplitResponse(BaseModel):
    success:     bool
    total_files: int               = Field(default=0)
    results:     list[SplitResult] = Field(default=[])
    error:       Optional[str]     = None

class SplitLightItem(BaseModel):
    file_name:      str
    pdf_base64:     str
    vendor_name:    str
    project_name:   str
    invoice_number: str
    drive_path:     str

class SplitLightResponse(BaseModel):
    success:     bool
    total_files: int                  = Field(default=0)
    results:     list[SplitLightItem] = Field(default=[])
    errors:      list[dict]           = Field(default=[])
    error:       Optional[str]        = None

class SplitExtractItem(BaseModel):
    file_name:  str
    pdf_base64: str
    extraction: InvoiceData
    model_used: Optional[str] = None

class SplitExtractError(BaseModel):
    file_name:  str
    error:      str
    page_start: int
    page_end:   int

class SplitExtractResponse(BaseModel):
    success:         bool
    total_found:     int                     = Field(default=0)
    total_extracted: int                     = Field(default=0)
    total_errors:    int                     = Field(default=0)
    results:         list[SplitExtractItem]  = Field(default=[])
    errors:          list[SplitExtractError] = Field(default=[])


# ═══════════════════════════════════════════════════════
# PDF → IMAGES BASE64
# ═══════════════════════════════════════════════════════

def pdf_bytes_to_images_b64(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
    """
    Chaque page du PDF → JPEG base64.
    DPI 200 pour meilleure lisibilité des tableaux serrés.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            buf = io.BytesIO(pix.tobytes("jpeg", jpg_quality=92))
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
            img.save(buf, format="JPEG", quality=92)
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
# DÉCOUPAGE PDF
# ═══════════════════════════════════════════════════════

def split_pdf_into_parts(pdf_bytes: bytes) -> list[dict]:
    split_points, devis_names = [], []
    reader_scan  = PdfReader(io.BytesIO(pdf_bytes))
    total_pages  = len(reader_scan.pages)

    for i in range(total_pages):
        try:
            txt   = reader_scan.pages[i].extract_text() or ""
            match = re.search(r"DE\d{4,10}", txt, re.IGNORECASE)
            if match:
                split_points.append(i)
                devis_names.append(f"devis_{match.group(0).lower()}")
        except Exception:
            continue

    if not split_points or split_points[0] != 0:
        split_points.insert(0, 0)
        devis_names.insert(0, "devis_inconnu")
    split_points.append(total_pages)
    del reader_scan; gc.collect()

    reader_write = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for i in range(len(split_points) - 1):
        s, e = split_points[i], split_points[i + 1]
        if s >= e:
            continue
        writer = PdfWriter()
        for j in range(s, e):
            writer.add_page(reader_write.pages[j])
        buf = io.BytesIO()
        writer.write(buf)
        parts.append({
            "file_name":  f"{devis_names[i]}.pdf",
            "pdf_bytes":  buf.getvalue(),
            "page_start": s + 1,
            "page_end":   e,
        })
        buf.close(); del writer

    del reader_write; gc.collect()
    return parts


# ═══════════════════════════════════════════════════════
# REGEX METADATA
# ═══════════════════════════════════════════════════════

def extract_metadata_regex(pdf_bytes: bytes, file_name: str) -> dict:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
    except Exception as ex:
        logger.warning(f"Erreur lecture {file_name}: {ex}")
        return {"vendor_name": "INCONNU", "project_name": "INCONNU",
                "invoice_number": file_name.replace(".pdf", "")}
    finally:
        gc.collect()

    lines = text.split("\n")

    m = re.search(r"(DE\d{4,10})", text, re.IGNORECASE)
    invoice_number = f"devis_{m.group(1).lower()}" if m else file_name.replace(".pdf", "")

    vendor_name = "INCONNU"
    for i, line in enumerate(lines):
        if re.match(r"^(Monsieur|Madame|M\.|Mme)\s+", line.strip(), re.IGNORECASE):
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not re.match(r"^\d", nxt):
                    vendor_name = nxt
                    break
    if vendor_name == "INCONNU":
        found = False
        for line in lines:
            if "email" in line.lower() or "fax" in line.lower():
                found = True; continue
            if found and line.strip() and line.strip().isupper() and len(line) > 3 and "DEVIS" not in line:
                vendor_name = line.strip(); break

    project_name = "INCONNU"
    for i, line in enumerate(lines):
        if "Chantier" in line:
            for offset in range(1, 6):
                if i + offset >= len(lines): break
                nxt = lines[i + offset].strip()
                if not nxt: continue
                if any(x in nxt.lower() for x in ["de l'offre", "validité"]): continue
                dm = re.search(r"(\d{2}/\d{2}/\d{4})", nxt)
                if dm:
                    project_name = nxt[:dm.start()].replace("|", "").strip()
                    break
            if project_name != "INCONNU": break

    for attr in [vendor_name, project_name]:
        attr = re.sub(r"[,;.\s]+$", "", attr)[:100]
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        vendor_name  = vendor_name.replace(c, '-')
        project_name = project_name.replace(c, '-')

    return {"vendor_name": vendor_name, "project_name": project_name,
            "invoice_number": invoice_number}


# ═══════════════════════════════════════════════════════
# PROMPT VISION  (inchangé depuis V12/V13)
# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════
# PROMPT VISION V15 — Corrections :
#   1. Règle "Amené et repli" : description vide si pas de texte descriptif
#   2. Règle "Cellule X - NOM" : inclure dans designation
#   3. Règle formats numériques : 1 000 = 1000, virgule décimale
#   4. Règle anti-mélange descriptions : frontière stricte entre items
#   5. Nouveaux exemples few-shot couvrant ces cas
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

MÉTHODE pour convertir un prix ou quantité :
  1. Supprimer tous les espaces dans le nombre
  2. Remplacer la virgule par un point décimal
  3. Résultat = float
  Exemples : "1 560,00" → "1560,00" → 1560.0 | "28,50" → 28.5 | "1 760,00" → 1760.0

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
  Exemples :
    Item "Réparation épaufrures" sous "Cellule 1 - THEBAULT"
    → designation = "Réparation épaufrures - Cellule 1 THEBAULT"
    Item "Nettoyage joints construction plat" sous "Cellule 1 - THEBAULT"
    → designation = "Nettoyage joints construction plat - Cellule 1 THEBAULT"
    Item "Réparation épaufrures" sous "Cellule 2 - THEBAULT"
    → designation = "Réparation épaufrures - Cellule 2 THEBAULT"

  Cela permet de distinguer des items identiques dans des cellules différentes.
  La cellule courante reste active jusqu'au prochain titre "Cellule X".

RÈGLE SPÉCIALE devis AMO / contrôle qualité :
  Ces devis ont souvent un titre en GRAS ou MAJUSCULES suivi d'un long bloc descriptif.
  Si le titre de section (ex: "CONTRÔLE QUALITÉ DALLAGE DE 6 600M²") a Qté=0 ET prix=0
  → NE PAS créer d'item pour ce titre.
  Les items SUIVANTS avec Qté + prix sont les vrais items à extraire.
  Exemples de designations courtes attendues :
    "Etude préliminaire des pièces marchés..." (1,00 FORF 1500€) → "Étude préliminaire phase conception"
    "En phase Exe:" (1,00 FORF 5240€) → "Mission phase exécution"
    "1 visite après coulage en phase OPR" (1,00 FORF 821€) → "Visite OPR après coulage"

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
  
  Exemples de lignes qui ont TOUJOURS ou presque une description VIDE :
    • "AMENÉ ET REPLI DU MATÉRIEL" → description = ""
    • "Trie des déchets" → description = ""
    • "TRAVAIL VENDREDI - SAMEDI - DIMANCHE" → description = ""
    • "Remise exceptionnelle" → description = ""
  
  Pour vérifier : est-ce que le texte en dessous commence par un NOUVEAU TITRE
  qui a ses propres chiffres Qté/P.U. HT ? Si oui → ce texte ne lui appartient pas.

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
  "28,50" → 28.5

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
  "7,70" → 7.7

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

ATTENTION : appliquer les mêmes règles de format numérique (espaces = milliers).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLES RÉELS — FEW-SHOT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── A — Titre de section → IGNORÉ ──────────────────
"LTM/WARNING   0,00   0,00   0,00" → NE PAS créer d'item (tout à 0)

── B — Item simple MAJUSCULES ─────────────────────
"ÉPROUVETTES SUPPLÉMENTAIRES   28,00   UNIT   22,00   616,00"
→ { "designation": "Éprouvettes supplémentaires", "description": "3 supplémentaires en compression (à 28 jours) x 7 jours. 1 supplémentaires en fendage (à 28 jours) x 7 jours", "quantity": 28.0, "unite": "U", "unit_price": 22.0 }

── C — Description longue ─────────────────────────
"Réparation épaufrures :   80,00   ML   125,00   10 000,00"
→ { "designation": "Réparation épaufrures", "description": "Sciage de part et d autre de l épaufrure sur largeur requise et profondeur de 15 mm, piquage, nettoyage, aspiration, application d un primaire d accrochage et application d un mortier de résine sans retrait. Surfaçage et reconstitution du joint, le cas échéant.", "quantity": 80.0, "unite": "ML", "unit_price": 125.0 }

── D — Description listes à tirets ────────────────
"Bords des regards :   2,00   UNIT   525,00   1 050,00"
→ { "designation": "Bords des regards", "description": "- Sciage périphérique autour du tampon et du regard - Piochage et évacuation des gravats - Recalage du tampon - Scellement du tampon par mortier de résine - Reprise de l ensemble au mortier de résine", "quantity": 2.0, "unite": "U", "unit_price": 525.0 }

── E — Description très longue multi-sections ──────
"Mise en conformité de la dalle...   815,00   M2   102,00   83 130,00"
→ { "designation": "Mise en conformité dalle coulis hydraulique Autostore", "description": "Mise en conformité de la dalle, par application d un coulis hydraulique conformément aux spécifications AUTOSTORE. 800m². Installation et livraison chantier : - Amené et repli du matériel - Installation de barrières et bâches de protection - Nettoyage de la zone en fin de chantier. Préparation de surface : - Grenaillage de la surface - Application d un primaire d accrochage sablé - Installation de coffrages et traitement des trous, impacts, joints et fissures - Mise en place de boulons réglés à +/- 0.5 mm respectant un maillage de 1.5 x 1.5 m². Application coulis hydraulique : - transport et mise en oeuvre du matériau par camion pompe - Réglage et finition conformément à la norme NF EN 13813, de type P4S avec résistance à l abrasion ARO.5. Contrôles de réception : - Scanner 3D de la surface - Essai de friction / glissance - Essai résistivité électrique", "quantity": 815.0, "unite": "M2", "unit_price": 102.0 }

── F — Remise prix négatif ─────────────────────────
"Remise exceptionnelle   1,00   FORF   -130,00   -130,00"
→ { "designation": "Remise exceptionnelle", "description": "", "quantity": 1.0, "unite": "FORF", "unit_price": -130.0 }

── G — Item texte NORMAL (non gras) — OBLIGATOIRE ──
Sur un devis où tous les autres items sont en GRAS :
"Fourniture d'une benne...   1,00   FORF   900,00   900,00"
→ ITEM OBLIGATOIRE car P.U. HT = 900 ≠ 0. Le style n'est pas un critère.
→ { "designation": "Fourniture benne évacuation déchets", "description": "Fourniture d'une benne pour l'évacuation et le traitement des déchets. Emplacement à convenir le plus proche possible de la zone de travail.", "quantity": 1.0, "unite": "FORF", "unit_price": 900.0 }

── H — Designation courte sans deux-points ─────────
"Impact   2,00   UNIT   120,00   240,00" (dans Cellule A2)
→ ITEM OBLIGATOIRE. "Impact" est une designation valide.
→ { "designation": "Impact - Cellule A2", "description": "Réparation en mortier de résine (30x30). Sciage périphérique de l épaufrure sur largeur et profondeur requise, piquage, nettoyage, aspiration, application d un primaire d accrochage et application d un mortier de résine sans retrait. Surfaçage final si nécessaire.", "quantity": 2.0, "unite": "U", "unit_price": 120.0 }

── I — Seuil de porte avec parenthèse ─────────────
"Réparation seuil de porte : (2 unités à 4ml)   2,00   UNIT   230,00   460,00"
⚠️ Le "2" dans la parenthèse n'est PAS la Qté. La vraie Qté est après ")".
→ { "designation": "Réparation seuil de porte - Cellule A3", "description": "(2 unités à 4ml). Sciage de part et d autre du seuil sur largeur et profondeur requise, piquage, nettoyage, aspiration, application d un primaire d accrochage et application d un mortier de résine sans retrait. Surfaçage final si nécessaire.", "quantity": 2.0, "unite": "U", "unit_price": 230.0 }

── J — Dates en tirets dans description ────────────
"SUIVI DE COULAGE - zone 1   7,00   FORF   459,00"
"Inclus slump tests... - 25/02/2026 - 26/02/2026..."
→ Les dates en tirets = partie de la DESCRIPTION, pas des items séparés.
→ { "designation": "Suivi de coulage - Zone 1", "description": "Inclus slump tests, 1 E/C, 3 tests de fibres... - 25/02/2026 - 26/02/2026 - 27/02/2026", "quantity": 7.0, "unite": "FORF", "unit_price": 459.0 }

── K — CGV à exclure ───────────────────────────────
"Acompte de 30% à la commande   0,00   0,00   0,00"
"Travaux réalisés en semaine du lundi au vendredi..."
"La société QUALIDAL n est pas responsable..."
→ Qté=0 + prix=0 + CGV → NE PAS créer d'items.

── L — Devis AMO : titre section à 0 + vrais items ─
"CONTRÔLE QUALITÉ DALLAGE de 6 600m²   —   —   0,00   0,00"
→ NE PAS créer d'item (prix = 0, c'est un titre de section)

"Etude préliminaire des pièces marchés...   1,00   FORF   1 500,00   1 500,00"
→ { "designation": "Étude préliminaire phase conception", "description": "Etude préliminaire des pièces marchés et 1ers échanges, analyse CCTP Lot Dallage VS Descriptif technique, analyse des offres techniques lot Dallage, analyse du dossier technique Dallage et prédimensionnement. Participation en visio à une mise au point technique lors de la consultation si nécessaire.", "quantity": 1.0, "unite": "FORF", "unit_price": 1500.0 }

"En phase Exe:   1,00   FORF   5 240,00   5 240,00"
→ { "designation": "Mission phase exécution", "description": "- Organisation d'une réunion préparatoire avec les lots concernés (géotechnique, dallage, VRD, G.O., Bureau de contrôle) - Analyse et vérification des derniers éléments avant coulage (Notes de calcul, dossier béton, plan EXE finalisé, essais béton, etc.) - Vérification des conditions de coulage après réglage final de la plateforme, lors de la préparation des coulages du lot Dallage - Présence d'une journée lors des coulages en Cellule 1 et en cellules 2 (soit 2 jours) - Rapport journalier - Analyse DOE en phase final avant réception, DOE à nous transmettre.", "quantity": 1.0, "unite": "FORF", "unit_price": 5240.0 }

"1 visite après coulage en phase OPR...   1,00   FORF   821,00   821,00"
→ { "designation": "Visite OPR après coulage", "description": "1 visite après coulage en phase OPR pour constat et réserves éventuelles.", "quantity": 1.0, "unite": "FORF", "unit_price": 821.0 }

"Plus-values :   —   —   0,00   0,00"
→ NE PAS créer d'item (prix = 0, texte informatif)

── M — AMENÉ ET REPLI — description VIDE ──────────
"AMENÉ ET REPLI DU MATÉRIEL - Zone 2   1,00   FORF   995,00   995,00"
→ Cette ligne n'a PAS de texte descriptif dans sa cellule, juste le titre.
→ { "designation": "Amené et repli du matériel - Zone 2", "description": "", "quantity": 1.0, "unite": "FORF", "unit_price": 995.0 }

⚠️ NE PAS mettre "Sciage de part et d'autre..." en description ici !
   Ce texte appartient à l'item SUIVANT (Réparation épaufrures).

── N — Items avec CELLULES — designation contextualisée ──
Section "Cellule 1 - THEBAULT :" (titre sans prix → pas d'item)
Puis :
"Réparation épaufrures.   1,00   ML   120,00   120,00"
"Sciage de part et d'autre de l'épaufrure sur largeur requise..."
→ { "designation": "Réparation épaufrures - Cellule 1 THEBAULT", "description": "Sciage de part et d'autre de l'épaufrure sur largeur requise et profondeur de 15 mm, piquage, nettoyage, aspiration, application d'un primaire d'accrochage et application d'un mortier de résine sans retrait. Surfaçage et reconstitution du joint, le cas échéant.", "quantity": 1.0, "unite": "ML", "unit_price": 120.0 }

"Réparation de fissure inférieure à 5 mm d'ouverture par calfeutrement   28,50   ML   25,00   712,50"
"avec un bi-composant epoxy."
→ { "designation": "Réparation fissure inférieure à 5 mm - Cellule 1 THEBAULT", "description": "Réparation de fissure inférieure à 5 mm d'ouverture par calfeutrement avec un bi-composant epoxy.", "quantity": 28.5, "unite": "ML", "unit_price": 25.0 }

Section "Cellule 2 - THEBAULT" (titre sans prix → pas d'item)
"Réparation épaufrures.   0,10   ML   120,00   12,00"
→ { "designation": "Réparation épaufrures - Cellule 2 THEBAULT", "description": "Sciage de part et d'autre de l'épaufrure sur largeur requise et profondeur de 15 mm, piquage, nettoyage, aspiration, application d'un primaire d'accrochage et application d'un mortier de résine sans retrait. Surfaçage et reconstitution du joint, le cas échéant.", "quantity": 0.1, "unite": "ML", "unit_price": 120.0 }

── O — Prix avec espace millier ────────────────────
"CAROTTE SUPPLÉMENTAIRE :   12,00   UNIT   130,00   1 560,00"
→ Vérification : 12 × 130 = 1560 ✓
→ { "designation": "Carotte supplémentaire", "description": "...", "quantity": 12.0, "unite": "U", "unit_price": 130.0 }

"Rédaction d'une note de calcul...   1,00   UNIT   1 760,00   1 760,00"
→ unit_price = 1760.0 (PAS 1.0 ni 760.0)
→ { "designation": "Rédaction note de calcul", "description": "...", "quantity": 1.0, "unite": "U", "unit_price": 1760.0 }

── P — Réparation divers Mortier — description propre ──
"Réparation divers Mortier de résine (10x10cm).   2,00   UNIT   45,00   90,00"
"Sciage périphérique de l'épaufrure sur largeur et profondeur requise,
piquage, nettoyage, aspiration, application d'un primaire d'accrochage
et application d'un mortier de résine sans retrait. Surfaçage final si nécessaire."
→ { "designation": "Réparation divers mortier résine 10x10 - Cellule 1 THEBAULT", "description": "Sciage périphérique de l'épaufrure sur largeur et profondeur requise, piquage, nettoyage, aspiration, application d'un primaire d'accrochage et application d'un mortier de résine sans retrait. Surfaçage final si nécessaire.", "quantity": 2.0, "unite": "U", "unit_price": 45.0 }

⚠️ La description de "Réparation divers Mortier (10x10)" est DIFFÉRENTE de
   celle de "Réparation divers Mortier (20x20)" même si elles se ressemblent.
   Chaque item a SA PROPRE cellule de description.

── Q — DEVIS COMPLET avec items SANS et AVEC description ──
Voici un devis type "Système Semi Lisse" avec 6 items. Attention aux frontières :

"AMENÉ ET REPLI DU MATÉRIEL - Zone 1   1,00   FORF   1 150,00   1 150,00"
"Préparation du support :               1,00   FORF   2 580,00   2 580,00"
"Ponçage diamant de la surface avec aspiration des poussières à la source."
"Ponçage diamant manuel des bords et des zones inaccessibles avec respiration à la source"
"Application du système :               1,00   FORF   9 480,00   9 480,00"
"Fourniture et application d'une couche d'égalisation à raison de 2,5kg/m² saupoudrée à refus de silice."
"Fourniture et application d'une couche de fond à raison de 2,5kg/m² saupoudrée à refus."
"Fourniture et application d'une couche de garnissage époxy à raison de 650gr/m²."
"Fourniture et application de relevés avec une gorge en mastic époxy dans l'angle sol/mur."
"Ponçage et aspiration de la zone entre chaque étape de l'intervention."
"Gorge + remonté de mur sur 1m:        46,00   ML     66,00     3 036,00"
"Mise en oeuvre de gorge périphérique au mortier de résine époxy"
"Remonté de mur en résine époxy teinte RAL 7040"
"Trie des déchets                        1,00   FORF   450,00      450,00"
"TRAVAIL VENDREDI - SAMEDI - DIMANCHE    1,00   FORF   2 220,00   2 220,00"
"Zone accessible à partir du lundi."

Résultat attendu — 6 items :
→ Item 1 : { "designation": "Amené et repli du matériel - Zone 1", "description": "", "quantity": 1.0, "unite": "FORF", "unit_price": 1150.0 }
  ⚠️ description="" car le texte "Préparation du support" en dessous est l'ITEM 2, pas une description.
→ Item 2 : { "designation": "Préparation du support", "description": "Ponçage diamant de la surface avec aspiration des poussières à la source. Ponçage diamant manuel des bords et des zones inaccessibles avec respiration à la source", "quantity": 1.0, "unite": "FORF", "unit_price": 2580.0 }
→ Item 3 : { "designation": "Application du système", "description": "Fourniture et application d'une couche d'égalisation à raison de 2,5kg/m² saupoudrée à refus de silice. Fourniture et application d'une couche de fond à raison de 2,5kg/m² saupoudrée à refus. Fourniture et application d'une couche de garnissage époxy à raison de 650gr/m². Fourniture et application de relevés avec une gorge en mastic époxy dans l'angle sol/mur. Ponçage et aspiration de la zone entre chaque étape de l'intervention.", "quantity": 1.0, "unite": "FORF", "unit_price": 9480.0 }
  ⚠️ La description inclut les 5 lignes de texte COMPLÈTES. Ne pas tronquer.
→ Item 4 : { "designation": "Gorge et remonté de mur sur 1m", "description": "Mise en oeuvre de gorge périphérique au mortier de résine époxy. Remonté de mur en résine époxy teinte RAL 7040", "quantity": 46.0, "unite": "ML", "unit_price": 66.0 }
→ Item 5 : { "designation": "Trie des déchets", "description": "", "quantity": 1.0, "unite": "FORF", "unit_price": 450.0 }
  ⚠️ description="" car aucun texte descriptif, "TRAVAIL VENDREDI..." est l'item suivant.
→ Item 6 : { "designation": "Travail vendredi - samedi - dimanche", "description": "", "quantity": 1.0, "unite": "FORF", "unit_price": 2220.0 }
  ⚠️ "Zone accessible à partir du lundi" = CGV, PAS une description. description=""

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
# EXTRACTION VISION  — V14 avec choose_model
# ═══════════════════════════════════════════════════════

async def extract_with_vision(
    pdf_bytes: bytes,
    openai_client: AsyncOpenAI
) -> tuple[InvoiceData, str]:
    """
    Retourne (InvoiceData, model_used).
    V14 : sélection automatique du modèle par analyse du contenu PDF.
    """
    try:
        images_b64 = pdf_bytes_to_images_b64(pdf_bytes, dpi=150)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    n_pages = len(images_b64)

    # V14 : choose_model analyse le contenu (remplace get_model basé sur nb pages)
    model = choose_model(pdf_bytes)
with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
    raw_text = "\n".join(p.extract_text() or "" for p in pdf.pages)

logger.info(f"[vision] {n_pages} page(s) → modèle sélectionné : {model}")

content = [{
    "type": "text",
    "text": (
        f"Voici les {n_pages} page(s) d'un devis Qualidal. "
        "IMPORTANT : parcours TOUTES les pages et extrait TOUS les items avec P.U. HT ≠ 0. "
        "Retourne UNIQUEMENT le JSON complet.\n\n"
        f"Texte brut extrait du PDF :\n{raw_text[:4000]}"
    )
}]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
        })

    last_error = None
    for attempt in range(1, 3):
        try:
            resp = await openai_client.chat.completions.create(
                model=model,
                max_tokens=8000,
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
    logger.info(f"[vision] ✓ {model} — {n_items} item(s) extraits")
    if n_items == 0:
        logger.warning("[vision] ⚠ AUCUN item extrait — vérifier le PDF")

    try:
        return InvoiceData(**data), model
    except Exception as e:
        logger.error(f"[vision] Pydantic: {e}")
        raise HTTPException(status_code=500, detail=f"Structure IA invalide: {e}")


# ═══════════════════════════════════════════════════════
# APP & ROUTES
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Invoice Extraction API", version="14.0.0")
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
        "version": "14.0.0",
        "model_selection": "auto (analyse contenu PDF : ratio texte/prix, mots-clés AMO, longueur lignes)",
        "model_override": os.getenv("OPENAI_MODEL", "non défini — mode auto actif"),
    }


# ── /split ──────────────────────────────────────────────────────────────────
@app.post("/split", response_model=SplitResponse)
async def split_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        parts   = split_pdf_into_parts(content)
        del content; gc.collect()
        results = [SplitResult(file_name=p["file_name"],
                               pdf_base64=base64.b64encode(p["pdf_bytes"]).decode())
                   for p in parts]
        del parts; gc.collect()
        return SplitResponse(success=True, total_files=len(results), results=results)
    except Exception as e:
        logger.exception("Error /split")
        return SplitResponse(success=False, error=str(e))


# ── /extract ─────────────────────────────────────────────────────────────────
@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        filename  = file.filename or "devis.pdf"
        data, model_used = await extract_with_vision(pdf_bytes, get_openai_client())
        file_url  = await upload_pdf_to_bubble(pdf_bytes, filename)
        del pdf_bytes; gc.collect()
        return ExtractionResponse(success=True, data=data, file_url=file_url, model_used=model_used)
    except HTTPException as e:
        return ExtractionResponse(success=False, error=e.detail)
    except Exception as e:
        logger.exception("Error /extract")
        return ExtractionResponse(success=False, error=str(e))


# ── /split-light ─────────────────────────────────────────────────────────────
@app.post("/split-light", response_model=SplitLightResponse)
async def split_light(file: UploadFile = File(...)):
    try:
        content = await file.read()
        parts   = split_pdf_into_parts(content)
        del content; gc.collect()

        results, errors = [], []
        for part in parts:
            try:
                meta   = extract_metadata_regex(part["pdf_bytes"], part["file_name"])
                vendor = meta["vendor_name"]
                proj   = meta["project_name"]
                inv    = meta["invoice_number"]
                letter = vendor[0].upper() if vendor and vendor[0].isalpha() else "#"
                path   = f"/PROD/{letter}/{vendor}/{proj}/Devis et commande/{inv}.pdf"
                b64    = base64.b64encode(part["pdf_bytes"]).decode()
                results.append(SplitLightItem(
                    file_name=part["file_name"], pdf_base64=b64,
                    vendor_name=vendor, project_name=proj,
                    invoice_number=inv, drive_path=path
                ))
            except Exception as e:
                errors.append({"file_name": part["file_name"], "error": str(e)})

        del parts; gc.collect()
        return SplitLightResponse(success=True, total_files=len(results),
                                  results=results, errors=errors)
    except Exception as e:
        logger.exception("Error /split-light")
        return SplitLightResponse(success=False, error=str(e))


# ── /split-and-extract ───────────────────────────────────────────────────────
@app.post("/split-and-extract", response_model=SplitExtractResponse)
async def split_and_extract(file: UploadFile = File(...)):
    try:
        content      = await file.read()
        parts        = split_pdf_into_parts(content)
        del content; gc.collect()

        results, errors = [], []
        openai_client   = get_openai_client()

        for part in parts:
            try:
                # choose_model appelé individuellement pour chaque devis splitté
                data, model_used = await extract_with_vision(part["pdf_bytes"], openai_client)
                b64  = base64.b64encode(part["pdf_bytes"]).decode()
                results.append(SplitExtractItem(
                    file_name=part["file_name"],
                    pdf_base64=b64,
                    extraction=data,
                    model_used=model_used
                ))
                gc.collect()
            except Exception as e:
                errors.append(SplitExtractError(
                    file_name=part["file_name"], error=str(e),
                    page_start=part["page_start"], page_end=part["page_end"]
                ))

        del parts; gc.collect()
        return SplitExtractResponse(
            success=True,
            total_found=len(results) + len(errors),
            total_extracted=len(results),
            total_errors=len(errors),
            results=results, errors=errors
        )
    except Exception as e:
        logger.exception("Error /split-and-extract")
        return SplitExtractResponse(success=False, error=str(e))
