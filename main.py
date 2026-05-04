"""
Invoice/Quote PDF Extraction API - V16.3 HYBRID + Ref Cde regex
================================================================
Mode HYBRIDE : pdfplumber extrait le TEXTE du PDF, envoyé à gpt-5.4-mini.
Plus besoin d'images → 5-8s au lieu de 30-40s, coût /10.

Changelog V16.3 :
  - Ajout extract_ref_cde() : regex deterministe (pas d'IA) pour récupérer
    la "Ref Cde" présente sur tous les devis Qualidal.
  - Schema Metadata enrichi avec champ ref_cde (Optional).
  - Injection ref_cde après parsing JSON OpenAI dans extract_with_text() et
    extract_with_vision().
  - Endpoint /extract-meta inchangé côté contrat — ref_cde apparait dans
    data.metadata.ref_cde si trouvée, sinon null.

  - extract_with_text() : fonction principale (texte pdfplumber → LLM)
  - extract_with_vision() : conservée en FALLBACK si pdfplumber échoue
  - Endpoints : /health + /extract + /extract-meta
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
    ref_cde:        Optional[str] = Field(None, description="Référence commande client (texte libre, extrait par regex)")

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


# Schema light pour /extract-meta : pas de line_items, pas de totals détaillés
class MetaOnly(BaseModel):
    vendor_name:    str = Field("", description="Nom de l'entreprise cliente")
    project_name:   str = Field("", description="Nom du projet / chantier")
    invoice_number: str = Field("", description="Référence devis_dexxxxxx")
    date:           str = Field("", description="Date YYYY-MM-DD")
    currency:       str = Field("EUR", description="Code devise ISO")
    total_ht:       float = Field(0.0, description="Total HT")
    ref_cde:        Optional[str] = Field(None, description="Référence commande client (regex)")

class MetaExtractionResponse(BaseModel):
    success:    bool
    data:       Optional[MetaOnly] = None
    error:      Optional[str]      = None
    file_url:   Optional[str]      = None
    model_used: Optional[str]      = None


# ═══════════════════════════════════════════════════════
# REF CDE — Extraction par regex (déterministe, sans IA)
# ═══════════════════════════════════════════════════════

def extract_ref_cde(text: str) -> Optional[str]:
    """
    Extract 'Ref Cde' from PDF text.
    Pattern: 'Ref Cde : <texte>' jusqu'à 'Dossier suivi par' OU 'DEXXXXXX'
    (numéro devis qui suit) OU double newline.
    Returns None if not found, empty, or matches only a devis number.

    Tested on 4 real devis (DE00005559, DE00005560, DE00005620, DE00005015) :
      → "Complément travaux AR floor"
      → "Sondage délamination – Zone AR"
      → "REMISE EN ETAT DALLAGE-LA CHEVROLIERE (44)"
      → "Option 1: Rectification de planéité selon EN15620 DM2 toute largeur"
    """
    if not text:
        return None

    match = re.search(
        r'Ref\s*Cde\s*:\s*(.+?)\s*(?=Dossier\s+suivi\s+par|DE\d{6,}|\n\s*\n|E[-\s]?[Mm]ail\s*:|\bT[ée]l\s*:|\bFax\s*:|N°\s+de\s+Tva)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if not match:
        return None

    ref = match.group(1).strip()
    ref = re.sub(r'\s+', ' ', ref)  # normalise espaces multiples + newlines

    # Filter : si vide ou si ressemble à numéro devis seul
    if not ref or re.fullmatch(r'DE\d+', ref, re.IGNORECASE):
        return None

    return ref


def extract_chantier(text: str) -> Optional[str]:
    """
    Extract 'Chantier' value from PDF text (regex, no LLM).
    Pattern: la ligne sous le header "Chantier Date Condition... N° de Tva intracom"
    et après la ligne "de l'offre" contient :
        [CHANTIER] DD/MM/YYYY [DATE_VALIDITE] [CONDITION] [TVA]
    On extrait tout ce qui précède la première date DD/MM/YYYY.

    Tested on 6 real devis :
      DE00005559 → "BVA3"
      DE00005560 → "BVA3"
      DE00005015 → "ARMOR IIMAK, LA CHEVROLIERE 44"
      DE00005478 → "33 Av du Bois de la Pie"
      DE00005534 → "ARMOR IIMAK, LA CHEVROLIERE 44"
      Devis_4-130-131 → "05 Rue Nicolas Appert, 44118"
    """
    if not text:
        return None

    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Trouver header "Chantier ... N° de Tva"
        if re.search(r'Chantier\s+Date.*Tva', line, re.IGNORECASE):
            # Chercher dans les 5 lignes suivantes la ligne avec une date DD/MM/YYYY
            for j in range(i+1, min(i+6, len(lines))):
                next_line = lines[j]
                # Match : tout ce qui est avant la première DD/MM/YYYY
                m = re.match(r'^(.+?)\s+(\d{1,2}/\d{1,2}/\d{4})', next_line)
                if m:
                    chantier = m.group(1).strip()
                    chantier = re.sub(r'\s+', ' ', chantier)
                    if chantier and not re.fullmatch(r'de\s+l\'offre', chantier, re.IGNORECASE):
                        return chantier
            return None
    return None


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
    out_pages = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                out_pages.append(txt)
    except Exception as e:
        logger.warning(f"[pdfplumber] échec : {e}")
        return ""
    return "\n\n".join(out_pages).strip()


# ═══════════════════════════════════════════════════════
# PROMPT TEXT
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT_TEXT = """\
Tu es un expert en extraction de données de devis de travaux BTP français.
Ces devis sont toujours émis par la société QUALIDAL (13 avenue du Parc Alata, 60100 Creil).
Tu reçois le TEXTE BRUT extrait d'un devis (toutes pages concaténées).
Retourne UNIQUEMENT un JSON valide, sans texte ni markdown autour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 1 — vendor_name
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

vendor_name = entreprise CLIENTE (jamais Qualidal).
Cherche le bloc d'adresse client en haut du devis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 2 — project_name (CRITIQUE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

project_name = TEXTE EXACT de la cellule sous l'en-tête "Chantier"
dans le tableau de tête (Chantier | Date | Condition de règlement | N° de Tva intracom).

Le project_name est UN SEUL champ court : la cellule sous "Chantier" du tableau,
généralement de quelques mots (ex: "BVA3", "33 Av du Bois de la Pie",
"ARMOR IIMAK, LA CHEVROLIERE 44").

⚠️ PIÈGES FRÉQUENTS À ÉVITER :

1. "Ref Cde" N'EST PAS le project_name. JAMAIS prendre Ref Cde comme project_name.
   - Ref Cde est un titre fonctionnel libre rédigé par le commercial.
   - Le Chantier est un identifiant court de lieu/code.

2. "Adresse du chantier" en bas du devis n'est PAS le project_name.
   - C'est l'adresse complète, plus longue.

3. Si tu vois deux champs proches (Ref Cde et Chantier), prends TOUJOURS Chantier.

EXEMPLES :
- Devis DE00005478 :
    Ref Cde : "Remise en état du dallage-PNII DC8-TREMBLAY EN FRANCE (93)"
    Chantier : "33 Av du Bois de la Pie"
    → project_name = "33 Av du Bois de la Pie"  (PAS la Ref Cde !)

- Devis DE00005534 :
    Ref Cde : "Option 2 : Rectification de planéité selon EN15620 DM2 toute largeur"
    Chantier : "ARMOR IIMAK, LA CHEVROLIERE 44"
    → project_name = "ARMOR IIMAK, LA CHEVROLIERE 44"

- Devis DE00005559 :
    Ref Cde : "Complément travaux AR floor"
    Chantier : "BVA3"
    → project_name = "BVA3"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 3 — invoice_number
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Format devis_dexxxxxxxx (lowercase, 8 chiffres, padding zéros).
Source : "DE00005559" → "devis_de00005559".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 4 — date
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Format YYYY-MM-DD. Source = champ "Date" du tableau de tête.
"11/02/2026" → "2026-02-11".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE 5 — line_items
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Un item = une prestation avec P.U. HT ≠ 0 OU Montant HT ≠ 0.
Lignes à 0,00 → NE PAS créer d'item.
Sous-totaux de zone → NE PAS créer d'item.
Titres de section ("Cellule 3 - ZIEGLER :") → NE PAS créer d'item.

── designation ──
Nom court (5-8 mots max). Première ligne de la cellule Description.

── description ──
Texte COMPLET de la cellule Description (sous-sections, tirets, détails).
EXCLURE CGV : "Travaux réalisés en semaine...", "Fourniture eau/électricité...",
"QUALIDAL n'est pas responsable...".
Ne JAMAIS prendre le texte d'un item voisin.

── quantity, unite, unit_price ──
Lire les chiffres sur la ligne de l'item.
Espaces = séparateurs de milliers : "1 022,00" = 1022.0 | "1 700" = 1700.0
Virgule = décimal : "4,60" = 4.6
Unités : FORF / ML / M2 / UNIT→U / Heures / Jours / Semaine

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES MÉTADONNÉES & ITEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

vendor_name = client (jamais Qualidal).
project_name = champ Chantier.
invoice_number = format devis_dexxxxxxxx (lowercase, 8 chiffres).
date = YYYY-MM-DD.
Espaces dans nombres = milliers ("1 700" = 1700.0).

Format JSON identique au prompt texte.
"""


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

    # Tentative extraction texte pour Ref Cde (même en mode vision, on récupère ce qu'on peut)
    pdf_text_for_ref = extract_text_from_pdf(pdf_bytes)
    ref_cde_value = extract_ref_cde(pdf_text_for_ref)
    logger.info(f"[ref_cde] (vision) regex → {ref_cde_value!r}")

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
                max_completion_tokens=8000,
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

    # Injection ref_cde (regex, indépendant du LLM)
    if "metadata" in data and isinstance(data["metadata"], dict):
        data["metadata"]["ref_cde"] = ref_cde_value
        # Override project_name avec valeur regex Chantier (priorité absolue sur LLM)
        chantier_value = extract_chantier(pdf_text_for_ref)
        if chantier_value:
            old_pn = data["metadata"].get("project_name", "")
            if old_pn != chantier_value:
                logger.info(f"[vision] override project_name: {old_pn!r} → {chantier_value!r} (regex)")
            data["metadata"]["project_name"] = chantier_value

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
# EXTRACTION HYBRIDE — V16.2 : gpt-5.4-mini partout
# ═══════════════════════════════════════════════════════

async def extract_with_text(
    pdf_bytes: bytes,
    openai_client: AsyncOpenAI
) -> tuple[InvoiceData, str]:
    """
    V16.3 HYBRIDE : extrait le texte via pdfplumber puis l'envoie à gpt-5.4-mini.
    Fallback sur extract_with_vision si pdfplumber échoue.
    Ref Cde extraite par regex sur pdf_text avant appel OpenAI.
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

    # Extraction Ref Cde par regex (avant OpenAI, pas de tokens supplémentaires)
    ref_cde_value = extract_ref_cde(pdf_text)
    if ref_cde_value:
        logger.info(f"[ref_cde] regex → {ref_cde_value!r}")
    else:
        logger.info("[ref_cde] regex → non trouvé")

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

    # Injection ref_cde (regex, indépendant du LLM)
    if "metadata" in data and isinstance(data["metadata"], dict):
        data["metadata"]["ref_cde"] = ref_cde_value
        # Override project_name avec valeur regex Chantier (priorité absolue sur LLM)
        chantier_value = extract_chantier(pdf_text)
        if chantier_value:
            old_pn = data["metadata"].get("project_name", "")
            if old_pn != chantier_value:
                logger.info(f"[text] override project_name: {old_pn!r} → {chantier_value!r} (regex)")
            data["metadata"]["project_name"] = chantier_value

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
# EXTRACTION META-ONLY (light) — pour /extract-meta
# ═══════════════════════════════════════════════════════

async def extract_meta_from_text(
    pdf_bytes: bytes,
    openai_client: AsyncOpenAI
) -> tuple[MetaOnly, str]:
    """
    Extraction métadonnées seules (pas de line_items détaillés).
    Plus rapide, moins cher. Pour le tableau de gestion devis.
    Réutilise extract_with_text() qui retourne InvoiceData complet,
    puis on map vers MetaOnly. Ref Cde déjà injectée par extract_with_text.
    """
    invoice_data, model_used = await extract_with_text(pdf_bytes, openai_client)
    meta = MetaOnly(
        vendor_name    = invoice_data.metadata.vendor_name,
        project_name   = invoice_data.metadata.project_name,
        invoice_number = invoice_data.metadata.invoice_number,
        date           = invoice_data.metadata.date,
        currency       = invoice_data.metadata.currency,
        total_ht       = invoice_data.totals.subtotal_ht,
        ref_cde        = invoice_data.metadata.ref_cde,
    )
    return meta, model_used


# ═══════════════════════════════════════════════════════
# APP & ROUTES
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Invoice Extraction API", version="16.3.0")
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
        "version": "16.3.0",
        "mode": "hybrid (text first, vision fallback) + ref_cde regex",
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


@app.post("/extract-meta", response_model=MetaExtractionResponse)
async def extract_meta(file: UploadFile = File(...)):
    """
    Extraction métadonnées uniquement (pas d'items).
    Plus rapide, moins cher. Pour le tableau de gestion devis.
    Inclut ref_cde extraite par regex.
    """
    try:
        pdf_bytes = await file.read()
        filename  = file.filename or "devis.pdf"
        data, model_used = await extract_meta_from_text(pdf_bytes, get_openai_client())
        file_url  = await upload_pdf_to_bubble(pdf_bytes, filename)
        del pdf_bytes; gc.collect()
        return MetaExtractionResponse(success=True, data=data, file_url=file_url, model_used=model_used)
    except HTTPException as e:
        return MetaExtractionResponse(success=False, error=e.detail)
    except Exception as e:
        logger.exception("Error /extract-meta")
        return MetaExtractionResponse(success=False, error=str(e))
