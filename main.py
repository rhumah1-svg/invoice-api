"""
Invoice/Quote PDF Extraction API - V19.0 HYBRID (Speed + Precision)
===================================================================
- STRUCTURED OUTPUTS : `client.beta.chat.completions.parse` (zéro erreur JSON)
- SMART ROUTING : choose_model() score la complexité → mini ou 4o
- VISION FALLBACK : Si texte extrait trop pauvre, envoie les pages en images
- LECTURE RAPIDE : PyMuPDF par défaut, fallback pypdf
"""

import os
import io
import re
import base64
import logging
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

MODEL_FAST = os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini")
MODEL_STRONG = os.getenv("OPENAI_MODEL_STRONG", "gpt-4o")

# Seuils pour choose_model()
MIN_TEXT_LENGTH = 100          # En dessous → vision fallback
COMPLEXITY_THRESHOLD = 60      # Score au-dessus → gpt-4o

# ═══════════════════════════════════════════════════════
# SCHEMAS PYDANTIC (Structured Outputs)
# ═══════════════════════════════════════════════════════

class Metadata(BaseModel):
    vendor_name:    str = Field(..., description="Nom de l'entreprise CLIENTE (jamais Qualidal)")
    project_name:   str = Field(..., description="Nom du projet / chantier (valeur après 'Chantier')")
    invoice_number: str = Field(..., description="Référence stricte format 'devis_dexxxxxx' en minuscules")
    date:           str = Field(..., description="Date du devis au format YYYY-MM-DD")
    currency:       str = Field(default="EUR", description="Code devise ISO")

class LineItem(BaseModel):
    designation: str   = Field(..., description="Nom COURT du produit/service (pas la description technique)")
    description: str   = Field(default="", description="Texte descriptif complet, conserver les retours à la ligne exacts")
    quantity:    float = Field(..., description="Quantité (nombre exact)")
    unite:       str   = Field(..., description="Unité exacte : ML, M2, FORF, U, H, J, SEM, ENS, etc.")
    unit_price:  float = Field(..., description="Prix unitaire HT")

class Totals(BaseModel):
    subtotal_ht: float = Field(..., description="Total HT")
    total_tax:   float = Field(..., description="TVA totale")
    total_ttc:   float = Field(..., description="Total TTC")

class InvoiceData(BaseModel):
    metadata:   Metadata
    line_items: list[LineItem]
    totals:     Totals

class ExtractionResponse(BaseModel):
    success:    bool
    data:       Optional[InvoiceData] = None
    error:      Optional[str]         = None
    model_used: Optional[str]         = None
    time_taken: Optional[str]         = None
    method:     Optional[str]         = None  # "text" ou "vision"

# ═══════════════════════════════════════════════════════
# LECTURE PDF
# ═══════════════════════════════════════════════════════

def extract_text_fast(pdf_bytes: bytes) -> str:
    """Extrait le texte via PyMuPDF, fallback pypdf."""
    t0 = time.time()
    text = ""
    
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        logger.info(f"[CHRONO] Lecture PDF (PyMuPDF) : {time.time() - t0:.3f}s — {len(text)} chars")
        return text.strip()
    except ImportError:
        logger.info("PyMuPDF non disponible, fallback pypdf...")
    
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        logger.info(f"[CHRONO] Lecture PDF (pypdf) : {time.time() - t0:.3f}s — {len(text)} chars")
    except Exception as e:
        logger.warning(f"[text] Erreur lecture PDF: {e}")
        
    return text.strip()


def pdf_to_images_base64(pdf_bytes: bytes, max_pages: int = 5) -> list[str]:
    """Convertit les pages PDF en images base64 pour le mode vision."""
    t0 = time.time()
    images = []
    
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            # Render à 200 DPI pour un bon compromis qualité/taille
            mat = fitz.Matrix(200/72, 200/72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append(b64)
        doc.close()
        logger.info(f"[CHRONO] PDF→Images ({len(images)} pages) : {time.time() - t0:.3f}s")
    except Exception as e:
        logger.error(f"[vision] Erreur conversion images: {e}")
    
    return images

# ═══════════════════════════════════════════════════════
# SMART MODEL ROUTING
# ═══════════════════════════════════════════════════════

def choose_model(text: str) -> tuple[str, int]:
    """
    Score la complexité du texte extrait.
    Retourne (model_name, score).
    Score > COMPLEXITY_THRESHOLD → gpt-4o
    """
    score = 0
    
    # Nombre de lignes avec des montants (indicateur de lignes de devis)
    amount_lines = len(re.findall(r'\d[\d\s]*[.,]\d{2}', text))
    if amount_lines > 15:
        score += 25  # Beaucoup de lignes = complexe
    elif amount_lines > 8:
        score += 10
    
    # Longueur du texte (proxy pour nombre de pages)
    if len(text) > 5000:
        score += 15
    if len(text) > 10000:
        score += 15
    
    # Présence de tableaux complexes (beaucoup de colonnes alignées)
    tab_indicators = len(re.findall(r'\t', text))
    if tab_indicators > 30:
        score += 15
    
    # Sous-totaux multiples (structure hiérarchique)
    subtotals = len(re.findall(r'(?i)sous[- ]?total|total\s+ht|total\s+lot', text))
    if subtotals > 3:
        score += 15
    
    # Texte avec beaucoup d'espaces mal parsés (colonnes mélangées)
    double_spaces = len(re.findall(r'  {3,}', text))
    if double_spaces > 20:
        score += 20
    
    model = MODEL_STRONG if score >= COMPLEXITY_THRESHOLD else MODEL_FAST
    logger.info(f"[ROUTING] Score={score} → {model} (amounts={amount_lines}, len={len(text)}, tabs={tab_indicators}, subtotals={subtotals})")
    
    return model, score

# ═══════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Tu es un expert en extraction de données de devis de travaux BTP français.
Ces devis sont émis par QUALIDAL.

══════════════════════════════════════════
RÈGLE ABSOLUE — NORMALISATION DES NOMBRES
══════════════════════════════════════════
Les nombres français utilisent un espace comme séparateur de milliers ET une virgule décimale.
Tu DOIS toujours :
1. Supprimer TOUS les espaces dans les nombres
2. Remplacer la virgule décimale par un point

EXEMPLES OBLIGATOIRES :
- "1 200"     → 1200.0
- "1 700"     → 1700.0
- "17 750"    → 17750.0
- "17 750,00" → 17750.0
- "1 200,50"  → 1200.5
- "176 602,00"→ 176602.0
- "35 320,40" → 35320.4

ATTENTION : "1 200" n'est PAS "1.2" — c'est mille deux cents.

══════════════════════════════════════════
RÈGLE — DÉTECTION D'UN PRODUIT VALIDE
══════════════════════════════════════════
Un produit valide est une ligne qui contient TOUS ces éléments :
  - Une quantité (Qté)
  - Une unité (ML, M2, FORF, UNIT, U...)
  - Un prix unitaire HT (P.U. HT) ≠ 0
  - Un montant HT ≠ 0

Structure typique sur le PDF :
  [Nom du produit]     [Qté]  [Unité]  [P.U. HT]  [Montant HT]  [TVA]

Exemples de lignes valides :
  "Réparation épaufrures :   3,00   ML   125,00   375,00   20,00"
  "Splits :   3 550,00   UNIT   5,00   17 750,00   20,00"
  "APPLICATION DE RESINE -   360,00   M2   45,60   16 416,00   20,00"

══════════════════════════════════════════
RÈGLE — DESIGNATION ET DESCRIPTION
══════════════════════════════════════════
Chaque produit a :
- designation : Le NOM COURT sur la ligne avec les prix (avant ou au début du bloc)
  Exemples : "Réparation épaufrures", "Splits", "APPLICATION DE RESINE", "Mastic"
  → Toujours la première ligne du bloc, SANS les chiffres de quantité/prix
  → Si le nom finit par ":" ou "-", inclure sans le caractère final

- description : Le texte descriptif SOUS la ligne de prix
  → Peut être multiligne
  → Conserver les sauts de ligne exacts
  → Peut être vide si pas de description

══════════════════════════════════════════
RÈGLE — CE QU'IL NE FAUT PAS EXTRAIRE
══════════════════════════════════════════
NE PAS créer de produit pour :
- Les lignes de section/titre sans prix : "C1 :", "C2 :", "LOCAL DE CHARGE :"
- Les lignes à zéro : quantité=0 OU montant=0 OU prix=0
- Les sous-totaux et totaux
- Les CGV : "Acompte", "Bon pour accord", "Plus-value", mentions légales
- Les lignes purement descriptives sans données chiffrées

══════════════════════════════════════════
RÈGLE — METADATA
══════════════════════════════════════════
- vendor_name   : L'entreprise CLIENTE (jamais Qualidal)
                  Exemple : "INGENIERIE 2K"
- project_name  : Valeur de "Chantier"
                  Exemple : "Savigny Le Temple (77)"
- invoice_number: Format strict "devis_de" + chiffres (ex: devis_de00005461)
                  Source : numéro "DE00005461" dans le document
- date          : Format YYYY-MM-DD
- currency      : "EUR"
"""

# ═══════════════════════════════════════════════════════
# EXTRACTION ENGINE
# ═══════════════════════════════════════════════════════

async def extract_text_mode(pdf_text: str, model: str, client: AsyncOpenAI) -> tuple[InvoiceData, str]:
    """Extraction via texte + Structured Outputs."""
    t0 = time.time()
    
    resp = await client.beta.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Voici le texte du devis :\n\n{pdf_text}"}
        ],
        response_format=InvoiceData
    )
    
    data = resp.choices[0].message.parsed
    elapsed = time.time() - t0
    logger.info(f"[CHRONO] IA text mode ({model}) : {elapsed:.3f}s — {len(data.line_items)} items extraits")
    return data, f"{model} (structured/text)"


async def extract_vision_mode(pdf_bytes: bytes, model: str, client: AsyncOpenAI) -> tuple[InvoiceData, str]:
    """Extraction via images des pages PDF (fallback quand le texte est mauvais)."""
    t0 = time.time()
    
    images_b64 = pdf_to_images_base64(pdf_bytes, max_pages=5)
    if not images_b64:
        raise HTTPException(status_code=500, detail="Impossible de convertir le PDF en images")
    
    # Construire le message avec images
    content = [{"type": "text", "text": "Voici les pages du devis en images. Extrais toutes les données."}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
        })
    
    # Note : .parse() avec vision fonctionne sur gpt-4o mais PAS sur gpt-4o-mini
    # On force gpt-4o pour le mode vision
    vision_model = MODEL_STRONG
    
    resp = await client.beta.chat.completions.parse(
        model=vision_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        response_format=InvoiceData
    )
    
    data = resp.choices[0].message.parsed
    elapsed = time.time() - t0
    logger.info(f"[CHRONO] IA vision mode ({vision_model}) : {elapsed:.3f}s — {len(data.line_items)} items extraits")
    return data, f"{vision_model} (structured/vision)"


async def extract_smart(pdf_bytes: bytes, client: AsyncOpenAI) -> tuple[InvoiceData, str, str]:
    """
    Pipeline intelligent :
    1. Extraire texte (rapide)
    2. Si texte trop court → vision fallback
    3. Sinon → choose_model() → text mode
    4. Validation basique → si suspect, retry vision
    """
    # Étape 1 : Extraction texte
    pdf_text = extract_text_fast(pdf_bytes)
    
    # Étape 2 : Texte trop court ? → Vision directe
    if len(pdf_text) < MIN_TEXT_LENGTH:
        logger.info(f"[ROUTING] Texte trop court ({len(pdf_text)} chars) → vision fallback")
        data, model_info = await extract_vision_mode(pdf_bytes, MODEL_STRONG, client)
        return data, model_info, "vision"
    
    # Étape 3 : Choisir le modèle selon la complexité
    model, score = choose_model(pdf_text)
    
    try:
        data, model_info = await extract_text_mode(pdf_text, model, client)
        
        # Étape 4 : Validation basique
        issues = validate_extraction(data, pdf_text)
        
        if issues and model == MODEL_FAST:
            # Problème détecté avec mini → retry avec gpt-4o
            logger.info(f"[VALIDATION] Problèmes détectés avec {MODEL_FAST}: {issues} → retry {MODEL_STRONG}")
            data, model_info = await extract_text_mode(pdf_text, MODEL_STRONG, client)
            issues2 = validate_extraction(data, pdf_text)
            
            if issues2:
                # Toujours des problèmes → dernier recours : vision
                logger.info(f"[VALIDATION] Toujours des problèmes → vision fallback")
                data, model_info = await extract_vision_mode(pdf_bytes, MODEL_STRONG, client)
                return data, model_info, "vision (auto-fallback)"
            
            return data, model_info, "text (retry)"
        
        return data, model_info, "text"
        
    except Exception as e:
        logger.warning(f"[FALLBACK] Erreur text mode: {e} → vision fallback")
        data, model_info = await extract_vision_mode(pdf_bytes, MODEL_STRONG, client)
        return data, model_info, "vision (error-fallback)"


def validate_extraction(data: InvoiceData, pdf_text: str) -> list[str]:
    """
    Vérifie la cohérence de l'extraction.
    Retourne une liste de problèmes détectés (vide = OK).
    """
    issues = []
    
    # 1. Aucun item extrait alors que le texte contient des montants
    amount_matches = re.findall(r'\d[\d\s]*[.,]\d{2}', pdf_text)
    if len(data.line_items) == 0 and len(amount_matches) > 3:
        issues.append(f"0 items extraits mais {len(amount_matches)} montants dans le texte")
    
    # 2. vendor_name = Qualidal (erreur fréquente)
    if data.metadata.vendor_name and 'qualidal' in data.metadata.vendor_name.lower():
        issues.append("vendor_name contient 'Qualidal' (devrait être le client)")
    
    # 3. invoice_number mal formaté
    if data.metadata.invoice_number and not re.match(r'^devis_de\d+$', data.metadata.invoice_number):
        issues.append(f"invoice_number mal formaté: {data.metadata.invoice_number}")
    
    # 4. Total HT = 0 alors qu'il y a des items
    if len(data.line_items) > 0 and data.totals.subtotal_ht == 0:
        issues.append("Total HT = 0 avec des items présents")
    
    # 5. Somme des items très différente du total déclaré (>10% d'écart)
    if data.totals.subtotal_ht > 0 and len(data.line_items) > 0:
        items_sum = sum(item.quantity * item.unit_price for item in data.line_items)
        if items_sum > 0:
            ecart = abs(items_sum - data.totals.subtotal_ht) / data.totals.subtotal_ht
            if ecart > 0.10:
                issues.append(f"Écart {ecart:.0%} entre somme items ({items_sum:.2f}) et total HT ({data.totals.subtotal_ht:.2f})")
    
    if issues:
        logger.warning(f"[VALIDATION] Problèmes: {issues}")
    
    return issues

# ═══════════════════════════════════════════════════════
# APP FASTAPI
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Invoice Extraction API", version="19.0.0 (Hybrid)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_openai_client() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY manquant")
    return AsyncOpenAI(api_key=key)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "19.0.0",
        "mode": "Hybrid (Speed + Precision)",
        "model_fast": MODEL_FAST,
        "model_strong": MODEL_STRONG
    }

@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    t_start = time.time()
    try:
        pdf_bytes = await file.read()
        
        data, model_used, method = await extract_smart(pdf_bytes, get_openai_client())
        
        total_time = f"{time.time() - t_start:.2f}s"
        logger.info(f"[CHRONO] TOTAL {file.filename} : {total_time} — méthode: {method}")
        
        return ExtractionResponse(
            success=True,
            data=data,
            model_used=model_used,
            time_taken=total_time,
            method=method
        )
        
    except HTTPException as e:
        return ExtractionResponse(success=False, error=e.detail)
    except Exception as e:
        logger.exception("Error /extract")
        return ExtractionResponse(success=False, error=str(e))
