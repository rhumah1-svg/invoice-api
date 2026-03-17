"""
Invoice/Quote PDF Extraction API - V17.1 PURE EXTRACT (OPTIMISÉE)
================================================================
- FIX : Résolution de l'erreur "SyntaxError: unterminated string literal".
- FOCUS UNIQUE : Extraction de données (Plus de /split).
- VITESSE & PRÉCISION : Utilisation de `layout=True` pour pdfplumber.
- CORRECTION BUGS : Prompt strict sur le formatage des nombres (4 000 -> 4000.0).
- RETOURS À LA LIGNE : Préservation stricte des sauts de ligne (\n) dans les descriptions.
- MODE HYBRIDE : Texte d'abord (gpt-4o-mini), Vision en secours.
- PERF V17.1 : Exécution simultanée (asyncio.gather) de l'IA et de l'upload Bubble.
"""

import os
import io
import logging
import base64
import gc
import json
from typing import Optional

import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# SÉLECTION DU MODÈLE (Pour la Vision en secours)
# ═══════════════════════════════════════════════════════

_AMO_KEYWORDS = [
    "AMO", "phase exe", "phase exé", "OPR", "DOE", "CCTP", "EN15620",
    "profilographe", "planéité", "réunion préparatoire", "dossier béton",
    "contrôle qualité", "bureau de contrôle", "pièces marchés"
]

def choose_model(pdf_bytes: bytes) -> str:
    override = os.getenv("OPENAI_MODEL", "").strip()
    if override in ("gpt-4o", "gpt-4o-mini"): return override

    try:
        full_text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() or ""
        
        full_text_lower = full_text.lower()
        keyword_hits = sum(1 for kw in _AMO_KEYWORDS if kw.lower() in full_text_lower)
        
        if len(full_text) > 4000 or keyword_hits >= 2:
            return "gpt-4o"
        return "gpt-4o-mini"
    except Exception:
        return "gpt-4o-mini"

# ═══════════════════════════════════════════════════════
# SCHEMAS PYDANTIC
# ═══════════════════════════════════════════════════════

class Metadata(BaseModel):
    vendor_name:    str = Field(..., description="Nom de l'entreprise cliente")
    project_name:   str = Field(..., description="Nom du projet / chantier")
    invoice_number: str = Field(..., description="Référence devis_dexxxxxx")
    date:           str = Field(..., description="Date YYYY-MM-DD")
    currency:       str = Field(..., description="Code devise ISO")

class LineItem(BaseModel):
    designation: str   = Field(..., description="Nom court du produit/service")
    description: str   = Field(default="", description="Texte complet descriptif")
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
    model_used: Optional[str]         = None

# ═══════════════════════════════════════════════════════
# PDF → IMAGES (VISION FALLBACK)
# ═══════════════════════════════════════════════════════

def pdf_bytes_to_images_b64(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            buf = io.BytesIO(pix.tobytes("jpeg", jpg_quality=75))
            out.append(base64.b64encode(buf.getvalue()).decode())
        doc.close()
        return out
    except ImportError:
        pass
    
    try:
        import pdf2image
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi, fmt="jpeg")
        out = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            out.append(base64.b64encode(buf.getvalue()).decode())
        return out
    except ImportError:
        raise RuntimeError("Aucune librairie de rendu PDF installée (pymupdf ou pdf2image).")

# ═══════════════════════════════════════════════════════
# EXTRACTION TEXTE (OPTIMISÉE AVEC LAYOUT=TRUE)
# ═══════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extrait le texte en préservant l'espacement visuel des colonnes (crucial pour l'IA)."""
    all_text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # layout=True simule les espaces entre les colonnes
                all_text += (page.extract_text(layout=True) or "") + "\n"
    except Exception as e:
        logger.warning(f"[text] pdfplumber échoué: {e}")
        return ""
    return all_text.strip()

# ═══════════════════════════════════════════════════════
# PROMPT TEXTE & VISION CONSOLIDÉS
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Tu es un expert en extraction de données de devis de travaux BTP français.
Ces devis sont émis par QUALIDAL (13 avenue du Parc Alata, 60100 Creil).
Retourne UNIQUEMENT un JSON valide, sans texte ni markdown autour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE ABSOLUE — NOMBRES ET FORMATS JSON (ANTI-CRASH)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Il est STRICTEMENT INTERDIT de mettre des espaces dans les nombres du JSON final.
Sur le devis français, les milliers sont souvent séparés par un espace (ex: "4 000,00").
Tu DOIS IMPÉRATIVEMENT :
1. Supprimer TOUS les espaces.
2. Remplacer la virgule par un point.
Exemples obligatoires : 
- "4 000,00" DOIT devenir 4000.0
- "16 750,50" DOIT devenir 16750.5
- "1 500" DOIT devenir 1500.0
- "-130,00" DOIT devenir -130.0
Si tu laisses un espace dans un nombre, le JSON sera invalide et le système plantera.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES METADATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- vendor_name : L'entreprise CLIENTE (à qui est adressé le devis). Jamais Qualidal.
- project_name : Colonne "Chantier".
- invoice_number : Format OBLIGATOIRE "devis_de" + chiffres en minuscules (ex: devis_de00004001).
- date : AAAA-MM-JJ

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES LINE ITEMS (PRESTATIONS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Un item valide = une ligne du tableau avec un P.U. HT ≠ 0 OU un Montant HT ≠ 0.

- designation : Nom court (4-8 mots). Première ligne de l'item. Inclure le nom de la cellule si présent (ex: "Réparation - Cellule 1").
- description : Tout le texte SOUS l'item, jusqu'au prochain item chiffré. Si aucun texte, mettre "". Ne jamais tronquer.
  -> OBLIGATION ABSOLUE : Tu DOIS conserver les retours à la ligne exacts du devis d'origine en utilisant le caractère "\\n" dans la chaîne JSON. Ne condense JAMAIS le texte sur une seule ligne. Ceci est particulièrement important pour les listes à tirets ou à puces.
- quantity : float.
- unite : FORF, ML, M2, U, Heures, Jours, Semaine.
- unit_price : float.

NE PAS créer d'item pour :
- Les lignes avec que des zéros
- Les sous-totaux
- Les titres de section sans prix
- Les CGV (ex: "Travail le week-end", "Acompte", "Bon pour accord").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DE SORTIE ATTENDU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "metadata": { "vendor_name": "...", "project_name": "...", "invoice_number": "devis_deXXXXXXXX", "date": "YYYY-MM-DD", "currency": "EUR" },
  "line_items": [
    { "designation": "...", "description": "Ligne 1\\n- Tiret 1\\n- Tiret 2", "quantity": 0.0, "unite": "...", "unit_price": 0.0 }
  ],
  "totals": { "subtotal_ht": 0.0, "total_tax": 0.0, "total_ttc": 0.0 }
}
"""

# ═══════════════════════════════════════════════════════
# MOTEURS D'EXTRACTION
# ═══════════════════════════════════════════════════════

async def extract_with_vision(pdf_bytes: bytes, openai_client: AsyncOpenAI) -> tuple[InvoiceData, str]:
    import time
    t0 = time.time()
    try: images_b64 = pdf_bytes_to_images_b64(pdf_bytes, dpi=150)
    except RuntimeError as e: raise HTTPException(status_code=500, detail=str(e))

    model = choose_model(pdf_bytes)
    content = [{"type": "text", "text": "Extrait TOUS les items avec P.U. HT ≠ 0. Retourne UNIQUEMENT le JSON."}]
    for img in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}})

    last_error = None
    for attempt in range(1, 3):
        try:
            resp = await openai_client.chat.completions.create(
                model=model, max_tokens=8000, temperature=0,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
            )
            raw = resp.choices[0].message.content.strip()
            # Nettoyage sécurisé sans expressions régulières
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            break
        except Exception as e:
            last_error = e
            continue
    else:
        raise HTTPException(status_code=500, detail=f"Vision JSON invalide: {last_error}")

    return InvoiceData(**data), f"{model} (vision)"

async def extract_with_text(pdf_bytes: bytes, openai_client: AsyncOpenAI) -> tuple[InvoiceData, str]:
    import time
    t0 = time.time()
    
    pdf_text = extract_text_from_pdf(pdf_bytes)
    
    # Si le PDF est un scan (pas de texte) -> bascule immédiate sur Vision
    if len(pdf_text) < 100:
        logger.warning("[text] Texte trop court, fallback vision")
        return await extract_with_vision(pdf_bytes, openai_client)

    model = "gpt-4o" if len(pdf_text) > 5000 else "gpt-4o-mini"
    max_tok = 8000 if len(pdf_text) > 5000 else 4000

    last_error = None
    for attempt in range(1, 3):
        try:
            resp = await openai_client.chat.completions.create(
                model=model, max_tokens=max_tok, temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Texte du devis (formaté avec espaces et retours à la ligne) :\n\n{pdf_text}"}
                ]
            )
            raw = resp.choices[0].message.content.strip()
            # Nettoyage sécurisé sans expressions régulières
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            break
        except Exception as e:
            last_error = e
            continue
    else:
        raise HTTPException(status_code=500, detail=f"Text JSON invalide: {last_error}")

    n_items = len(data.get("line_items", []))
    if n_items == 0:
        logger.warning("[text] 0 items trouvés -> fallback vision")
        return await extract_with_vision(pdf_bytes, openai_client)

    return InvoiceData(**data), f"{model} (text)"

# ═══════════════════════════════════════════════════════
# BUBBLE UPLOAD & APP FASTAPI
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Invoice Extraction API", version="17.2.0 (Pure Extract)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_openai_client() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY manquant")
    return AsyncOpenAI(api_key=key)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "17.2.0", "mode": "Pure Extract Hybrid"}

@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        
        # ⚡ Extraction pure des données (Texte d'abord, Vision si besoin)
        data, model_used = await extract_with_text(pdf_bytes, get_openai_client())
        
        del pdf_bytes; gc.collect()
        return ExtractionResponse(success=True, data=data, model_used=model_used)
        
    except HTTPException as e:
        return ExtractionResponse(success=False, error=e.detail)
    except Exception as e:
        logger.exception("Error /extract")
        return ExtractionResponse(success=False, error=str(e))
