"""
Invoice/Quote PDF Extraction API - V18.0 PURE EXTRACT (SPEED DEMON)
===================================================================
- VITESSE EXTRÊME : Utilisation de `client.beta.chat.completions.parse` (Structured Outputs).
- LECTURE INSTANTANÉE : PyMuPDF par défaut, fallback sur pypdf (0.05s max).
- SÉCURITÉ : Plus de manipulation de chaînes JSON, Pydantic gère tout nativement.
- DEBUGGING : Surveillez les [CHRONO] dans les logs de votre console Render.
"""

import os
import io
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
# SCHEMAS PYDANTIC (Utilisés nativement par OpenAI)
# ═══════════════════════════════════════════════════════

class Metadata(BaseModel):
    vendor_name:    str = Field(..., description="Nom de l'entreprise cliente")
    project_name:   str = Field(..., description="Nom du projet / chantier")
    invoice_number: str = Field(..., description="Référence stricte format 'devis_dexxxxxx'")
    date:           str = Field(..., description="Date YYYY-MM-DD")
    currency:       str = Field(..., description="Code devise ISO (ex: EUR)")

class LineItem(BaseModel):
    designation: str   = Field(..., description="Nom court du produit/service")
    description: str   = Field(default="", description="Texte complet descriptif, conserver les retours à la ligne exacts.")
    quantity:    float = Field(..., description="Quantité (nombre exact, sans espaces)")
    unite:       str   = Field(..., description="Unité (ML, M2, FORF, U, etc.)")
    unit_price:  float = Field(..., description="Prix unitaire HT (nombre exact, sans espaces)")

class Totals(BaseModel):
    subtotal_ht: float = Field(..., description="Total HT (sans espaces)")
    total_tax:   float = Field(..., description="TVA (sans espaces)")
    total_ttc:   float = Field(..., description="Total TTC (sans espaces)")

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

# ═══════════════════════════════════════════════════════
# LECTURE PDF ULTRA-RAPIDE (0.01s - 0.05s)
# ═══════════════════════════════════════════════════════

def extract_text_fast(pdf_bytes: bytes) -> str:
    """Extrait le texte instantanément via PyMuPDF (fitz), fallback pypdf."""
    t0 = time.time()
    text = ""
    
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        logger.info(f"[CHRONO] Lecture PDF (PyMuPDF) : {time.time() - t0:.3f}s")
        return text.strip()
    except ImportError:
        logger.info("PyMuPDF non installé, basculement sur PyPDF...")
    
    # Fallback ultra-rapide (PyPDF est instantané, contrairement à pdfplumber)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            text += page.extract_text() + "\n"
        logger.info(f"[CHRONO] Lecture PDF (pypdf) : {time.time() - t0:.3f}s")
    except Exception as e:
        logger.warning(f"[text] Erreur de lecture PDF: {e}")
        
    return text.strip()

# ═══════════════════════════════════════════════════════
# PROMPT CONCIS (Le format JSON est géré par Pydantic)
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Tu es un expert en extraction de données de devis de travaux BTP français.
Ces devis sont émis par QUALIDAL.

RÈGLES ABSOLUES POUR LES NOMBRES (CRITIQUE) :
Les milliers ont souvent un espace (ex: "4 000,00"). 
Tu DOIS :
1. Supprimer TOUS les espaces.
2. Remplacer la virgule par un point.
Exemples : "4 000,00" -> 4000.0 | "16 750,50" -> 16750.5 | "1 500" -> 1500.0.

RÈGLES METADATA :
- vendor_name : L'entreprise CLIENTE (jamais Qualidal).
- project_name : Valeur de "Chantier".
- invoice_number : Format "devis_de" + chiffres minuscules (ex: devis_de00004001).

RÈGLES PRESTATIONS :
Un item valide = ligne avec P.U. HT ≠ 0 OU Montant HT ≠ 0.
Ne PAS créer d'item pour : Les lignes à zéro, les sous-totaux, les titres de section, les CGV ("Acompte", "Bon pour accord").
description : Tout le texte sous l'item. Conserver ABSOLUMENT les sauts de ligne exacts.
"""

# ═══════════════════════════════════════════════════════
# MOTEUR D'EXTRACTION STRUCTURED OUTPUTS
# ═══════════════════════════════════════════════════════

async def extract_with_structured_outputs(pdf_bytes: bytes, openai_client: AsyncOpenAI) -> tuple[InvoiceData, str]:
    t0 = time.time()
    
    # 1. Lecture instantanée
    pdf_text = extract_text_fast(pdf_bytes)
    
    # 2. On force gpt-4o-mini (Le plus rapide pour ce type de tâche)
    model = "gpt-4o-mini"
    
    # 3. Appel OpenAI via `.parse` (Nouvelle API Structured Outputs = Ultra rapide)
    t_openai_start = time.time()
    try:
        resp = await openai_client.beta.chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Voici le texte du devis :\n\n{pdf_text}"}
            ],
            response_format=InvoiceData # La magie s'opère ici !
        )
        # Les données sont déjà transformées en objet Pydantic valide, pas besoin de json.loads() !
        data = resp.choices[0].message.parsed
        
        logger.info(f"[CHRONO] IA OpenAI (.parse) a répondu en : {time.time() - t_openai_start:.3f}s")
        return data, f"{model} (structured)"
        
    except Exception as e:
        logger.error(f"[ERROR] Échec OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur d'extraction IA: {str(e)}")

# ═══════════════════════════════════════════════════════
# APP FASTAPI
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Invoice Extraction API", version="18.0.0 (Speed Demon)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_openai_client() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY manquant")
    return AsyncOpenAI(api_key=key)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "18.0.0", "mode": "Pure Extract Ultra Fast"}

@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    t_start = time.time()
    try:
        pdf_bytes = await file.read()
        
        # Extraction via Structured Outputs
        data, model_used = await extract_with_structured_outputs(pdf_bytes, get_openai_client())
        
        total_time = f"{time.time() - t_start:.2f}s"
        logger.info(f"[CHRONO] Temps TOTAL API pour {file.filename} : {total_time}")
        
        return ExtractionResponse(success=True, data=data, model_used=model_used, time_taken=total_time)
        
    except HTTPException as e:
        return ExtractionResponse(success=False, error=e.detail)
    except Exception as e:
        logger.exception("Error /extract")
        return ExtractionResponse(success=False, error=str(e))
