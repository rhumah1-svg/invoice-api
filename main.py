"""
Invoice/Quote PDF Extraction API - V10.4 (Stable & Low RAM)
===========================================================
- Découpage : Via PyPDF (Rapide & Léger)
- Metadata : Lecture de la PAGE 1 UNIQUEMENT (Pas de crop physique risqué)
- Logique : Algorithme de détection "Chantier" avec saut de lignes parasites
"""

import os
import io
import logging
import base64
import re
import asyncio
import gc
from typing import Optional

import pdfplumber
from pypdf import PdfReader, PdfWriter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class Metadata(BaseModel):
    vendor_name: str = Field(..., description="Nom de l'entreprise cliente")
    project_name: str = Field(..., description="Nom du projet / chantier")
    invoice_number: str = Field(..., description="Référence devis_dexxxxxx")
    date: str = Field(..., description="Date YYYY-MM-DD")
    currency: str = Field(..., description="Code devise ISO")

class LineItem(BaseModel):
    designation: str = Field(..., description="Nom court du produit/service")
    quantity: float = Field(..., description="Quantité")
    unite: str = Field(..., description="Unité (ML, M2, FORF, U, etc.)")
    unit_price: float = Field(..., description="Prix unitaire HT")

class Totals(BaseModel):
    subtotal_ht: float = Field(..., description="Total HT")
    total_tax: float = Field(..., description="TVA")
    total_ttc: float = Field(..., description="Total TTC")

class InvoiceData(BaseModel):
    metadata: Metadata
    line_items: list[LineItem]
    totals: Totals

class ExtractionResponse(BaseModel):
    success: bool
    data: Optional[InvoiceData] = None
    error: Optional[str] = None

class SplitResult(BaseModel):
    file_name: str
    pdf_base64: str

class SplitResponse(BaseModel):
    success: bool
    total_files: int = Field(default=0)
    results: list[SplitResult] = Field(default=[])
    error: Optional[str] = None

# ── Schemas /split-light ──

class SplitLightItem(BaseModel):
    file_name: str = Field(..., description="Nom du fichier (devis_deXXXXX.pdf)")
    pdf_base64: str = Field(..., description="PDF base64")
    vendor_name: str = Field(..., description="Entreprise (regex)")
    project_name: str = Field(..., description="Chantier (regex)")
    invoice_number: str = Field(..., description="Numéro devis formaté")
    drive_path: str = Field(..., description="Chemin OneDrive calculé")

class SplitLightResponse(BaseModel):
    success: bool
    total_files: int = Field(default=0)
    results: list[SplitLightItem] = Field(default=[])
    errors: list[dict] = Field(default=[])
    error: Optional[str] = None

# ── Schemas /split-and-extract ──

class SplitExtractItem(BaseModel):
    file_name: str
    pdf_base64: str
    extraction: InvoiceData

class SplitExtractError(BaseModel):
    file_name: str
    error: str
    page_start: int
    page_end: int

class SplitExtractResponse(BaseModel):
    success: bool
    total_found: int = Field(default=0)
    total_extracted: int = Field(default=0)
    total_errors: int = Field(default=0)
    results: list[SplitExtractItem] = Field(default=[])
    errors: list[SplitExtractError] = Field(default=[])


# ─────────────────────────────────────────────
# 1. Extraction texte PDF (Pour IA - Complet)
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extraction complète pour l'analyse IA (/extract)"""
    all_text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text_parts: list[str] = [f"--- Page {page_num} ---"]
            try:
                tables = page.extract_tables(table_settings={"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"})
                if tables:
                    for table in tables:
                        for row in table:
                            cleaned = [cell.strip() if cell else "" for cell in row]
                            page_text_parts.append(" | ".join(cleaned))
            except Exception: pass
            
            non_table_text = page.extract_text()
            if non_table_text: page_text_parts.append(non_table_text)
            all_text_parts.append("\n".join(page_text_parts))
            page.flush_cache()

    gc.collect()
    full_text = "\n\n".join(all_text_parts)
    if not full_text.strip(): raise ValueError("PDF vide ou illisible")
    return full_text


# ─────────────────────────────────────────────
# 2. Découpage PDF (Optimisé PyPDF)
# ─────────────────────────────────────────────
def split_pdf_into_parts(pdf_bytes: bytes) -> list[dict]:
    """Découpe le fichier en utilisant pypdf (léger en RAM)"""
    split_points = []
    devis_names = []
    
    # Scan rapide
    reader_scan = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader_scan.pages)
    
    for i in range(total_pages):
        try:
            page_text = reader_scan.pages[i].extract_text() or ""
            match = re.search(r"DE\d{4,10}", page_text, re.IGNORECASE)
            if match:
                split_points.append(i)
                clean_name = f"devis_{match.group(0).lower()}"
                devis_names.append(clean_name)
        except Exception: continue

    if not split_points or split_points[0] != 0:
        split_points.insert(0, 0)
        devis_names.insert(0, "devis_inconnu")
    split_points.append(total_pages)

    del reader_scan
    gc.collect()

    # Écriture des parties
    reader_write = PdfReader(io.BytesIO(pdf_bytes))
    parts = []

    for i in range(len(split_points) - 1):
        start_page = split_points[i]
        end_page = split_points[i + 1]
        if start_page >= end_page: continue

        writer = PdfWriter()
        for j in range(start_page, end_page):
            writer.add_page(reader_write.pages[j])

        sub_pdf_io = io.BytesIO()
        writer.write(sub_pdf_io)
        parts.append({
            "file_name": f"{devis_names[i]}.pdf",
            "pdf_bytes": sub_pdf_io.getvalue(),
            "page_start": start_page + 1,
            "page_end": end_page,
        })
        sub_pdf_io.close()
        del writer

    del reader_write
    gc.collect()
    return parts


# ─────────────────────────────────────────────
# 3. Extraction Metadata (Regex - Page 1 Seule)
# ─────────────────────────────────────────────
def extract_metadata_regex(pdf_bytes: bytes, file_name: str) -> dict:
    """
    Extrait Client/Chantier/Devis via Regex.
    Optimisation RAM : Ne lit que la PAGE 1 (texte brut).
    """
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if pdf.pages:
                # On lit juste la première page entière. 
                # C'est sûr et peu gourmand (quelques Ko de texte).
                text = pdf.pages[0].extract_text() or ""
    except Exception as e:
        logger.warning(f"Erreur lecture PDF {file_name}: {e}")
        return {"vendor_name": "INCONNU", "project_name": "INCONNU", "invoice_number": file_name.replace(".pdf", "")}
    finally:
        gc.collect()

    lines = text.split("\n")

    # ── Invoice Number ──
    invoice_match = re.search(r"(DE\d{4,10})", text, re.IGNORECASE)
    if invoice_match:
        invoice_number = f"devis_{invoice_match.group(1).lower()}"
    else:
        invoice_number = file_name.replace(".pdf", "")

    # ── Vendor Name ──
    vendor_name = "INCONNU"
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if re.match(r"^(Monsieur|Madame|M\.|Mme)\s+", line_stripped, re.IGNORECASE):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r"^\d", next_line):
                    vendor_name = next_line
                    break
    
    if vendor_name == "INCONNU":
        found_block = False
        for line in lines:
            line_stripped = line.strip()
            if "email" in line_stripped.lower() or "fax" in line_stripped.lower():
                found_block = True
                continue
            if found_block and line_stripped:
                is_upper = (line_stripped == line_stripped.upper())
                is_clean = not any(kw in line_stripped for kw in ["DEVIS", "FACTURE", "TVA", "HT", "TOTAL", "QUALIDAL", "CREIL"])
                if len(line_stripped) > 2 and len(line_stripped) < 50 and is_upper and is_clean:
                    vendor_name = line_stripped
                    break

    # ── Project Name (Logique Skip Lignes) ──
    project_name = "INCONNU"
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if "Chantier" in line_stripped:
            # On regarde les 4 lignes suivantes
            for offset in range(1, 5):
                if i + offset >= len(lines): break
                next_line = lines[i + offset].strip()
                if not next_line: continue

                # Détection parasite
                is_parasite = any(x in next_line.lower() for x in ["de l'offre", "validité", "condition"])
                has_date = re.search(r"\d{2}/\d{2}/\d{4}", next_line)
                
                # Si parasite et pas de date, on saute
                if is_parasite and not has_date: continue

                if has_date:
                    candidate = next_line[:has_date.start()].strip()
                    project_name = candidate.strip(" |")
                    break 
                elif len(next_line) > 3 and not is_parasite:
                    clean = re.split(r"\s+(?:de l'offre|Date|Condition|VIREMENT|N°|Tva|€)", next_line, flags=re.IGNORECASE)[0].strip()
                    if clean:
                        project_name = clean
                        break
            if project_name != "INCONNU": break

    if project_name == "INCONNU":
        chantier_match = re.search(r"Chantier\s*[:\-]?\s*(.+?)(?:\d{2}/\d{2}/\d{4}|$)", text, re.IGNORECASE)
        if chantier_match:
            project_name = chantier_match.group(1).strip()

    # Nettoyage final
    vendor_name = re.sub(r"[,;.\s]+$", "", vendor_name)[:100]
    project_name = re.sub(r"[,;.\s]+$", "", project_name)[:100]
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        vendor_name = vendor_name.replace(char, '-')
        project_name = project_name.replace(char, '-')

    return {
        "vendor_name": vendor_name,
        "project_name": project_name,
        "invoice_number": invoice_number,
    }


# ─────────────────────────────────────────────
# Service LLM OpenAI
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert document parser specialising in French invoices and quotes.
TASK: Return a single JSON object matching the schema exactly.
RULES:
1. `vendor_name`: Client name.
2. `project_name`: Site location/project name (under 'Chantier').
3. `invoice_number`: Strictly 'devis_deXXXXXX'.
4. `date`: YYYY-MM-DD.
5. Extract ONLY lines with quantity AND unit price.
6. `designation`: Short name only.
7. `unite`: ML, M2, FORF, U, etc.
8. No sub-totals or descriptions as items.
"""

class InvoiceExtractor:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("OPENAI_API_KEY missing")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def extract(self, pdf_text: str) -> InvoiceData:
        max_chars = 60_000
        if len(pdf_text) > max_chars: pdf_text = pdf_text[:max_chars] + "\n[TRUNCATED]"
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pdf_text}],
            response_format=InvoiceData,
        )
        return response.choices[0].message.parsed


# ─────────────────────────────────────────────
# FastAPI App & Routes
# ─────────────────────────────────────────────

app = FastAPI(title="Invoice Extraction API", version="10.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_extractor: Optional[InvoiceExtractor] = None
def get_extractor() -> InvoiceExtractor:
    global _extractor
    if _extractor is None: _extractor = InvoiceExtractor()
    return _extractor

@app.get("/health")
async def health(): return {"status": "ok", "version": "10.4.0"}

# Route 1: Split Seul
@app.post("/split", response_model=SplitResponse)
async def split_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        parts = split_pdf_into_parts(content)
        del content; gc.collect()
        results = [SplitResult(file_name=p["file_name"], pdf_base64=base64.b64encode(p["pdf_bytes"]).decode("utf-8")) for p in parts]
        del parts; gc.collect()
        return SplitResponse(success=True, total_files=len(results), results=results)
    except Exception as e:
        logger.exception("Error /split")
        return SplitResponse(success=False, error=str(e))

# Route 2: Extract Seul
@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = extract_text_from_pdf(content)
        del content; gc.collect()
        data = await get_extractor().extract(text)
        return ExtractionResponse(success=True, data=data)
    except Exception as e:
        logger.exception("Error /extract")
        return ExtractionResponse(success=False, error=str(e))

# Route 3: Split-Light (Votre route principale)
@app.post("/split-light", response_model=SplitLightResponse)
async def split_light(file: UploadFile = File(...)):
    try:
        content = await file.read()
        parts = split_pdf_into_parts(content)
        del content; gc.collect()
        
        results = []
        errors = []
        
        for part in parts:
            try:
                # Metadata sur Page 1 uniquement
                meta = extract_metadata_regex(part["pdf_bytes"], part["file_name"])
                
                vendor = meta["vendor_name"]
                project = meta["project_name"]
                inv_num = meta["invoice_number"]
                
                letter = vendor[0].upper() if vendor and vendor[0].isalpha() else "#"
                path = f"/split dossier/{letter}/{vendor}/{project}/Devis et commande/{inv_num}.pdf"
                
                b64 = base64.b64encode(part["pdf_bytes"]).decode("utf-8")
                results.append(SplitLightItem(
                    file_name=part["file_name"], pdf_base64=b64,
                    vendor_name=vendor, project_name=project, invoice_number=inv_num, drive_path=path
                ))
            except Exception as e:
                errors.append({"file_name": part["file_name"], "error": str(e)})
        
        del parts; gc.collect()
        return SplitLightResponse(success=True, total_files=len(results), results=results, errors=errors)
    except Exception as e:
        logger.exception("Error /split-light")
        return SplitLightResponse(success=False, error=str(e))

# Route 4: Split + Extract
@app.post("/split-and-extract", response_model=SplitExtractResponse)
async def split_and_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()
        parts = split_pdf_into_parts(content)
        del content; gc.collect()
        
        results = []
        errors = []
        extractor = get_extractor()

        for part in parts:
            try:
                text = extract_text_from_pdf(part["pdf_bytes"])
                data = await extractor.extract(text)
                b64 = base64.b64encode(part["pdf_bytes"]).decode("utf-8")
                results.append(SplitExtractItem(file_name=part["file_name"], pdf_base64=b64, extraction=data))
                del text; gc.collect()
            except Exception as e:
                errors.append(SplitExtractError(file_name=part["file_name"], error=str(e), page_start=part["page_start"], page_end=part["page_end"]))

        del parts; gc.collect()
        return SplitExtractResponse(success=True, total_found=len(results)+len(errors), total_extracted=len(results), total_errors=len(errors), results=results, errors=errors)
    except Exception as e:
        return SplitExtractResponse(success=False, error=str(e))
