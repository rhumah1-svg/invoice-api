"""
Invoice/Quote PDF Extraction API - V10.8 (Final Complete)
=========================================================
1. /split-light : Regex STRICTE (Chantier -> Date) pour le classement.
2. /extract     : Prompt IA COMPLET (V10.1) pour l'extraction détaillée.
3. Performance  : PyPDF + Garbage Collection pour la RAM.
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
# Logging & Config
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
    file_name: str
    pdf_base64: str
    vendor_name: str
    project_name: str
    invoice_number: str
    drive_path: str

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
# 1. Extraction Texte PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
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
# 2. Découpage PDF
# ─────────────────────────────────────────────
def split_pdf_into_parts(pdf_bytes: bytes) -> list[dict]:
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

    # Découpage physique
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
# 3. Extraction Metadata (REGEX STRICTE)
# ─────────────────────────────────────────────
def extract_metadata_regex(pdf_bytes: bytes, file_name: str) -> dict:
    """
    Logique Chantier : On cherche la DATE et on prend tout ce qui est AVANT.
    """
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
    except Exception as e:
        logger.warning(f"Erreur lecture PDF {file_name}: {e}")
        return {"vendor_name": "INCONNU", "project_name": "INCONNU", "invoice_number": file_name.replace(".pdf", "")}
    finally:
        gc.collect()

    lines = text.split("\n")

    # A. Invoice Number
    invoice_match = re.search(r"(DE\d{4,10})", text, re.IGNORECASE)
    invoice_number = f"devis_{invoice_match.group(1).lower()}" if invoice_match else file_name.replace(".pdf", "")

    # B. Vendor Name
    vendor_name = "INCONNU"
    for i, line in enumerate(lines):
        if re.match(r"^(Monsieur|Madame|M\.|Mme)\s+", line.strip(), re.IGNORECASE):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r"^\d", next_line):
                    vendor_name = next_line
                    break
    
    if vendor_name == "INCONNU":
        found_block = False
        for line in lines:
            if "email" in line.lower() or "fax" in line.lower(): found_block = True; continue
            if found_block and line.strip():
                if line.strip().isupper() and len(line) > 3 and "DEVIS" not in line:
                    vendor_name = line.strip()
                    break

    # C. Project Name (LOGIQUE STRICTE : CHANTIER -> DATE)
    project_name = "INCONNU"
    
    for i, line in enumerate(lines):
        if "Chantier" in line:
            for offset in range(1, 6):
                if i + offset >= len(lines): break
                next_line = lines[i + offset].strip()
                if not next_line: continue
                
                # Ignorer parasites
                if any(x in next_line.lower() for x in ["de l'offre", "validité"]): continue

                # Mur de Date
                date_match = re.search(r"(\d{2}/\d{2}/\d{4})", next_line)
                
                if date_match:
                    raw_chantier = next_line[:date_match.start()]
                    project_name = raw_chantier.replace("|", "").strip()
                    break
            
            if project_name != "INCONNU": break

    # Nettoyage
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
# 4. Service LLM OpenAI (PROMPT COMPLET V10.1)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert document parser specialising in French invoices and quotes
(devis, factures) for construction, building repair, and technical services.

TASK:
Given the raw text extracted from a PDF invoice or quote, return a single
JSON object that matches the provided schema **exactly**.

RULES FOR METADATA:
1. `vendor_name`: The client/company name the document is billed to (e.g., 'EUROTECH CHAMPAGNE').
2. `project_name`: The site location or project name, usually found under the table header 'Chantier' (e.g., 'TERRIA IMMO - Puiseux (62)' or 'LAGNY SUR MARNE (77)').
3. `invoice_number`: YOU MUST FORMAT THIS STRICTLY AS 'devis_deXXXXXX' (all lowercase). 
   For example, if you see 'DE00004894' or 'DE00005445' on the document, you MUST output 'devis_de00004894'.
4. `date`: The document creation date. Convert strictly to YYYY-MM-DD format.
5. `currency`: ISO 4217 code. Infer from symbols (€ → EUR).

RULES FOR LINE ITEMS:
1. Extract ONLY lines from the pricing table that have a quantity AND a unit price.
2. For `designation`: use ONLY the short product/service name.
   Do NOT include descriptions, technical details, or sub-text.
   Examples:
   - GOOD: "Réparation joint épaufré"
   - BAD:  "Réparation joint épaufré : Sciage de part et d'autre..."
3. NORMALIZE designations: Capitalize first letter of each significant word, fix typos.
   Keep zone/area identifiers (e.g., "- Zone 1", "- Cellule A1").
4. For `unite`: use the unit code as written (ML, M2, FORF, U, KG, etc.)
5. For `unit_price`: price per unit excluding tax (PU HT)
6. For `quantity`: the quantity (Qté)
7. Do NOT extract section headers, sub-totals, or description paragraphs.

RULES FOR TOTALS:
1. Use values explicitly written in the document.
2. `subtotal_ht`: Total before tax.
3. `total_tax`: Total TVA amount.
4. `total_ttc`: Grand total including tax.

GENERAL RULES:
- Monetary values MUST be plain floats.
- If a field is truly absent, use "" for strings, 0.0 for numbers.
- Never invent data.
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

app = FastAPI(title="Invoice Extraction API", version="10.8.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_extractor: Optional[InvoiceExtractor] = None
def get_extractor() -> InvoiceExtractor:
    global _extractor
    if _extractor is None: _extractor = InvoiceExtractor()
    return _extractor

@app.get("/health")
async def health(): return {"status": "ok", "version": "10.8.0"}

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

# Route 2: Extract Seul (Utilise l'IA)
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

# Route 3: Split-Light (UTILISE LA REGEX STRICTE)
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
                meta = extract_metadata_regex(part["pdf_bytes"], part["file_name"])
                
                vendor = meta["vendor_name"]
                project = meta["project_name"]
                inv_num = meta["invoice_number"]
                
                letter = vendor[0].upper() if vendor and vendor[0].isalpha() else "#"
                
                # DOSSIER DE BASE
                base_folder = "/split dossier"
                path = f"{base_folder}/{letter}/{vendor}/{project}/Devis et commande/{inv_num}.pdf"
                
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

# Route 4: Split + Extract (Utilise l'IA)
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
