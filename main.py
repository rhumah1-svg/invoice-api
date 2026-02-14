"""
Invoice/Quote PDF Extraction API - V10.2 (Low Memory Edition)
=============================================================
- Optimisation RAM : Utilisation de pypdf pour le scanning (léger) au lieu de pdfplumber (lourd).
- Garbage Collection forcé.
"""

import os
import io
import logging
import base64
import re
import asyncio
import gc  # Import important pour la gestion mémoire
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
# Extraction texte PDF (Extraction fine)
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Utilise pdfplumber pour extraire le texte avec précision (tables).
    Gourmand en RAM, à utiliser uniquement sur des fichiers découpés.
    """
    all_text_parts: list[str] = []
    
    # Context manager pour s'assurer que pdfplumber libère la mémoire
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text_parts: list[str] = [f"--- Page {page_num} ---"]
            
            # Extraction tables (lourd)
            try:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                    }
                )
                if tables:
                    for table in tables:
                        for row in table:
                            cleaned = [cell.strip() if cell else "" for cell in row]
                            page_text_parts.append(" | ".join(cleaned))
            except Exception:
                pass # Si table échoue, on continue
                
            non_table_text = page.extract_text()
            if non_table_text:
                page_text_parts.append(non_table_text)
            
            all_text_parts.append("\n".join(page_text_parts))
            
            # Nettoyage explicite par page si nécessaire
            del page 

    full_text = "\n\n".join(all_text_parts)
    gc.collect() # Forcer le nettoyage RAM
    
    if not full_text.strip():
        raise ValueError("Aucun texte n'a pu être extrait du PDF.")
    return full_text


# ─────────────────────────────────────────────
# Découpage PDF (OPTIMISÉ RAM)
# ─────────────────────────────────────────────

def split_pdf_into_parts(pdf_bytes: bytes) -> list[dict]:
    """
    Découpe le PDF.
    OPTIMISATION : Utilise pypdf pour scanner le texte (beaucoup plus léger que pdfplumber).
    """
    split_points = []
    devis_names = []
    
    # 1. SCAN LÉGER avec PyPDF (évite de charger le layout graphique en RAM)
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
        except Exception:
            continue

    if not split_points or split_points[0] != 0:
        split_points.insert(0, 0)
        devis_names.insert(0, "devis_inconnu")
    split_points.append(total_pages)

    logger.info(f"Découpage : {len(split_points) - 1} devis sur {total_pages} pages.")
    
    # Libérer le reader de scan
    del reader_scan
    gc.collect()

    # 2. DÉCOUPAGE
    reader_write = PdfReader(io.BytesIO(pdf_bytes))
    parts = []

    for i in range(len(split_points) - 1):
        start_page = split_points[i]
        end_page = split_points[i + 1]
        if start_page >= end_page:
            continue

        writer = PdfWriter()
        for j in range(start_page, end_page):
            writer.add_page(reader_write.pages[j])

        sub_pdf_io = io.BytesIO()
        writer.write(sub_pdf_io)
        
        # On récupère les bytes tout de suite et on ferme le buffer
        result_bytes = sub_pdf_io.getvalue()
        sub_pdf_io.close()

        parts.append({
            "file_name": f"{devis_names[i]}.pdf",
            "pdf_bytes": result_bytes,
            "page_start": start_page + 1,
            "page_end": end_page,
        })
        
        # Nettoyage intermédiaire pour les gros fichiers
        del writer
        
    del reader_write
    gc.collect() # Gros nettoyage avant de renvoyer
    return parts


# ─────────────────────────────────────────────
# Extraction Metadata par REGEX (Corrigée & Optimisée)
# ─────────────────────────────────────────────

def extract_metadata_regex(pdf_bytes: bytes, file_name: str) -> dict:
    text = ""
    # On utilise pdfplumber ici car on a besoin de la précision spatiale pour les tableaux
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
    except Exception as e:
        logger.warning(f"Erreur lecture PDF {file_name}: {e}")
        return {
            "vendor_name": "INCONNU",
            "project_name": "INCONNU",
            "invoice_number": file_name.replace(".pdf", ""),
        }
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
        found_qualidal_block = False
        for line in lines:
            line_stripped = line.strip()
            if "email" in line_stripped.lower() or "fax" in line_stripped.lower():
                found_qualidal_block = True
                continue
            if found_qualidal_block and line_stripped:
                is_uppercase = (line_stripped == line_stripped.upper())
                is_clean = not any(kw in line_stripped for kw in [
                    "DEVIS", "FACTURE", "TVA", "HT", "TTC", "PAGE", 
                    "DATE", "TOTAL", "QUALIDAL", "CREIL", "CEDEX"
                ])
                if len(line_stripped) > 2 and len(line_stripped) < 50 and is_uppercase and is_clean:
                    vendor_name = line_stripped
                    break

    # ── Project Name (chantier) ──
    project_name = "INCONNU"
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if "Chantier" in line_stripped:
            for offset in range(1, 5):
                if i + offset >= len(lines): break
                next_line = lines[i + offset].strip()
                if not next_line: continue

                is_parasite = ("de l'offre" in next_line.lower() or 
                               "validité" in next_line.lower() or
                               "condition" in next_line.lower())
                has_date = re.search(r"\d{2}/\d{2}/\d{4}", next_line)
                
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
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def extract(self, pdf_text: str) -> InvoiceData:
        logger.info("Appel OpenAI (%s) avec %d caractères", self.model, len(pdf_text))
        max_chars = 60_000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "\n\n[...TRUNCATED...]"

        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract data from this document:\n\n{pdf_text}"},
            ],
            response_format=InvoiceData,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI a renvoyé une réponse non analysable")
        return parsed


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Invoice Extraction & Split API",
    version="10.2.0",
    description="V10.2: Low Memory Optimization (PyPDF for detection + GC).",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_extractor: Optional[InvoiceExtractor] = None

def get_extractor() -> InvoiceExtractor:
    global _extractor
    if _extractor is None:
        _extractor = InvoiceExtractor()
    return _extractor

@app.get("/")
async def root():
    return {"message": "API Active. Low RAM mode."}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "10.2.0"}


# ─────────────────────────────────────────────
# ROUTE 1: /split
# ─────────────────────────────────────────────
@app.post("/split", response_model=SplitResponse)
async def split_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")
    try:
        content = await file.read()
        parts = split_pdf_into_parts(content)
        
        # Libérer la mémoire du fichier source immédiatement
        del content
        gc.collect()

        results = [
            SplitResult(
                file_name=p["file_name"],
                pdf_base64=base64.b64encode(p["pdf_bytes"]).decode("utf-8"),
            )
            for p in parts
        ]
        
        # Nettoyage
        del parts
        gc.collect()
        
        return SplitResponse(success=True, total_files=len(results), results=results)
    except Exception as e:
        logger.exception("Erreur /split")
        return SplitResponse(success=False, error=str(e))


# ─────────────────────────────────────────────
# ROUTE 2: /extract
# ─────────────────────────────────────────────
@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")
    try:
        pdf_bytes = await file.read()
        
        # Extraction texte
        pdf_text = extract_text_from_pdf(pdf_bytes)
        
        # Libération immédiate des bytes
        del pdf_bytes
        gc.collect()

        extractor = get_extractor()
        invoice_data = await extractor.extract(pdf_text)
        return ExtractionResponse(success=True, data=invoice_data)
    except Exception as e:
        logger.exception("Erreur /extract")
        return ExtractionResponse(success=False, error=str(e))


# ─────────────────────────────────────────────
# ROUTE 3: /split-light
# ─────────────────────────────────────────────
@app.post("/split-light", response_model=SplitLightResponse)
async def split_light(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")

    try:
        pdf_bytes = await file.read()
        logger.info("=== SPLIT-LIGHT : Début ===")
        
        parts = split_pdf_into_parts(pdf_bytes)
        
        # On peut supprimer les bytes originaux maintenant pour faire de la place
        del pdf_bytes
        gc.collect()
        
        logger.info(f"{len(parts)} devis détectés")

        results: list[SplitLightItem] = []
        errors: list[dict] = []

        for part in parts:
            try:
                meta = extract_metadata_regex(part["pdf_bytes"], part["file_name"])

                vendor_name = meta["vendor_name"]
                project_name = meta["project_name"]
                invoice_number = meta["invoice_number"]

                letter = vendor_name[0].upper() if vendor_name and vendor_name[0].isalpha() else "#"
                base_folder = "/TEST"
                drive_path = f"{base_folder}/{letter}/{vendor_name}/{project_name}/Devis et commande/{invoice_number}.pdf"

                pdf_base64 = base64.b64encode(part["pdf_bytes"]).decode("utf-8")

                results.append(SplitLightItem(
                    file_name=part["file_name"],
                    pdf_base64=pdf_base64,
                    vendor_name=vendor_name,
                    project_name=project_name,
                    invoice_number=invoice_number,
                    drive_path=drive_path,
                ))
            except Exception as e:
                logger.warning(f"  ❌ {part['file_name']}: {e}")
                errors.append({
                    "file_name": part["file_name"],
                    "error": str(e),
                })
        
        # Gros nettoyage final
        del parts
        gc.collect()

        logger.info(f"=== SPLIT-LIGHT TERMINÉ : {len(results)} OK ===")

        return SplitLightResponse(
            success=True,
            total_files=len(results),
            results=results,
            errors=errors,
        )

    except Exception as e:
        logger.exception("Erreur /split-light")
        return SplitLightResponse(success=False, error=str(e))


# ─────────────────────────────────────────────
# ROUTE 4: /split-and-extract
# ─────────────────────────────────────────────
@app.post("/split-and-extract", response_model=SplitExtractResponse)
async def split_and_extract(file: UploadFile = File(...)):
    # ATTENTION : Cette route est la plus dangereuse pour la RAM
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")

    try:
        pdf_bytes = await file.read()
        parts = split_pdf_into_parts(pdf_bytes)
        del pdf_bytes
        gc.collect()
        
        total_found = len(parts)
        if total_found == 0:
            return SplitExtractResponse(success=False, error="Aucun devis détecté")

        extractor = get_extractor()
        results: list[SplitExtractItem] = []
        errors: list[SplitExtractError] = []

        # Batch size 1 pour éviter de faire exploser la RAM lors des appels OpenAI
        for part in parts:
            try:
                pdf_text = extract_text_from_pdf(part["pdf_bytes"])
                invoice_data = await extractor.extract(pdf_text)
                
                pdf_base64 = base64.b64encode(part["pdf_bytes"]).decode("utf-8")
                
                results.append(SplitExtractItem(
                    file_name=part["file_name"],
                    pdf_base64=pdf_base64,
                    extraction=invoice_data,
                ))
                # On nettoie le texte extrait
                del pdf_text
                gc.collect()
                
            except Exception as e:
                logger.warning(f"  ❌ {part['file_name']}: {e}")
                errors.append(SplitExtractError(
                    file_name=part["file_name"],
                    error=str(e),
                    page_start=part["page_start"],
                    page_end=part["page_end"],
                ))

        del parts
        gc.collect()

        return SplitExtractResponse(
            success=True,
            total_found=total_found,
            total_extracted=len(results),
            total_errors=len(errors),
            results=results,
            errors=errors,
        )

    except Exception as e:
        logger.exception("Erreur /split-and-extract")
        return SplitExtractResponse(success=False, error=str(e))
