"""
Invoice/Quote PDF Extraction API - V9.0 (Mass Processing)
==========================================================
- /split       → Découpage seul (retourne base64)
- /extract     → Extraction IA seule (1 PDF)
- /split-and-extract → Split + Extract en masse (retourne JSON léger)
"""

import os
import io
import logging
import base64
import re
import asyncio
from typing import Optional

import pdfplumber
from pypdf import PdfReader, PdfWriter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# ─────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class Metadata(BaseModel):
    vendor_name: str = Field(..., description="Nom de l'entreprise cliente (ex: EUROTECH CHAMPAGNE)")
    project_name: str = Field(..., description="Nom du projet ou lieu du chantier, situé sous l'en-tête 'Chantier'")
    invoice_number: str = Field(..., description="Référence du document formatée strictement en devis_dexxxxxx")
    date: str = Field(..., description="Date de création du document au format YYYY-MM-DD")
    currency: str = Field(..., description="Code devise ISO (ex: EUR)")

class LineItem(BaseModel):
    designation: str = Field(
        ...,
        description=(
            "Normalized product/service name. Use a clean, consistent short name "
            "without descriptions. Example: 'Réparation joint épaufré' not "
            "'Réparation joint épaufré : Sciage de part et d'autre...'"
        ),
    )
    quantity: float = Field(..., description="Quantité (Qté)")
    unite: str = Field(..., description="Unité de mesure (ML, M2, FORF, U, KG, etc.)")
    unit_price: float = Field(..., description="Prix unitaire hors taxe (P.U. HT)")

class Totals(BaseModel):
    subtotal_ht: float = Field(..., description="Total HT (avant taxes)")
    total_tax: float = Field(..., description="Montant total de la TVA")
    total_ttc: float = Field(..., description="Total TTC (taxes incluses)")

class InvoiceData(BaseModel):
    metadata: Metadata
    line_items: list[LineItem]
    totals: Totals

class ExtractionResponse(BaseModel):
    success: bool
    data: Optional[InvoiceData] = None
    error: Optional[str] = None

class SplitResult(BaseModel):
    file_name: str = Field(..., description="Nom du fichier généré")
    pdf_base64: str = Field(..., description="Fichier PDF découpé encodé en base64")

class SplitResponse(BaseModel):
    success: bool
    total_files: int = Field(default=0)
    results: list[SplitResult] = Field(default=[])
    error: Optional[str] = None

# ── Schemas pour /split-and-extract ──

class SplitExtractItem(BaseModel):
    """Un devis extrait avec succès"""
    file_name: str = Field(..., description="Nom du fichier (ex: devis_de00004894.pdf)")
    pdf_base64: str = Field(..., description="PDF en base64 pour upload OneDrive")
    extraction: InvoiceData = Field(..., description="Données extraites par l'IA")

class SplitExtractError(BaseModel):
    """Un devis qui a échoué"""
    file_name: str = Field(..., description="Nom du fichier qui a échoué")
    error: str = Field(..., description="Description de l'erreur")
    page_start: int = Field(..., description="Page de début dans le PDF original")
    page_end: int = Field(..., description="Page de fin dans le PDF original")

class SplitExtractResponse(BaseModel):
    success: bool
    total_found: int = Field(default=0, description="Nombre total de devis détectés")
    total_extracted: int = Field(default=0, description="Nombre de devis extraits avec succès")
    total_errors: int = Field(default=0, description="Nombre de devis en erreur")
    results: list[SplitExtractItem] = Field(default=[])
    errors: list[SplitExtractError] = Field(default=[])


# ─────────────────────────────────────────────
# Moteur d'Extraction de Texte (PDFPlumber)
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    all_text_parts: list[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text_parts: list[str] = [f"--- Page {page_num} ---"]

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
                non_table_text = page.extract_text()
                if non_table_text:
                    page_text_parts.append(non_table_text)
            else:
                raw = page.extract_text()
                if raw:
                    page_text_parts.append(raw)

            all_text_parts.append("\n".join(page_text_parts))

    full_text = "\n\n".join(all_text_parts)
    if not full_text.strip():
        raise ValueError("Aucun texte n'a pu être extrait du PDF.")
    return full_text


# ─────────────────────────────────────────────
# Découpage PDF (fonction réutilisable)
# ─────────────────────────────────────────────

def split_pdf_into_parts(pdf_bytes: bytes) -> list[dict]:
    """
    Découpe un gros PDF en sous-PDFs individuels.
    Retourne une liste de dicts avec file_name, pdf_bytes, page_start, page_end.
    """
    split_points = []
    devis_names = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        for i in range(total_pages):
            page_text = pdf.pages[i].extract_text() or ""
            match = re.search(r"DE\d{4,10}", page_text, re.IGNORECASE)

            if match:
                split_points.append(i)
                clean_name = f"devis_{match.group(0).lower()}"
                devis_names.append(clean_name)

        if not split_points or split_points[0] != 0:
            split_points.insert(0, 0)
            devis_names.insert(0, "devis_inconnu")

        split_points.append(total_pages)

    logger.info(f"Découpage : {len(split_points) - 1} devis détectés sur {total_pages} pages.")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []

    for i in range(len(split_points) - 1):
        start_page = split_points[i]
        end_page = split_points[i + 1]

        if start_page >= end_page:
            continue

        writer = PdfWriter()
        for j in range(start_page, end_page):
            writer.add_page(reader.pages[j])

        sub_pdf_io = io.BytesIO()
        writer.write(sub_pdf_io)
        sub_pdf_bytes = sub_pdf_io.getvalue()

        parts.append({
            "file_name": f"{devis_names[i]}.pdf",
            "pdf_bytes": sub_pdf_bytes,
            "page_start": start_page + 1,
            "page_end": end_page,
        })

    return parts


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
4. `date`: The document creation date, usually found in the table next to 'Chantier' under the header 'Date'. Convert it strictly to YYYY-MM-DD format (e.g., '05/01/2026' becomes '2026-01-05').
5. `currency`: ISO 4217 code. Infer from symbols (€ → EUR).

RULES FOR LINE ITEMS:
1. Extract ONLY lines from the pricing table that have a quantity AND a unit price.
2. For `designation`: use ONLY the short product/service name (the bold or first line).
   Do NOT include descriptions, technical details, or sub-text.
   Examples:
   - GOOD: "Réparation joint épaufré"
   - BAD:  "Réparation joint épaufré : Sciage de part et d'autre sur largeur..."
   - GOOD: "Pianotage"
   - BAD:  "Pianotage : Traitement du pianotage par injection de résine..."
   - GOOD: "AMENÉ ET REPLI DU MATÉRIEL - Zone 1"
3. NORMALIZE designations for consistency:
   - Capitalize first letter of each significant word
   - Fix obvious typos (e.g., "épaufré" → "Épaufré")
   - Use consistent naming (always "Réparation Joint Épaufré", never
     "Rep. joint epaufré" or "REPARATION JOINT EPAUFRE")
   - Keep zone/area identifiers (e.g., "- Zone 1", "- Cellule A1")
4. For `unite`: use the unit code as written (ML, M2, FORF, U, KG, etc.)
5. For `unit_price`: the price per unit excluding tax (PU HT)
6. For `quantity`: the quantity (Qté)
7. Do NOT extract:
   - Section headers (e.g., "Cellule A1") that have no price
   - Sub-total lines or summary lines
   - Description paragraphs below product names
   - Lines with only text and no numeric values

RULES FOR TOTALS:
1. Use values explicitly written in the document.
2. `subtotal_ht`: Total before tax.
3. `total_tax`: Total TVA amount.
4. `total_ttc`: Grand total including tax.

GENERAL RULES:
- Monetary values MUST be plain floats (no currency symbols, no spaces).
- If a field is truly absent, use "" for strings, 0.0 for numbers.
- Never invent data. Prefer values explicitly written over computed ones.
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

        logger.info("Extraction réussie: %d items trouvés", len(parsed.line_items))
        return parsed


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Invoice Extraction & Split API",
    version="9.0.0",
    description=(
        "V9: Added /split-and-extract for mass processing. "
        "Split + Extract in one call, returns JSON + base64, with error reporting."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_extractor: Optional[InvoiceExtractor] = None


def get_extractor() -> InvoiceExtractor:
    global _extractor
    if _extractor is None:
        _extractor = InvoiceExtractor()
    return _extractor


@app.get("/")
async def root():
    return {"message": "API Active. Allez sur /docs pour tester."}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "9.0.0"}


# ─────────────────────────────────────────────
# ROUTE 1: /split (Découpage seul, retourne base64)
# ─────────────────────────────────────────────
@app.post("/split", response_model=SplitResponse)
async def split_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")

    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        parts = split_pdf_into_parts(pdf_bytes)

        results = []
        for part in parts:
            pdf_base64 = base64.b64encode(part["pdf_bytes"]).decode("utf-8")
            results.append(SplitResult(
                file_name=part["file_name"],
                pdf_base64=pdf_base64,
            ))

        return SplitResponse(
            success=True,
            total_files=len(results),
            results=results,
        )

    except Exception as e:
        logger.exception("Erreur lors du découpage")
        return SplitResponse(success=False, error=str(e))


# ─────────────────────────────────────────────
# ROUTE 2: /extract (Extraction IA pour 1 PDF)
# ─────────────────────────────────────────────
@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")

    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        pdf_text = extract_text_from_pdf(pdf_bytes)
        extractor = get_extractor()
        invoice_data = await extractor.extract(pdf_text)
        return ExtractionResponse(success=True, data=invoice_data)

    except Exception as e:
        logger.exception("Erreur inattendue lors de l'extraction")
        return ExtractionResponse(success=False, error=str(e))


# ─────────────────────────────────────────────
# ROUTE 3: /split-and-extract (Traitement en masse)
# ─────────────────────────────────────────────
@app.post("/split-and-extract", response_model=SplitExtractResponse)
async def split_and_extract(file: UploadFile = File(...)):
    """
    Traitement en masse :
    1. Découpe le gros PDF en devis individuels
    2. Extrait les données de chaque devis via IA
    3. Retourne les résultats + base64 pour upload OneDrive
    4. Continue même si un devis échoue (rapport d'erreurs)
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")

    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # ── Étape 1 : Découpage ──
        logger.info("=== SPLIT-AND-EXTRACT : Début du découpage ===")
        parts = split_pdf_into_parts(pdf_bytes)
        total_found = len(parts)
        logger.info(f"Découpage terminé : {total_found} devis trouvés")

        if total_found == 0:
            return SplitExtractResponse(
                success=False,
                error="Aucun devis détecté dans le PDF",
            )

        # ── Étape 2 : Extraction IA par lots ──
        extractor = get_extractor()
        results: list[SplitExtractItem] = []
        errors: list[SplitExtractError] = []

        BATCH_SIZE = 5  # Traite 5 devis à la fois pour éviter la surcharge

        for batch_start in range(0, total_found, BATCH_SIZE):
            batch = parts[batch_start:batch_start + BATCH_SIZE]
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (total_found + BATCH_SIZE - 1) // BATCH_SIZE
            logger.info(f"--- Lot {batch_num}/{total_batches} ({len(batch)} devis) ---")

            # Traiter chaque devis du lot séquentiellement
            # (pour éviter de surcharger l'API OpenAI)
            for part in batch:
                file_name = part["file_name"]
                try:
                    # Extraire le texte
                    pdf_text = extract_text_from_pdf(part["pdf_bytes"])

                    # Appeler OpenAI
                    invoice_data = await extractor.extract(pdf_text)

                    # Encoder en base64 pour OneDrive
                    pdf_base64 = base64.b64encode(part["pdf_bytes"]).decode("utf-8")

                    results.append(SplitExtractItem(
                        file_name=file_name,
                        pdf_base64=pdf_base64,
                        extraction=invoice_data,
                    ))
                    logger.info(f"  ✅ {file_name} - {len(invoice_data.line_items)} items")

                except Exception as e:
                    logger.warning(f"  ❌ {file_name} - Erreur: {e}")
                    errors.append(SplitExtractError(
                        file_name=file_name,
                        error=str(e),
                        page_start=part["page_start"],
                        page_end=part["page_end"],
                    ))

            # Petite pause entre les lots pour ne pas surcharger OpenAI
            if batch_start + BATCH_SIZE < total_found:
                logger.info("Pause de 2 secondes entre les lots...")
                await asyncio.sleep(2)

        logger.info(f"=== TERMINÉ : {len(results)} OK, {len(errors)} erreurs ===")

        return SplitExtractResponse(
            success=True,
            total_found=total_found,
            total_extracted=len(results),
            total_errors=len(errors),
            results=results,
            errors=errors,
        )

    except Exception as e:
        logger.exception("Erreur fatale lors du split-and-extract")
        return SplitExtractResponse(success=False, error=str(e))
