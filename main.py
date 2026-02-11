"""
Invoice/Quote PDF Extraction API
=================================
FastAPI service that receives a PDF file, extracts text via pdfplumber,
sends it to OpenAI with structured outputs, and returns a strictly
formatted JSON ready for Bubble integration.

V2 - Added:
- Simplified LineItem (designation, quantity, unite, unit_price only)
- No description extraction (to reduce AI load)
- Product normalization for consistent naming across quotes
"""

import os
import io
import logging
from typing import Optional

import pdfplumber
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
# Pydantic schemas (= contract with Bubble)
# ─────────────────────────────────────────────

class Metadata(BaseModel):
    vendor_name: str = Field(..., description="Company or vendor name on the invoice")
    invoice_number: str = Field(..., description="Invoice or quote reference number")
    date: str = Field(..., description="Document date in YYYY-MM-DD format")
    currency: str = Field(..., description="ISO 4217 currency code, e.g. EUR, USD")


class LineItem(BaseModel):
    designation: str = Field(
        ...,
        description=(
            "Normalized product/service name. Use a clean, consistent short name "
            "without descriptions. Example: 'Réparation joint épaufré' not "
            "'Réparation joint épaufré : Sciage de part et d'autre...'"
        ),
    )
    quantity: float = Field(..., description="Quantity ordered")
    unite: str = Field(
        ...,
        description=(
            "Unit of measurement as written on the document. "
            "Common values: ML, M2, M3, FORF, U, KG, L, ENS, H, JR"
        ),
    )
    unit_price: float = Field(..., description="Unit price excluding tax (PU HT)")


class Totals(BaseModel):
    subtotal_ht: float = Field(..., description="Subtotal before tax (HT)")
    total_tax: float = Field(..., description="Total tax amount (TVA)")
    total_ttc: float = Field(..., description="Grand total including tax (TTC)")


class InvoiceData(BaseModel):
    metadata: Metadata
    line_items: list[LineItem]
    totals: Totals


class ExtractionResponse(BaseModel):
    success: bool
    data: Optional[InvoiceData] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
# PDF text extraction (pdfplumber)
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from every page of a PDF while preserving table
    structure when possible. Falls back to raw text if table
    extraction yields nothing.
    """
    all_text_parts: list[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text_parts: list[str] = [f"--- Page {page_num} ---"]

            # Try table extraction first (preserves columns)
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
                # Also grab text outside of tables
                non_table_text = page.extract_text()
                if non_table_text:
                    page_text_parts.append(non_table_text)
            else:
                # Fallback: full-page text extraction
                raw = page.extract_text()
                if raw:
                    page_text_parts.append(raw)

            all_text_parts.append("\n".join(page_text_parts))

    full_text = "\n\n".join(all_text_parts)
    if not full_text.strip():
        raise ValueError(
            "No text could be extracted from the PDF. It may be image-based (scanned)."
        )
    return full_text


# ─────────────────────────────────────────────
# LLM extraction service (OpenAI Structured Outputs)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert document parser specialising in French invoices and quotes
(devis, factures) for construction, building repair, and technical services.

TASK:
Given the raw text extracted from a PDF invoice or quote, return a single
JSON object that matches the provided schema **exactly**.

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

RULES FOR METADATA:
1. `vendor_name`: the company issuing the invoice/quote
2. `invoice_number`: the document reference (Devis N°, Facture N°, Ref, etc.)
3. `date`: document date in YYYY-MM-DD format. Convert from any format.
4. `currency`: ISO 4217 code. Infer from symbols (€ → EUR)

RULES FOR TOTALS:
1. Use values explicitly written in the document
2. `subtotal_ht`: Total before tax
3. `total_tax`: Total TVA amount
4. `total_ttc`: Grand total including tax

GENERAL RULES:
- Monetary values MUST be plain floats (no currency symbols, no spaces)
- If a field is truly absent, use "" for strings, 0.0 for numbers
- Never invent data
- Prefer values explicitly written in the document over computed ones
"""


class InvoiceExtractor:
    """Handles the OpenAI API call with structured outputs."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def extract(self, pdf_text: str) -> InvoiceData:
        """
        Send extracted PDF text to OpenAI and get back a validated
        InvoiceData object using Structured Outputs (response_format).
        """
        logger.info(
            "Calling OpenAI (%s) with %d chars of text", self.model, len(pdf_text)
        )

        # Truncate very long documents to stay within context limits
        max_chars = 60_000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "\n\n[...TRUNCATED...]"

        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0,  # deterministic
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract data from this document:\n\n{pdf_text}",
                },
            ],
            response_format=InvoiceData,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned an unparseable response")

        logger.info(
            "Extraction successful: %d line items found", len(parsed.line_items)
        )
        return parsed


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Invoice Extraction API",
    version="2.0.0",
    description=(
        "Upload a PDF invoice/quote → get structured JSON back. "
        "V2: simplified line items (designation, quantity, unite, unit_price), "
        "normalized product names for consistent matching."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-init extractor so missing env var doesn't crash at import time
_extractor: Optional[InvoiceExtractor] = None


def get_extractor() -> InvoiceExtractor:
    global _extractor
    if _extractor is None:
        _extractor = InvoiceExtractor()
    return _extractor


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    """
    Upload a PDF invoice/quote file.
    Returns structured JSON with metadata, line items, and totals.
    """
    # ── Validate file type ──
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a PDF file, got {file.content_type}",
        )

    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # ── Step 1: Extract text ──
        logger.info(
            "Extracting text from %s (%d bytes)", file.filename, len(pdf_bytes)
        )
        pdf_text = extract_text_from_pdf(pdf_bytes)
        logger.info("Extracted %d characters of text", len(pdf_text))

        # ── Step 2: LLM structured extraction ──
        extractor = get_extractor()
        invoice_data = await extractor.extract(pdf_text)

        return ExtractionResponse(success=True, data=invoice_data)

    except ValueError as e:
        logger.warning("Extraction error: %s", e)
        return ExtractionResponse(success=False, error=str(e))
    except Exception as e:
        logger.exception("Unexpected error during extraction")
        return ExtractionResponse(
            success=False, error=f"Internal error: {type(e).__name__}: {e}"
        )
