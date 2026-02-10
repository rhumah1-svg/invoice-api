"""
Invoice/Quote PDF Extraction API
=================================
FastAPI service that receives a PDF file, extracts text via pdfplumber,
sends it to OpenAI with structured outputs, and returns a strictly
formatted JSON ready for Bubble integration.
"""

import os
import io
import logging
from datetime import date
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
# Pydantic schemas  (= contract with Bubble)
# ─────────────────────────────────────────────

class Metadata(BaseModel):
    vendor_name: str = Field(..., description="Company or vendor name on the invoice")
    invoice_number: str = Field(..., description="Invoice or quote reference number")
    date: str = Field(..., description="Document date in YYYY-MM-DD format")
    currency: str = Field(..., description="ISO 4217 currency code, e.g. EUR, USD")


class LineItem(BaseModel):
    description: str = Field(..., description="Full product/service description (multi-line merged)")
    quantity: float = Field(..., description="Quantity ordered")
    unit_price: float = Field(..., description="Unit price excluding tax")
    total_price: float = Field(..., description="Line total excluding tax")
    sku_reference: Optional[str] = Field(None, description="SKU or product reference if present")


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
    structure when possible.  Falls back to raw text if table
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
        raise ValueError("No text could be extracted from the PDF. It may be image-based (scanned).")
    return full_text


# ─────────────────────────────────────────────
# LLM extraction service (OpenAI Structured Outputs)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert document parser specialising in invoices and quotes.

TASK:
Given the raw text extracted from a PDF invoice or quote, return a single
JSON object that matches the provided schema **exactly**.

RULES:
1. Identify fields semantically – never rely on position. Look for labels
   like "Date", "Facture N°", "Total HT", "TVA", "TTC", "Montant", etc.
2. If a product description spans multiple lines, merge them into one
   coherent `description` string.
3. Dates MUST be returned as YYYY-MM-DD. Convert from any format.
4. Monetary values MUST be plain floats (no currency symbols, no spaces).
5. If a field is truly absent from the document, use "" for strings and
   0.0 for numbers. Never invent data.
6. `currency` should be the ISO 4217 code (EUR, USD, GBP …). Infer from
   symbols (€ → EUR, $ → USD) or context.
7. `sku_reference` is optional – set to null if no SKU/reference exists.
8. Prefer values explicitly written in the document over computed ones.
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
        logger.info("Calling OpenAI (%s) with %d chars of text", self.model, len(pdf_text))

        # Truncate very long documents to stay within context limits
        max_chars = 60_000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "\n\n[...TRUNCATED...]"

        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0,  # deterministic
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract data from this document:\n\n{pdf_text}"},
            ],
            response_format=InvoiceData,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            # Should not happen with structured outputs, but guard anyway
            raise ValueError("OpenAI returned an unparseable response")

        logger.info("Extraction successful: %s items found", len(parsed.line_items))
        return parsed


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Invoice Extraction API",
    version="1.0.0",
    description="Upload a PDF invoice/quote → get structured JSON back.",
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
    return {"status": "ok"}


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
        logger.info("Extracting text from %s (%d bytes)", file.filename, len(pdf_bytes))
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
        return ExtractionResponse(success=False, error=f"Internal error: {type(e).__name__}: {e}")
