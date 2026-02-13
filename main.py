"""
Invoice/Quote PDF Extraction API - V10.0
=========================================
- /split            → Découpage seul (retourne base64)
- /extract          → Extraction IA seule (1 PDF)
- /split-and-extract → Split + Extract IA en masse
- /split-light      → Split + Metadata regex (SANS IA, rapide) pour classement OneDrive
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
# Extraction texte PDF
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
# Découpage PDF (fonction partagée)
# ─────────────────────────────────────────────

def split_pdf_into_parts(pdf_bytes: bytes) -> list[dict]:
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

    logger.info(f"Découpage : {len(split_points) - 1} devis sur {total_pages} pages.")

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

        parts.append({
            "file_name": f"{devis_names[i]}.pdf",
            "pdf_bytes": sub_pdf_io.getvalue(),
            "page_start": start_page + 1,
            "page_end": end_page,
        })

    return parts


# ─────────────────────────────────────────────
# Extraction Metadata par REGEX (sans IA)
# ─────────────────────────────────────────────

def extract_metadata_regex(pdf_bytes: bytes, file_name: str) -> dict:
    """
    Extrait vendor_name, project_name et invoice_number depuis le texte
    du PDF en utilisant uniquement des regex. Ultra rapide, pas d'IA.
    
    Structure devis Qualidal :
    - En-tête droite : "Monsieur Jean-Eudes GOHARD" puis "IDEC" (entreprise)
    - Tableau : Chantier | Date | ... avec "AREFIM - REIMS (51)" sous Chantier
    - Numéro : DE00001898 en haut à droite
    """
    text = ""
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

    lines = text.split("\n")

    # ── Invoice Number ──
    invoice_match = re.search(r"(DE\d{4,10})", text, re.IGNORECASE)
    if invoice_match:
        invoice_number = f"devis_{invoice_match.group(1).lower()}"
    else:
        invoice_number = file_name.replace(".pdf", "")

    # ── Vendor Name (entreprise cliente) ──
    # Structure Qualidal : "Monsieur/Madame Prénom NOM" puis ligne suivante = entreprise
    vendor_name = "INCONNU"
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Chercher "Monsieur" ou "Madame" suivi d'un nom
        if re.match(r"^(Monsieur|Madame|M\.|Mme)\s+", line_stripped, re.IGNORECASE):
            # La ligne suivante est souvent le nom de l'entreprise
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Vérifier que c'est pas une adresse (pas de chiffre au début)
                if next_line and not re.match(r"^\d", next_line):
                    vendor_name = next_line
                    break
    
    # Fallback : chercher une ligne courte en majuscules après le bloc Qualidal
    if vendor_name == "INCONNU":
        found_qualidal_block = False
        for line in lines:
            line_stripped = line.strip()
            # Détecter qu'on est après le bloc Qualidal (après "Email" ou "Fax")
            if "email" in line_stripped.lower() or "fax" in line_stripped.lower():
                found_qualidal_block = True
                continue
            if found_qualidal_block and line_stripped:
                # Ligne courte en majuscules = probablement l'entreprise
                if (len(line_stripped) > 2 and len(line_stripped) < 50 and
                    line_stripped == line_stripped.upper() and
                    not any(kw in line_stripped for kw in [
                        "DEVIS", "FACTURE", "TVA", "HT", "TTC", "PAGE", 
                        "DATE", "TOTAL", "QUALIDAL", "CREIL", "CS ", 
                        "CEDEX", "RUE", "AVENUE", "BOULEVARD"
                    ])):
                    vendor_name = line_stripped
                    break

    # ── Project Name (chantier) ──
    # Le chantier est dans le tableau juste après l'en-tête "Chantier"
    project_name = "INCONNU"
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Chercher la ligne d'en-tête du tableau contenant "Chantier"
        if "Chantier" in line_stripped and "Date" in line_stripped:
            # La ligne suivante contient les valeurs du tableau
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Le chantier est au début de la ligne, avant la date
                # Format : "AREFIM - REIMS (51) 09/02/2021 ..."
                date_match = re.search(r"\d{2}/\d{2}/\d{4}", next_line)
                if date_match:
                    project_name = next_line[:date_match.start()].strip()
                else:
                    # Prendre le premier segment significatif
                    project_name = next_line.split("  ")[0].strip()
            break
    
    # Fallback : chercher après "Chantier :" ou "Chantier:"
    if project_name == "INCONNU":
        chantier_match = re.search(
            r"Chantier\s*[:\-]?\s*(.+?)(?:\d{2}/\d{2}/\d{4}|$)", 
            text, re.IGNORECASE
        )
        if chantier_match:
            project_name = chantier_match.group(1).strip()

    # ── Nettoyage final ──
    vendor_name = re.sub(r"[,;.\s]+$", "", vendor_name)[:100]
    project_name = re.sub(r"[,;.\s]+$", "", project_name)[:100]
    
    # Remplacer les caractères interdits dans les noms de dossiers OneDrive
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        vendor_name = vendor_name.replace(char, '-')
        project_name = project_name.replace(char, '-')

    logger.info(f"  Regex: vendor='{vendor_name}', project='{project_name}', invoice='{invoice_number}'")

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
        logger.info("Extraction réussie: %d items trouvés", len(parsed.line_items))
        return parsed


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Invoice Extraction & Split API",
    version="10.0.0",
    description="V10: Added /split-light for fast regex-based splitting with OneDrive path generation.",
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
    return {"message": "API Active. Allez sur /docs pour tester."}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "10.0.0"}


# ─────────────────────────────────────────────
# ROUTE 1: /split (Découpage seul)
# ─────────────────────────────────────────────
@app.post("/split", response_model=SplitResponse)
async def split_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        parts = split_pdf_into_parts(pdf_bytes)
        results = [
            SplitResult(
                file_name=p["file_name"],
                pdf_base64=base64.b64encode(p["pdf_bytes"]).decode("utf-8"),
            )
            for p in parts
        ]
        return SplitResponse(success=True, total_files=len(results), results=results)
    except Exception as e:
        logger.exception("Erreur /split")
        return SplitResponse(success=False, error=str(e))


# ─────────────────────────────────────────────
# ROUTE 2: /extract (Extraction IA - 1 PDF)
# ─────────────────────────────────────────────
@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        pdf_text = extract_text_from_pdf(pdf_bytes)
        extractor = get_extractor()
        invoice_data = await extractor.extract(pdf_text)
        return ExtractionResponse(success=True, data=invoice_data)
    except Exception as e:
        logger.exception("Erreur /extract")
        return ExtractionResponse(success=False, error=str(e))


# ─────────────────────────────────────────────
# ROUTE 3: /split-light (Split + Regex, SANS IA)
# Pour le WF1 masse : classe les PDFs dans OneDrive
# ─────────────────────────────────────────────
@app.post("/split-light", response_model=SplitLightResponse)
async def split_light(file: UploadFile = File(...)):
    """
    Découpe un gros PDF en devis individuels et extrait les metadata
    par REGEX (sans IA). Ultra rapide. Retourne le chemin OneDrive
    pour chaque devis.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")

    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        logger.info("=== SPLIT-LIGHT : Début ===")
        parts = split_pdf_into_parts(pdf_bytes)
        logger.info(f"{len(parts)} devis détectés")

        results: list[SplitLightItem] = []
        errors: list[dict] = []

        for part in parts:
            try:
                # Extraire metadata par regex (rapide, pas d'IA)
                meta = extract_metadata_regex(part["pdf_bytes"], part["file_name"])

                vendor_name = meta["vendor_name"]
                project_name = meta["project_name"]
                invoice_number = meta["invoice_number"]

                # Construire le chemin OneDrive
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
                logger.info(f"  ✅ {part['file_name']} → {drive_path}")

            except Exception as e:
                logger.warning(f"  ❌ {part['file_name']}: {e}")
                errors.append({
                    "file_name": part["file_name"],
                    "error": str(e),
                    "page_start": part["page_start"],
                    "page_end": part["page_end"],
                })

        logger.info(f"=== SPLIT-LIGHT TERMINÉ : {len(results)} OK, {len(errors)} erreurs ===")

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
# ROUTE 4: /split-and-extract (Split + IA en masse)
# ─────────────────────────────────────────────
@app.post("/split-and-extract", response_model=SplitExtractResponse)
async def split_and_extract(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Expected a PDF file")

    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        logger.info("=== SPLIT-AND-EXTRACT : Début ===")
        parts = split_pdf_into_parts(pdf_bytes)
        total_found = len(parts)

        if total_found == 0:
            return SplitExtractResponse(success=False, error="Aucun devis détecté")

        extractor = get_extractor()
        results: list[SplitExtractItem] = []
        errors: list[SplitExtractError] = []

        BATCH_SIZE = 5
        for batch_start in range(0, total_found, BATCH_SIZE):
            batch = parts[batch_start:batch_start + BATCH_SIZE]
            for part in batch:
                try:
                    pdf_text = extract_text_from_pdf(part["pdf_bytes"])
                    invoice_data = await extractor.extract(pdf_text)
                    pdf_base64 = base64.b64encode(part["pdf_bytes"]).decode("utf-8")
                    results.append(SplitExtractItem(
                        file_name=part["file_name"],
                        pdf_base64=pdf_base64,
                        extraction=invoice_data,
                    ))
                    logger.info(f"  ✅ {part['file_name']}")
                except Exception as e:
                    logger.warning(f"  ❌ {part['file_name']}: {e}")
                    errors.append(SplitExtractError(
                        file_name=part["file_name"],
                        error=str(e),
                        page_start=part["page_start"],
                        page_end=part["page_end"],
                    ))

            if batch_start + BATCH_SIZE < total_found:
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
        logger.exception("Erreur /split-and-extract")
        return SplitExtractResponse(success=False, error=str(e))
