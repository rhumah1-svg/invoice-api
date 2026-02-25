"""
Invoice/Quote PDF Extraction API - V11.0
=========================================
Base : V10.8. Modifications minimales et ciblées :
  1. LineItem     : + champ `description` (texte complet cellule, sauf CGV)
  2. /extract     : vision base64 gpt-4o-mini  (remplace pdfplumber+LLM texte)
  3. /split-and-extract : idem vision
  4. /split + /split-light : INCHANGÉS
  5. Dépendance   : pymupdf  (pip install pymupdf)
"""

import os, io, logging, base64, re, gc, json
import httpx
from typing import Optional

import pdfplumber
from pypdf import PdfReader, PdfWriter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# SCHEMAS  (identiques à v10.8 sauf LineItem)
# ═══════════════════════════════════════════════════════

class Metadata(BaseModel):
    vendor_name:    str = Field(..., description="Nom de l'entreprise cliente")
    project_name:   str = Field(..., description="Nom du projet / chantier")
    invoice_number: str = Field(..., description="Référence devis_dexxxxxx")
    date:           str = Field(..., description="Date YYYY-MM-DD")
    currency:       str = Field(..., description="Code devise ISO")

class LineItem(BaseModel):
    designation: str   = Field(..., description="Nom court du produit/service (5-8 mots max)")
    # ── NOUVEAU v11 ───────────────────────────────────────────────────────────
    description:  str  = Field(
        default="",
        description=(
            "Texte COMPLET de la cellule Description pour cet item. "
            "Inclure: sous-sections, tirets, détails techniques, normes. "
            "EXCLURE CGV: 'Travaux réalisés en semaine...', "
            "'Fourniture eau/électricité...', 'QUALIDAL n est pas responsable...'"
        )
    )
    # ─────────────────────────────────────────────────────────────────────────
    quantity:     float = Field(..., description="Quantité")
    unite:        str   = Field(..., description="Unité (ML, M2, FORF, U, etc.)")
    unit_price:   float = Field(..., description="Prix unitaire HT")

class Totals(BaseModel):
    subtotal_ht: float = Field(..., description="Total HT")
    total_tax:   float = Field(..., description="TVA")
    total_ttc:   float = Field(..., description="Total TTC")

class InvoiceData(BaseModel):
    metadata:   Metadata
    line_items: list[LineItem]
    totals:     Totals

class ExtractionResponse(BaseModel):
    success:  bool
    data:     Optional[InvoiceData] = None
    error:    Optional[str]         = None
    file_url: Optional[str]         = None

class SplitResult(BaseModel):
    file_name:  str
    pdf_base64: str

class SplitResponse(BaseModel):
    success:     bool
    total_files: int              = Field(default=0)
    results:     list[SplitResult] = Field(default=[])
    error:       Optional[str]    = None

class SplitLightItem(BaseModel):
    file_name:      str
    pdf_base64:     str
    vendor_name:    str
    project_name:   str
    invoice_number: str
    drive_path:     str

class SplitLightResponse(BaseModel):
    success:     bool
    total_files: int                 = Field(default=0)
    results:     list[SplitLightItem] = Field(default=[])
    errors:      list[dict]           = Field(default=[])
    error:       Optional[str]        = None

class SplitExtractItem(BaseModel):
    file_name:  str
    pdf_base64: str
    extraction: InvoiceData

class SplitExtractError(BaseModel):
    file_name:  str
    error:      str
    page_start: int
    page_end:   int

class SplitExtractResponse(BaseModel):
    success:         bool
    total_found:     int                   = Field(default=0)
    total_extracted: int                   = Field(default=0)
    total_errors:    int                   = Field(default=0)
    results:         list[SplitExtractItem] = Field(default=[])
    errors:          list[SplitExtractError] = Field(default=[])


# ═══════════════════════════════════════════════════════
# PDF → IMAGES BASE64  (NOUVEAU v11)
# ═══════════════════════════════════════════════════════

def pdf_bytes_to_images_b64(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
    """
    Chaque page du PDF → JPEG base64.
    Priorité : pymupdf (pip install pymupdf), fallback pdf2image.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            buf = io.BytesIO(pix.tobytes("jpeg"))
            out.append(base64.b64encode(buf.getvalue()).decode())
        doc.close()
        logger.info(f"[vision] pymupdf {len(out)} page(s) @ {dpi} DPI")
        return out
    except ImportError:
        logger.warning("[vision] pymupdf absent, essai pdf2image")

    try:
        import pdf2image
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi, fmt="jpeg")
        out = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            out.append(base64.b64encode(buf.getvalue()).decode())
        logger.info(f"[vision] pdf2image {len(out)} page(s)")
        return out
    except ImportError:
        pass

    raise RuntimeError(
        "Aucune librairie de rendu PDF.\n"
        "Solution : pip install pymupdf"
    )


# ═══════════════════════════════════════════════════════
# DÉCOUPAGE PDF  (INCHANGÉ v10.8)
# ═══════════════════════════════════════════════════════

def split_pdf_into_parts(pdf_bytes: bytes) -> list[dict]:
    split_points, devis_names = [], []
    reader_scan  = PdfReader(io.BytesIO(pdf_bytes))
    total_pages  = len(reader_scan.pages)

    for i in range(total_pages):
        try:
            txt   = reader_scan.pages[i].extract_text() or ""
            match = re.search(r"DE\d{4,10}", txt, re.IGNORECASE)
            if match:
                split_points.append(i)
                devis_names.append(f"devis_{match.group(0).lower()}")
        except Exception:
            continue

    if not split_points or split_points[0] != 0:
        split_points.insert(0, 0)
        devis_names.insert(0, "devis_inconnu")
    split_points.append(total_pages)
    del reader_scan; gc.collect()

    reader_write = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for i in range(len(split_points) - 1):
        s, e = split_points[i], split_points[i + 1]
        if s >= e:
            continue
        writer = PdfWriter()
        for j in range(s, e):
            writer.add_page(reader_write.pages[j])
        buf = io.BytesIO()
        writer.write(buf)
        parts.append({
            "file_name":  f"{devis_names[i]}.pdf",
            "pdf_bytes":  buf.getvalue(),
            "page_start": s + 1,
            "page_end":   e,
        })
        buf.close(); del writer

    del reader_write; gc.collect()
    return parts


# ═══════════════════════════════════════════════════════
# REGEX METADATA  (INCHANGÉ v10.8)
# ═══════════════════════════════════════════════════════

def extract_metadata_regex(pdf_bytes: bytes, file_name: str) -> dict:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
    except Exception as ex:
        logger.warning(f"Erreur lecture {file_name}: {ex}")
        return {"vendor_name": "INCONNU", "project_name": "INCONNU",
                "invoice_number": file_name.replace(".pdf", "")}
    finally:
        gc.collect()

    lines = text.split("\n")

    # Invoice number
    m = re.search(r"(DE\d{4,10})", text, re.IGNORECASE)
    invoice_number = f"devis_{m.group(1).lower()}" if m else file_name.replace(".pdf", "")

    # Vendor
    vendor_name = "INCONNU"
    for i, line in enumerate(lines):
        if re.match(r"^(Monsieur|Madame|M\.|Mme)\s+", line.strip(), re.IGNORECASE):
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not re.match(r"^\d", nxt):
                    vendor_name = nxt
                    break
    if vendor_name == "INCONNU":
        found = False
        for line in lines:
            if "email" in line.lower() or "fax" in line.lower():
                found = True; continue
            if found and line.strip() and line.strip().isupper() and len(line) > 3 and "DEVIS" not in line:
                vendor_name = line.strip(); break

    # Project
    project_name = "INCONNU"
    for i, line in enumerate(lines):
        if "Chantier" in line:
            for offset in range(1, 6):
                if i + offset >= len(lines): break
                nxt = lines[i + offset].strip()
                if not nxt: continue
                if any(x in nxt.lower() for x in ["de l'offre", "validité"]): continue
                dm = re.search(r"(\d{2}/\d{2}/\d{4})", nxt)
                if dm:
                    project_name = nxt[:dm.start()].replace("|", "").strip()
                    break
            if project_name != "INCONNU": break

    # Nettoyage
    for attr in [vendor_name, project_name]:
        attr = re.sub(r"[,;.\s]+$", "", attr)[:100]
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        vendor_name  = vendor_name.replace(c, '-')
        project_name = project_name.replace(c, '-')

    return {"vendor_name": vendor_name, "project_name": project_name,
            "invoice_number": invoice_number}


# ═══════════════════════════════════════════════════════
# PROMPT VISION  (NOUVEAU v11)
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT_VISION = """\
Tu es un expert en extraction de données de devis de travaux BTP français.
Ces devis sont toujours émis par la société QUALIDAL (13 avenue du Parc Alata, 60100 Creil).
Tu reçois une ou plusieurs images de pages d'un même devis.
Retourne UNIQUEMENT un JSON valide, sans texte ni markdown autour.

⚠️ RÈGLE FONDAMENTALE — ZÉRO HALLUCINATION :
  Tu dois UNIQUEMENT retranscrire ce qui est VISUELLEMENT PRÉSENT.
  Il est STRICTEMENT INTERDIT d'inventer, de déduire ou de paraphraser.
  Ne jamais fusionner le contenu de deux lignes facturables différentes.
  La description d'un item se termine EXACTEMENT là où commence la ligne suivante avec une Qté.

═══════════════════════════════════════════════════════════
RÈGLE 1 — vendor_name (entreprise CLIENTE)
═══════════════════════════════════════════════════════════
C'est le destinataire du devis (en haut à droite, ou sous les infos QUALIDAL).
- Repère le bloc d'adresse destinataire.
- Prends TOUTES les lignes désignant l'entité avant l'adresse physique (rue).
- Si tu vois la mention "C/O" (Care Of), inclus la ligne précédente ET la ligne "C/O".
  Ex: "IVANHOE LOGISTIQUE BONDOUFLE / C/O COGESTRA" -> extraire "IVANHOE LOGISTIQUE BONDOUFLE - C/O COGESTRA".

═══════════════════════════════════════════════════════════
RÈGLE 2 — project_name (chantier)
═══════════════════════════════════════════════════════════
Valeur exacte de la colonne "Chantier" dans le tableau récapitulatif.
Si la valeur est très courte (ex: "Autostore"), complète avec la mention "Adresse du chantier" en bas de page.
Si absent, cherche "Ref Cde :" en haut.

═══════════════════════════════════════════════════════════
RÈGLE 3 — invoice_number & date
═══════════════════════════════════════════════════════════
- invoice_number : Cherche "Devis DE..." en haut à droite. Format OBLIGATOIRE : "devis_de" + 8 chiffres (ex: DE00004894 -> "devis_de00004894").
- date : Colonne "Date" du tableau (format AAAA-MM-JJ).

═══════════════════════════════════════════════════════════
RÈGLE 4 — LINE ITEMS (prestations facturées)
═══════════════════════════════════════════════════════════
Un item est VALIDE et DOIT être extrait si et seulement si :
Il possède un P.U. HT ≠ 0 OU un Montant HT ≠ 0 (même si c'est un prix négatif).
OU Il possède une Qté avec une unité (même si prix = 0).

NE PAS CRÉER D'ITEM POUR :
- Les titres de zones/sections SANS prix ni quantité (ex: "Cellule A1", "CONTRÔLE QUALITÉ DALLAGE"). (Leur nom doit être intégré à la `designation` des items qui les suivent).
- Les CGV avec Qté=0 et Prix=0 (ex: "Acompte de 30%...", "Travaux réalisés en semaine...").

── designation (4 à 8 mots max) ──
- Si la cellule commence par un titre en gras : prendre le titre.
- Si la ligne est un long tiret textuel sous un titre de zone (ex: sous "CONTRÔLE QUALITÉ", un item commence par "- Prise en charge du dossier..."), la designation doit combiner le titre de zone et le début du tiret : "Contrôle Qualité - Prise en charge dossier".
- Si le nom d'une zone précède la prestation, l'inclure (ex: "Amené du matériel - Zone 1", "Impact - Cellule A2").

── description (Texte complet) ──
Copier FIDÈLEMENT tout le texte explicatif de la cellule, y compris :
- Le titre de section qui le précédait si pertinent.
- Les listes à tirets, normes (NF EN...), dimensions (30x30...).
- Les parenthèses informatives (ex: "(2 unités à 4ml)").
EXCLURE les conditions générales de vente (horaires, accès eau/élec, acompte, validité).

── quantity, unite, unit_price ──
- quantity : Valeur numérique de la colonne Qté. ATTENTION : Ne JAMAIS prendre un chiffre situé dans la description entre parenthèses (ex: dans "(2 unités à 4ml) 2,00 UNIT", la quantité est 2.00, issue de la colonne Qté).
- unite : Normaliser (FORF, M2, ML, Heures, Jours, Semaine, U).
- unit_price : Colonne P.U. HT.

═══════════════════════════════════════════════════════════
EXEMPLES DE CAS PIÈGES RÉSOLUS
═══════════════════════════════════════════════════════════

── Cas 1 : Titre collé à la prestation (ex: LTM/WARNING)
Texte visible : "LTM/WARNING AMENÉ ET REPLI DU MATÉRIEL - Zone 1" avec Qté=1, P.U=685,00.
-> C'est UN SEUL item valide (puisqu'il y a un prix).
-> designation: "LTM/WARNING - Amené et repli du matériel - Zone 1"
-> description: "LTM/WARNING AMENÉ ET REPLI DU MATÉRIEL - Zone 1"

── Cas 2 : Ligne de prestation commençant par un tiret sans titre
Texte visible : "CONTRÔLE QUALITÉ DALLAGE..." (sans prix)
Puis ligne suivante : "- Prise en charge du dossier comprenant..." avec Qté=1, P.U=27500,00.
-> designation: "Contrôle qualité - Prise en charge dossier"
-> description: "- Prise en charge du dossier comprenant échanges préliminaires, analyse des pièces marché..."

── Cas 3 : Item "Impact" très court, non gras, sans deux-points
Texte visible : "Impact" avec Qté=2, U=UNIT, P.U=120,00, suivi de "Réparation en mortier..."
-> C'est bien un item ! designation: "Impact", description: "Réparation en mortier..."

═══════════════════════════════════════════════════════════
FORMAT DE SORTIE (JSON STRICT)
═══════════════════════════════════════════════════════════
{
  "metadata": {
    "vendor_name": "...",
    "project_name": "...",
    "invoice_number": "devis_deXXXXXXXX",
    "date": "YYYY-MM-DD",
    "currency": "EUR"
  },
  "line_items": [
    {
      "designation": "...",
      "description": "...",
      "quantity": 0.0,
      "unite": "...",
      "unit_price": 0.0
    }
  ],
  "totals": {
    "subtotal_ht": 0.0,
    "total_tax": 0.0,
    "total_ttc": 0.0
  }
}
"""

# ═══════════════════════════════════════════════════════
# EXTRACTION VISION  (NOUVEAU v11)
# ═══════════════════════════════════════════════════════

async def extract_with_vision(pdf_bytes: bytes, openai_client: AsyncOpenAI) -> InvoiceData:
    """Remplace extract_text_from_pdf + extractor.extract() de v10.8."""
    try:
        images_b64 = pdf_bytes_to_images_b64(pdf_bytes, dpi=150)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"[vision] {len(images_b64)} page(s) → gpt-4o-mini")

    content = [{
        "type": "text",
        "text": f"Voici les {len(images_b64)} page(s) d'un devis Qualidal. Retourne UNIQUEMENT le JSON."
    }]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
        })

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=4000,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_VISION},
                {"role": "user",   "content": content}
            ]
        )
    except Exception as e:
        logger.exception("[vision] Erreur OpenAI")
        raise HTTPException(status_code=500, detail=f"Erreur OpenAI: {e}")

    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[vision] JSON invalide: {e}\n{raw[:400]}")
        raise HTTPException(status_code=500, detail=f"Réponse IA non parseable: {e}")

    # Sécurité invoice_number
    inv = data.get("metadata", {}).get("invoice_number", "")
    m   = re.search(r'DE(\d{4,10})', inv, re.IGNORECASE)
    if m:
        data["metadata"]["invoice_number"] = f"devis_de{m.group(1).zfill(8)}"

    try:
        return InvoiceData(**data)
    except Exception as e:
        logger.error(f"[vision] Pydantic: {e}")
        raise HTTPException(status_code=500, detail=f"Structure IA invalide: {e}")


# ═══════════════════════════════════════════════════════
# APP & ROUTES
# ═══════════════════════════════════════════════════════

app = FastAPI(title="Invoice Extraction API", version="11.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_openai_client() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY manquant")
    return AsyncOpenAI(api_key=key)


# ── Upload PDF vers Bubble File Storage ─────────────────────────────────────
async def upload_pdf_to_bubble(pdf_bytes: bytes, filename: str) -> str:
    """
    Upload un PDF vers Bubble File Storage.
    Retourne l URL S3 du fichier stocké, ou "" en cas d erreur.
    """
    bubble_token = os.getenv("BUBBLE_API_KEY", "")
    bubble_base  = os.getenv("BUBBLE_BASE_URL", "https://www.portail-qualidal.com/version-test")
    upload_url   = f"{bubble_base}/api/1.1/files/uploadprivate"

    if not bubble_token:
        logger.warning("BUBBLE_API_KEY manquant — upload PDF ignoré")
        return ""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                upload_url,
                headers={"Authorization": f"Bearer {bubble_token}"},
                files={"file": (filename, pdf_bytes, "application/pdf")},
            )
            resp.raise_for_status()
            file_url = resp.json().get("fileUrl", "")
            logger.info(f"PDF uploadé vers Bubble : {file_url[:80]}...")
            return file_url
    except Exception as e:
        logger.warning(f"Upload PDF Bubble échoué : {e}")
        return ""


@app.get("/health")
async def health():
    return {"status": "ok", "version": "11.0.0"}


# ── /split  (INCHANGÉ) ──────────────────────────────────────────────────────
@app.post("/split", response_model=SplitResponse)
async def split_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        parts   = split_pdf_into_parts(content)
        del content; gc.collect()
        results = [SplitResult(file_name=p["file_name"],
                               pdf_base64=base64.b64encode(p["pdf_bytes"]).decode())
                   for p in parts]
        del parts; gc.collect()
        return SplitResponse(success=True, total_files=len(results), results=results)
    except Exception as e:
        logger.exception("Error /split")
        return SplitResponse(success=False, error=str(e))


# ── /extract  (VISION BASE64) ───────────────────────────────────────────────
@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        filename  = file.filename or "devis.pdf"
        # Extraction IA
        data      = await extract_with_vision(pdf_bytes, get_openai_client())
        # Upload vers Bubble File Storage
        file_url  = await upload_pdf_to_bubble(pdf_bytes, filename)
        del pdf_bytes; gc.collect()
        return ExtractionResponse(success=True, data=data, file_url=file_url)
    except HTTPException as e:
        return ExtractionResponse(success=False, error=e.detail)
    except Exception as e:
        logger.exception("Error /extract")
        return ExtractionResponse(success=False, error=str(e))


# ── /split-light  (INCHANGÉ) ────────────────────────────────────────────────
@app.post("/split-light", response_model=SplitLightResponse)
async def split_light(file: UploadFile = File(...)):
    try:
        content = await file.read()
        parts   = split_pdf_into_parts(content)
        del content; gc.collect()

        results, errors = [], []
        for part in parts:
            try:
                meta   = extract_metadata_regex(part["pdf_bytes"], part["file_name"])
                vendor = meta["vendor_name"]
                proj   = meta["project_name"]
                inv    = meta["invoice_number"]
                letter = vendor[0].upper() if vendor and vendor[0].isalpha() else "#"
                path   = f"/PROD/{letter}/{vendor}/{proj}/Devis et commande/{inv}.pdf"
                b64    = base64.b64encode(part["pdf_bytes"]).decode()
                results.append(SplitLightItem(
                    file_name=part["file_name"], pdf_base64=b64,
                    vendor_name=vendor, project_name=proj,
                    invoice_number=inv, drive_path=path
                ))
            except Exception as e:
                errors.append({"file_name": part["file_name"], "error": str(e)})

        del parts; gc.collect()
        return SplitLightResponse(success=True, total_files=len(results),
                                  results=results, errors=errors)
    except Exception as e:
        logger.exception("Error /split-light")
        return SplitLightResponse(success=False, error=str(e))


# ── /split-and-extract  (VISION BASE64) ────────────────────────────────────
@app.post("/split-and-extract", response_model=SplitExtractResponse)
async def split_and_extract(file: UploadFile = File(...)):
    try:
        content       = await file.read()
        parts         = split_pdf_into_parts(content)
        del content; gc.collect()

        results, errors  = [], []
        openai_client    = get_openai_client()

        for part in parts:
            try:
                data = await extract_with_vision(part["pdf_bytes"], openai_client)
                b64  = base64.b64encode(part["pdf_bytes"]).decode()
                results.append(SplitExtractItem(
                    file_name=part["file_name"], pdf_base64=b64, extraction=data
                ))
                gc.collect()
            except Exception as e:
                errors.append(SplitExtractError(
                    file_name=part["file_name"], error=str(e),
                    page_start=part["page_start"], page_end=part["page_end"]
                ))

        del parts; gc.collect()
        return SplitExtractResponse(
            success=True,
            total_found=len(results) + len(errors),
            total_extracted=len(results),
            total_errors=len(errors),
            results=results, errors=errors
        )
    except Exception as e:
        logger.exception("Error /split-and-extract")
        return SplitExtractResponse(success=False, error=str(e))
