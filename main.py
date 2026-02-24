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
    success: bool
    data:    Optional[InvoiceData] = None
    error:   Optional[str]        = None

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

═══════════════════════════════════════════════════════════
STRUCTURE GÉNÉRALE D'UN DEVIS QUALIDAL
═══════════════════════════════════════════════════════════

Page 1 en-tête (haut gauche) : logo + adresse Qualidal
Page 1 en-tête (haut droite) : destinataire → "Monsieur/Madame Prénom NOM" puis nom entreprise
Tableau récapitulatif       : colonnes Chantier | Date | Date validité | Condition règlement | N° TVA
Tableau des prestations     : colonnes Description | Qté | U | P.U. HT | (R%) | Montant HT | TVA
Dernière page bas           : Totaux + "Adresse du chantier" + "BON POUR ACCORD"

Le tableau des prestations peut s'étendre sur plusieurs pages.
La colonne R% (remise) peut être absente.
Il peut y avoir des sous-totaux ou des séparateurs de zones (ex: "Zone 1", "Cellule A").

═══════════════════════════════════════════════════════════
RÈGLE 1 — vendor_name (entreprise CLIENTE)
═══════════════════════════════════════════════════════════

C'est l'entreprise À QUI le devis est adressé. Jamais Qualidal.

Méthode principale :
  Repère "Monsieur", "Madame", "M.", "Mme" suivi d'un prénom et nom.
  La ligne SUIVANTE est le nom de l'entreprise cliente.
  Ex: "Monsieur Jean-Eudes Gohard" → ligne suivante → "IDEC"

Méthode fallback (si pas de civilité) :
  Après le bloc Qualidal (après "Email :" ou "Fax :"),
  chercher la première ligne en majuscules qui n'est pas :
  une adresse, un numéro de téléphone, "DEVIS", "FACTURE", ou une ville connue.

═══════════════════════════════════════════════════════════
RÈGLE 2 — project_name (chantier)
═══════════════════════════════════════════════════════════

Source primaire : valeur de la colonne "Chantier" dans le tableau récapitulatif.

Cas 1 — valeur explicite et complète :
  "AREFIM - REIMS (51)" → garder tel quel
  "LOZENNES (59)" → garder tel quel

Cas 2 — valeur courte ou nom propre seul (ex: "Autostore", "Amazon", "Lidl") :
  Chercher "Adresse du chantier" ou "Ref Cde" en bas de page pour compléter.
  Ex: Chantier="Autostore" + Adresse="Ussel" + Ref="Projet Autostore Ussel (19)" → "Autostore Ussel (19)"
  Ex: Chantier="Autostore" + Adresse="Ussel" (sans département) → "Autostore Ussel"

Cas 3 — valeur vide :
  Utiliser "Ref Cde" si présent, sinon "INCONNU"

═══════════════════════════════════════════════════════════
RÈGLE 3 — invoice_number
═══════════════════════════════════════════════════════════

Cherche la référence commençant par "DE" suivie de 4 à 10 chiffres.
Elle apparaît en haut à droite du devis, souvent dans un encart "Devis".
Exemples trouvés dans les documents : DE00001898, DE00004001, DE00005445

Format de sortie OBLIGATOIRE : "devis_de" + numéro en minuscules, sur 8 chiffres minimum.
  DE00004001 → "devis_de00004001"
  DE1898     → "devis_de00001898"  (compléter avec des zéros à gauche)

═══════════════════════════════════════════════════════════
RÈGLE 4 — date
═══════════════════════════════════════════════════════════

Colonne "Date" du tableau récapitulatif (date d'émission du devis, pas la date de validité).
Convertir JJ/MM/AAAA → AAAA-MM-JJ.
Si absente : "".

═══════════════════════════════════════════════════════════
RÈGLE 5 — LINE ITEMS (prestations facturées)
═══════════════════════════════════════════════════════════

── Qu'est-ce qu'un item valide ? ──────────────────────────

Un item valide est une ligne du tableau des prestations qui remplit AU MOINS UNE de ces conditions :
  • P.U. HT est un nombre différent de 0 (positif ou négatif)
  • Montant HT est un nombre différent de 0 (positif ou négatif)
  • Qté est renseignée ET l'unité est renseignée (même si prix=0, c'est une prestation réelle)

Ne PAS créer d'item pour :
  • Lignes entièrement à 0,00 (séparateurs visuels vides)
  • Sous-totaux ou totaux intermédiaires
  • Lignes de type "section" sans quantité ni prix (ex: "Installation et livraison chantier :")
  • Conditions générales (voir liste ci-dessous)
  • La ligne de total général (Total HT, Total TVA, Total TTC)

── designation ─────────────────────────────────────────────

Règle : extraire le NOM COURT de la prestation. 4 à 8 mots maximum.

Si la cellule contient du texte court (une seule ligne) : prendre tel quel.
  Ex: "Réparation joint épaufré" → "Réparation joint épaufré"

Si la cellule contient du texte long avec une première phrase/ligne en gras :
  prendre uniquement cette première phrase/ligne en gras comme designation.
  Ex: cellule commence par "Mise en conformité de la dalle, par application d'un coulis hydraulique"
      → designation = "Mise en conformité dalle coulis hydraulique"

Si la prestation est une remise/ristourne/escompte :
  → designation = "Remise exceptionnelle" (ou libellé exact s'il est court)

Conserver les identifiants de zone s'ils sont présents :
  Ex: "Grenaillage surface - Zone 1", "Ragréage béton - Cellule A3"

── description ─────────────────────────────────────────────

Règle : copier FIDÈLEMENT tout le texte de la cellule Description pour cet item.

Inclure absolument :
  • Le texte principal (même s'il est long)
  • Les sous-sections avec titres (ex: "Préparation de surface :", "Contrôles de réception:")
  • Les listes à tirets et leur contenu
  • Les surfaces, dimensions, normes (ex: "815 m²", "NF EN 13813", "ARO.5")
  • Les spécifications techniques

Ne PAS inclure dans description (ces textes sont des CGV, pas des prestations) :
  • Tout paragraphe sur les horaires de travail (lundi-vendredi, 8h-18h)
  • Tout paragraphe sur la fourniture d'eau ou d'électricité par le client
  • Tout paragraphe de non-responsabilité Qualidal sur le béton
  • Les mentions légales (Siret, RCS, capital)
  • "Devis gratuit. Les prix TTC sont établis..."
  • Les pénalités de retard de paiement

Pour une remise/ristourne : description = "" (chaîne vide)

── quantity ────────────────────────────────────────────────

Valeur numérique de la colonne Qté.
Si la colonne est vide ou absente : 0.0
Toujours un float : 1 → 1.0, 815 → 815.0

── unite ───────────────────────────────────────────────────

Normaliser selon ce tableau :
  FORF / Forfait / FF / Ens / Ensemble      → "FORF"
  M2 / m² / M²                             → "M2"
  ML / ml / m / Lin                         → "ML"
  H / Heure / Heures / HR                  → "Heures"
  J / Jour / Jours                         → "Jours"
  Sem / Semaine                             → "Semaine"
  U / unité / pce / pièce / ens (ambigu)   → "U"
  Vide ou non reconnu                       → "U"

── unit_price ──────────────────────────────────────────────

Valeur de la colonne P.U. HT (prix unitaire hors taxes).
Peut être négatif (remise). Ex: -130.0
Si vide : 0.0

── Gestion des zones et sous-sections (CAS FRÉQUENT) ───────

Les devis Qualidal sont souvent structurés par zones géographiques ou bâtiments.
Exemples de titres de zone rencontrés :
  "ZONE 1 - Entrepôt principal"       "Bâtiment A"
  "Cellule A1 / Cellule A2"           "Zone stockage"
  "Bâtiment B - Extension"            "Nef 1", "Nef 2"
  "Parking extérieur"                 "Hall d entrée"

RÈGLE — Comment identifier un titre de zone :
  Pas de valeur dans les colonnes Qté, U, P.U. HT, Montant HT
  Texte en majuscules ou semi-majuscules, souvent court (1 à 5 mots)
  Peut avoir une ligne de séparation visuelle (bordure, fond gris)
  → NE PAS créer d item pour ces lignes

RÈGLE — Les prestations à l intérieur d une zone SONT des items normaux.
  Intégrer le nom de la zone dans designation pour lever toute ambiguïté :
  Ex: zone = "ZONE 1" + prestation = "Grenaillage surface"
      → designation = "Grenaillage surface - Zone 1"
  Ex: zone = "Cellule A2" + prestation = "Ragréage béton"
      → designation = "Ragréage béton - Cellule A2"

RÈGLE — Sous-totaux de zone :
  Certains devis affichent un sous-total par zone ("Sous-total Zone 1 : 12 500,00 EUR").
  → NE PAS créer d item pour ces lignes de sous-total.

RÈGLE — Colonne R% (remise par ligne) parfois absente :
  Si la colonne R% n existe pas dans le tableau, ignorer simplement.
  Ne pas chercher à calculer ou inventer une remise.
  Les remises globales apparaissent comme une ligne dédiée avec un montant négatif.

═══════════════════════════════════════════════════════════
RÈGLE 6 — totals
═══════════════════════════════════════════════════════════

Lire les valeurs dans le tableau de totaux en bas de dernière page.
  subtotal_ht : ligne "Total HT"
  total_tax   : ligne "Total TVA"
  total_ttc   : ligne "Total TTC" ou "Net à payer"
Si une valeur est absente : 0.0

═══════════════════════════════════════════════════════════
FORMAT DE SORTIE — JSON STRICT, SANS MARKDOWN
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
}"""


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
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}
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
        content = await file.read()
        data    = await extract_with_vision(content, get_openai_client())
        del content; gc.collect()
        return ExtractionResponse(success=True, data=data)
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
