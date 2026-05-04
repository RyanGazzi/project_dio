from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


@dataclass
class ParsedOCRPage:
    page_number: int
    content: str
    confidence: float


def parse_scanned_pdf(pdf_path: Path, lang: str = "por+eng") -> list[ParsedOCRPage]:
    """
    Para PDFs escaneados: extrai cada página como imagem e aplica OCR.
    lang: idiomas do tesseract (por = português, eng = inglês)
    """
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc, start=1):
        # Renderiza a página em alta resolução (300 DPI)
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # OCR com dados de confiança
        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )
        text = pytesseract.image_to_string(img, lang=lang)

        # Calcula confiança média (ignora -1 que é ruído do tesseract)
        confidences = [c for c in data["conf"] if c != -1]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        pages.append(ParsedOCRPage(
            page_number=page_num,
            content=text,
            confidence=avg_conf,
        ))

    return pages