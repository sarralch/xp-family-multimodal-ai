"""OCR of cosmetic labels. The Tesseract binary path comes from settings, not a
hardcoded Windows path, so the same code runs in Docker and on any OS."""

from __future__ import annotations

from typing import BinaryIO

from PIL import Image, ImageOps

from xp_family.config import Settings
from xp_family.observability import trace


class OCRUnavailableError(RuntimeError):
    pass


def extract_text(image: BinaryIO, settings: Settings, lang: str = "fra+eng") -> str:
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise OCRUnavailableError("Install the 'ocr' extra: uv sync --extra ocr") from exc
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    # Grayscale + auto-contrast is a cheap, reliable gain on glossy packaging photos.
    picture = ImageOps.autocontrast(ImageOps.grayscale(Image.open(image)))
    with trace("ocr", lang=lang, width=picture.width, height=picture.height) as span:
        try:
            text = pytesseract.image_to_string(picture, lang=lang)
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRUnavailableError("Tesseract binary not found; set TESSERACT_CMD") from exc
        except pytesseract.TesseractError:
            # French language data is often missing on local installs; English still
            # reads INCI names, which are Latin/English by regulation.
            text = pytesseract.image_to_string(picture, lang="eng")
            span.set(lang_fallback="eng")
        span.set(chars=len(text))
    return text.strip()
