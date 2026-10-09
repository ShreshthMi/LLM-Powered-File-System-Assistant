"""Turn the bytes of PDF, DOCX and plain-text files into text (used by ``fs_tools.read_file``).

Every extractor returns an :class:`Extraction` or raises :class:`ExtractionError` with a
message meant for the user or the model. :func:`extract` picks the reader from the file's
real content first and its extension second, so a mislabelled file is still read correctly.
"""

from __future__ import annotations

import codecs
import io
import logging
import re
import threading
import zipfile
from typing import Any, Callable, NamedTuple

import docx
from pypdf import PdfReader
from pypdf.errors import DependencyError, EmptyFileError

TEXT_EXTENSIONS = frozenset({".txt", ".md", ".csv", ".json"})
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}

MAX_PDF_PAGES = 300
PDF_TIME_LIMIT_SECONDS = 20.0  # a decompression-bomb PDF must not freeze the assistant
MAX_DOCX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024  # guards against zip bombs
BINARY_SNIFF_BYTES = 8192

LEGACY_DOC_MESSAGE = "This is a legacy Word .doc file, which isn't supported. Save it as .docx or PDF first."

# pypdf logs every quirk of a malformed PDF; the tools report problems themselves.
logging.getLogger("pypdf").setLevel(logging.ERROR)

_BOMS = (  # UTF-32 first: its little-endian BOM starts with the UTF-16 one
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_PDF_SIGNATURE = re.compile(rb"%PDF-\d")  # a version digit, so a text note starting "%PDF-" isn't a PDF

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_W_T, _W_TAB, _W_BR, _W_CR, _W_P, _W_TBL, _W_TR, _W_TC, _W_NB_HYPHEN = (
    _W + tag for tag in ("t", "tab", "br", "cr", "p", "tbl", "tr", "tc", "noBreakHyphen")
)
# Formatting/property elements carry no visible text (w:pPr/w:tabs/w:tab is a tab stop,
# not a tab character); w:moveFrom duplicates text that was moved elsewhere.
_W_SKIP = frozenset(
    _W + tag
    for tag in ("pPr", "rPr", "tblPr", "tblPrEx", "trPr", "tcPr", "sectPr", "sdtPr", "sdtEndPr", "tblGrid", "moveFrom")
)
# Text boxes are stored twice (DrawingML + a legacy VML copy under mc:Fallback).
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


class ExtractionError(Exception):
    """The file can't be turned into text; the message says why."""


class Extraction(NamedTuple):
    text: str
    metadata: dict[str, Any]
    warnings: tuple[str, ...]


def _megabytes(size: int) -> str:
    """Format a byte count as megabytes for user-facing messages."""
    return f"{size / (1024 * 1024):.1f} MB"


def sniff_format(data: bytes) -> str | None:
    """Detect a file's real format from its first bytes, regardless of its name."""
    head = data[:8]
    if _PDF_SIGNATURE.match(head):
        return ".pdf"
    if head.startswith(b"PK\x03\x04"):
        return ".docx"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return ".doc"
    return None


# --------------------------------------------------------------------------- text


def _decode_text(raw: bytes) -> tuple[str, list[str]]:
    """Decode BOM-aware text or a fallback encoding; reject binary input."""
    for bom, encoding in _BOMS:
        if raw.startswith(bom):
            return raw.decode(encoding, errors="replace"), []
    if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
        raise ExtractionError("The file looks binary, not plain text (it contains NUL bytes). Is the extension wrong?")
    try:
        return raw.decode("utf-8"), []
    except UnicodeDecodeError:
        warning = "The file is not valid UTF-8; it was decoded as Windows-1252, so a few characters may be off."
        return raw.decode("cp1252", errors="replace"), [warning]


def _extract_text(data: bytes) -> Extraction:
    """Extract plain text and report empty or unusual encoding warnings."""
    text, warnings = _decode_text(data)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        warnings.append("The file is empty." if not data else "The file contains only whitespace.")
    return Extraction(text, {}, tuple(warnings))


# ---------------------------------------------------------------------------- PDF


def _unlock_pdf(reader: PdfReader) -> None:
    """Open PDFs that are encrypted with an empty password (permissions-only protection)."""
    try:
        unlocked = reader.decrypt("")
    except DependencyError:
        raise ExtractionError(
            "The PDF is AES-encrypted and the 'cryptography' package is missing (pip install -r requirements.txt)."
        ) from None
    except Exception as exc:  # pypdf raises assorted errors for unusual encryption setups
        raise ExtractionError(f"The PDF is encrypted and could not be decrypted ({type(exc).__name__}).") from None
    if not unlocked:
        raise ExtractionError("The PDF is password-protected. Provide an unlocked copy to read it.")


def _open_pdf(data: bytes) -> PdfReader:
    """Open a PDF and handle encryption or damaged input."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            _unlock_pdf(reader)
        return reader
    except ExtractionError:
        raise
    except EmptyFileError:
        raise ExtractionError("The PDF is empty (0 bytes).") from None
    except Exception as exc:  # PdfReadError and friends for damaged files
        raise ExtractionError(f"Could not open the PDF - the file looks damaged ({type(exc).__name__}: {exc}).") from None


def _read_pdf_pages(reader: PdfReader, count: int) -> tuple[list[str], list[str], bool]:
    """Extract page texts on a worker thread, giving up after PDF_TIME_LIMIT_SECONDS.

    A small PDF can decompress into megabytes of drawing commands; without a time limit
    one such file would freeze the assistant. Returns (page texts, failed page numbers,
    finished in time).
    """
    pages: list[str] = []
    failed: list[str] = []
    stop = threading.Event()

    def work() -> None:
        """Read page text until extraction completes or the worker is stopped."""
        for index in range(count):
            if stop.is_set():
                return
            try:
                pages.append((reader.pages[index].extract_text() or "").strip())
            except Exception:  # one broken page shouldn't hide the rest of the document
                failed.append(str(index + 1))
                pages.append("")

    worker = threading.Thread(target=work, name="pdf-text", daemon=True)
    worker.start()
    worker.join(PDF_TIME_LIMIT_SECONDS)
    stop.set()
    return list(pages), list(failed), not worker.is_alive()


def _extract_pdf(data: bytes) -> Extraction:
    """Extract bounded PDF content, page counts and partial-read warnings."""
    reader = _open_pdf(data)
    try:
        total = len(reader.pages)
    except Exception as exc:
        raise ExtractionError(f"Could not read the PDF's pages - the file looks damaged ({type(exc).__name__}).") from None
    pages, failed, finished = _read_pdf_pages(reader, min(total, MAX_PDF_PAGES))
    text = "\n\n".join(page for page in pages if page)
    warnings = []
    if failed:
        warnings.append(f"Text could not be extracted from page(s) {', '.join(failed)}.")
    if not finished:
        warnings.append(
            f"Stopped after {len(pages)} of {total} pages: the PDF took more than {PDF_TIME_LIMIT_SECONDS:.0f} s "
            "to process (it may be malformed)."
        )
    elif total > MAX_PDF_PAGES:
        warnings.append(f"Only the first {MAX_PDF_PAGES} of {total} pages were read.")
    if finished and not text:
        warnings.append("No extractable text - the PDF may be a scanned image, which would need OCR.")
    metadata: dict[str, Any] = {"page_count": total}
    if len(pages) < total:
        metadata["pages_read"] = len(pages)
    return Extraction(text, metadata, tuple(warnings))


# --------------------------------------------------------------------------- DOCX


def _nearest(element: Any, tag: str) -> Any:
    """Return the closest XML ancestor with the requested tag."""
    return next(element.iterancestors(tag), None)


def _docx_text(element: Any) -> str:
    """Flatten WordprocessingML: paragraphs become lines, table rows become 'a | b' lines."""
    pieces: list[str] = []
    for child in element:
        tag = child.tag
        if tag == _W_T:
            pieces.append(child.text or "")
        elif tag == _W_TAB:
            pieces.append("\t")
        elif tag in (_W_BR, _W_CR):
            pieces.append("\n")
        elif tag == _W_NB_HYPHEN:
            pieces.append("-")
        elif tag == _W_P:
            pieces.append(_docx_text(child) + "\n")
        elif tag == _W_TBL:
            pieces.append(_docx_table_text(child))
        elif tag not in _W_SKIP and tag != _MC_FALLBACK:
            pieces.append(_docx_text(child))  # runs, hyperlinks, content controls, text boxes
    return "".join(pieces)


def _docx_table_text(table: Any) -> str:
    """Flatten table rows while avoiding duplicate merged-cell text."""
    lines = []
    for row in table.iter(_W_TR):
        if _nearest(row, _W_TBL) is not table:
            continue  # belongs to a nested table, which its own cell already covers
        cells = [" ".join(_docx_text(cell).split()) for cell in row.iter(_W_TC) if _nearest(cell, _W_TR) is row]
        cells = [cell for cell in cells if cell]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines) + "\n" if lines else ""


def _check_docx_archive(data: bytes) -> None:
    """Reject oversized or invalid Word ZIP containers before extraction."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            expanded = sum(info.file_size for info in archive.infolist())
    except (zipfile.BadZipFile, OSError, ValueError):
        raise ExtractionError("Could not open the DOCX - it is not a valid .docx (ZIP) file.") from None
    if expanded > MAX_DOCX_UNCOMPRESSED_BYTES:
        raise ExtractionError(f"The DOCX expands to {_megabytes(expanded)}, which is suspiciously large; refusing to open it.")


def _extract_docx(data: bytes) -> Extraction:
    """Extract Word body and supporting sections with layout-aware text."""
    _check_docx_archive(data)
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises several types for broken packages
        raise ExtractionError(
            f"Could not open the DOCX - the file may be corrupt or not a real Word document ({type(exc).__name__})."
        ) from None
    headers, footers = [], []
    for part in document.part.package.iter_parts():
        name, element = str(part.partname), getattr(part, "element", None)
        bucket = headers if name.startswith("/word/header") else footers if name.startswith("/word/footer") else None
        if bucket is None or element is None:
            continue
        text = _docx_text(element).strip()
        if text and text not in bucket:
            bucket.append(text)
    combined = "\n".join([*headers, _docx_text(document.element.body), *footers])
    lines = [line.rstrip() for line in combined.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return Extraction(text, {}, () if text else ("The document contains no text.",))


_EXTRACTORS: dict[str, Callable[[bytes], Extraction]] = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    **{extension: _extract_text for extension in TEXT_EXTENSIONS},
}


def extract(data: bytes, extension: str) -> Extraction:
    """Extract text from a file's bytes; ``extension`` is the lower-case name suffix, e.g. ".pdf".

    The real format (from the first bytes) wins over the name, with a warning. If that
    fails for a file named like plain text, the file is read as plain text after all.
    """
    if extension not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(f"'{extension or '(no extension)'}' files aren't supported.")
    sniffed = sniff_format(data)
    if sniffed == ".doc":
        if "EncryptedPackage".encode("utf-16-le") in data:
            raise ExtractionError(
                "This Word file is password-protected. Save an unencrypted .docx copy first."
            )
        raise ExtractionError(LEGACY_DOC_MESSAGE)
    if sniffed is None or sniffed == extension:
        return _EXTRACTORS[extension](data)
    try:
        extraction = _EXTRACTORS[sniffed](data)
    except ExtractionError:
        if extension in TEXT_EXTENSIONS:  # e.g. a note that merely starts with "%PDF-1"
            return _EXTRACTORS[extension](data)
        raise
    note = f"The file is named '{extension}' but is actually a {sniffed.lstrip('.').upper()} file; it was read as one."
    return extraction._replace(warnings=(note, *extraction.warnings))
