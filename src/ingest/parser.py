"""
parser.py
---------
Parses a 10-Q PDF into a structured, ordered list of "elements":
  - text blocks (narrative paragraphs), tagged with the section header they fall under
  - tables, cleaned and serialized to Markdown
  - images/figures, extracted and saved to disk with a page reference

Design notes (see docs/approach_writeup.pdf for full reasoning):
- We use pdfplumber for text + table extraction because it exposes per-word
  bounding boxes and a dedicated table-detection algorithm (based on ruling
  lines/whitespace), which is more reliable on financial statements than
  naive `page.extract_text()` alone.
- We use PyMuPDF (fitz) only for image extraction, since pdfplumber's image
  support is limited to bounding boxes, not raw image bytes.
- Section headers are detected with a regex over known 10-Q conventions
  ("Item N.", "Note N –", "PART I/II") so every chunk can carry a
  human-readable "section" tag for citation purposes.
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Optional

import pdfplumber
import pymupdf  # PyMuPDF (formerly imported as `fitz`)


# ---- Section header patterns typical of SEC 10-Q filings ----
SECTION_PATTERNS = [
    re.compile(r"^\s*PART\s+[I]{1,3}\b", re.IGNORECASE),
    re.compile(r"^\s*Item\s+\d+[A-Z]?\.\s+.+", re.IGNORECASE),
    re.compile(r"^\s*Note\s+\d+\s*[–-]\s*.+", re.IGNORECASE),
]


@dataclass
class Element:
    """A single unit of extracted content, ready for chunking."""
    kind: str                      # "text" | "table" | "image"
    content: str                   # text, markdown table, or image file path
    page: int                      # 1-indexed page number
    section: Optional[str] = None  # nearest preceding section header
    metadata: dict = field(default_factory=dict)


def _is_section_header(line: str) -> bool:
    return any(p.match(line.strip()) for p in SECTION_PATTERNS)


def _clean_table_cell(cell) -> str:
    """Normalize a single table cell: strip stray '$' tokens, None -> ''."""
    if cell is None:
        return ""
    cell = str(cell).replace("\n", " ").strip()
    return cell


def _table_to_markdown(table: List[List[str]]) -> str:
    """
    Convert pdfplumber's raw table (list of rows) into a clean Markdown table.

    pdfplumber often splits a single logical column (e.g. "$ 63,355") into
    two cells: "$" and "63,355". We merge lone "$" / "(" / ")" tokens into
    the adjacent numeric cell so each column reads naturally, then drop
    fully-empty rows/columns which are common artifacts of merged cells
    in the original PDF layout.
    """
    cleaned_rows = []
    for row in table:
        cells = [_clean_table_cell(c) for c in row]
        merged = []
        i = 0
        while i < len(cells):
            if cells[i] in ("$",) and i + 1 < len(cells):
                merged.append(f"$ {cells[i + 1]}".strip())
                i += 2
            else:
                merged.append(cells[i])
                i += 1
        if any(c != "" for c in merged):
            cleaned_rows.append(merged)

    if not cleaned_rows:
        return ""

    # Drop columns that are empty across every row (common artifact)
    n_cols = max(len(r) for r in cleaned_rows)
    cleaned_rows = [r + [""] * (n_cols - len(r)) for r in cleaned_rows]
    keep_cols = [c for c in range(n_cols) if any(r[c] != "" for r in cleaned_rows)]
    cleaned_rows = [[r[c] for c in keep_cols] for r in cleaned_rows]

    header, *body = cleaned_rows
    md = ["| " + " | ".join(header) + " |"]
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        md.append("| " + " | ".join(row) + " |")
    return "\n".join(md)


def extract_images(pdf_path: str, out_dir: str) -> List[Element]:
    """
    Extract embedded raster images from the PDF using PyMuPDF.

    Each extracted image is saved to disk and represented as an Element
    carrying its file path, page number, and image metadata (dimensions,
    format, and the PDF xref it came from, useful for debugging duplicate
    or malformed images).
    """
    os.makedirs(out_dir, exist_ok=True)
    elements = []

    doc = pymupdf.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                extension = base_image["ext"]

                filename = f"page{page_index + 1}_img{img_index + 1}.{extension}"
                file_path = os.path.join(out_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(base_image["image"])

                elements.append(
                    Element(
                        kind="image",
                        content=file_path,
                        page=page_index + 1,
                        metadata={
                            "width": base_image.get("width"),
                            "height": base_image.get("height"),
                            "extension": extension,
                            "xref": xref,
                        },
                    )
                )
    finally:
        doc.close()

    return elements


def parse_pdf(pdf_path: str, image_out_dir: str = "data/extracted_images") -> List[Element]:
    """
    Main entry point. Returns an ordered list of Elements spanning the
    whole document: text blocks (with section tags) and tables (as markdown).
    Images are extracted separately and appended at the end (order doesn't
    matter for figures since they are retrieved independently by page ref).

    Text and tables are merged and sorted by vertical position (top-to-bottom)
    on each page, so that:
      (a) section headers are attributed to the correct, immediately-following
          table/text (not leaked over from a previous page), and
      (b) a table's preceding caption/column-header lines (e.g. "Three Months
          Ended ... 2022 2021") — which pdfplumber often extracts as text
          just above the table's bounding box rather than as its first row —
          can be captured and prefixed onto the table's markdown for context.
    """
    elements: List[Element] = []
    current_section = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            # Collect text lines OUTSIDE any table bbox, each tagged with its
            # vertical position, so we can interleave them with tables below.
            def not_in_table(obj):
                for (x0, top, x1, bottom) in table_bboxes:
                    if x0 <= obj["x0"] <= x1 and top <= obj["top"] <= bottom:
                        return False
                return True

            filtered_page = page.filter(not_in_table)
            words = filtered_page.extract_words()

            # Group words into lines by their 'top' coordinate (pdfplumber
            # words already come reading-order sorted; we bucket by line).
            lines = []
            current_line, current_top = [], None
            for w in words:
                if current_top is None or abs(w["top"] - current_top) < 3:
                    current_line.append(w)
                    current_top = w["top"] if current_top is None else current_top
                else:
                    lines.append((current_top, " ".join(x["text"] for x in current_line)))
                    current_line, current_top = [w], w["top"]
            if current_line:
                lines.append((current_top, " ".join(x["text"] for x in current_line)))

            # Build a combined, position-sorted stream of ("text", top, str) /
            # ("table", top, Table) items for this page.
            stream = [("text", top, txt) for top, txt in lines]
            stream += [("table", t.bbox[1], t) for t in tables]  # bbox[1] = top
            stream.sort(key=lambda item: item[1])

            pending_caption_lines: List[str] = []  # text immediately preceding a table

            for kind, top, payload in stream:
                if kind == "text":
                    line = payload.strip()
                    if not line:
                        continue
                    if _is_section_header(line):
                        current_section = line
                        pending_caption_lines.clear()
                        continue
                    elements.append(
                        Element(kind="text", content=line, page=page_num, section=current_section)
                    )
                    # Keep the last couple of lines as potential table caption
                    # (covers cases like "Three Months Ended ... 2022 2021"
                    # sitting directly above a table with no explicit header row).
                    pending_caption_lines.append(line)
                    pending_caption_lines[:] = pending_caption_lines[-3:]
                else:
                    raw_table = payload.extract()
                    md_table = _table_to_markdown(raw_table)
                    if md_table:
                        caption = " | ".join(pending_caption_lines)
                        content = f"[Context: {caption}]\n{md_table}" if caption else md_table
                        elements.append(
                            Element(kind="table", content=content, page=page_num, section=current_section)
                        )
                    pending_caption_lines.clear()

    # --- Images (figures/logos) ---
    elements.extend(extract_images(pdf_path, image_out_dir))

    return elements


if __name__ == "__main__":
    els = parse_pdf("data/2022_Q3_AAPL.pdf")
    kinds = {}
    for e in els:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print("Extracted element counts:", kinds)
    print("\nSample table element:\n", next(e for e in els if e.kind == "table").content[:500])
