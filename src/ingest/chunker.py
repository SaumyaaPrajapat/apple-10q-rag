"""
chunker.py
----------
Converts parsed PDF Elements into retrieval-ready Chunks.

Design:
- TABLE elements remain one chunk each so row/column relationships are
  preserved.
- IMAGE elements become metadata chunks referencing the extracted image.
- TEXT elements are grouped by BOTH section and page so page-level
  citations remain accurate (a chunk spanning multiple pages would
  otherwise get assigned the wrong page number in the final citation).
- Text is split into approximately 800-character chunks with 100-character
  overlap.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.ingest.parser import Element

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


@dataclass
class Chunk:
    """Retrieval-ready chunk. kind is one of: "text" | "table" | "image"."""
    id: str
    kind: str
    content: str
    page: int
    section: Optional[str]


def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into ~`size`-character chunks, preferring to break at a
    sentence boundary and falling back to a hard cut when none is found
    within the window. A small overlap is kept between chunks so context
    isn't lost right at a boundary.
    """
    if len(text) <= size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            boundary = text.rfind(". ", start, end)
            boundary = boundary + 1 if boundary > start else end
        else:
            boundary = len(text)

        piece = text[start:boundary].strip()
        if piece:
            chunks.append(piece)

        next_start = boundary - overlap
        start = next_start if next_start > start else boundary

    return chunks


def _table_header(element: Element) -> str:
    return f"[Source: {element.section or 'Unknown section'}, Page {element.page}]\n"


def build_chunks(elements: List[Element]) -> List[Chunk]:
    """
    Convert parsed Elements into retrieval-ready chunks.

    Tables and images each become one chunk per element. Text elements are
    grouped by (section, page) — not just section — before being split to
    size, so that a chunk never straddles a page boundary and citations
    stay accurate.
    """
    chunks: List[Chunk] = []
    chunk_id = 0

    # --- Tables: one chunk each, never split ---
    for element in elements:
        if element.kind != "table":
            continue
        content = _table_header(element) + element.content
        chunks.append(Chunk(id=f"chunk_{chunk_id}", kind="table", content=content,
                             page=element.page, section=element.section))
        chunk_id += 1

    # --- Images: one lightweight descriptive chunk each ---
    for element in elements:
        if element.kind != "image":
            continue
        width = element.metadata.get("width")
        height = element.metadata.get("height")
        content = (
            f"[Figure/Image extracted from Page {element.page}. File: {element.content}. "
            f"Dimensions: {width}x{height}. This image was extracted from the filing. "
            f"It should be displayed for visual inspection rather than treated as a "
            f"text-based financial table.]"
        )
        chunks.append(Chunk(id=f"chunk_{chunk_id}", kind="image", content=content,
                             page=element.page, section=element.section))
        chunk_id += 1

    # --- Text: group by (section, page), then split to size ---
    text_elements = [e for e in elements if e.kind == "text"]
    buffer: List[str] = []
    buffer_section, buffer_page = None, None

    def flush_text_buffer():
        nonlocal chunk_id, buffer, buffer_section, buffer_page
        merged = " ".join(buffer).strip()
        buffer = []
        if not merged:
            return
        for piece in _split_text(merged):
            header = f"[Source: {buffer_section or 'Unknown section'}, Page {buffer_page}]\n"
            chunks.append(Chunk(id=f"chunk_{chunk_id}", kind="text", content=header + piece,
                                 page=buffer_page or 0, section=buffer_section))
            chunk_id += 1

    for element in text_elements:
        if element.section != buffer_section or element.page != buffer_page:
            flush_text_buffer()
            buffer_section, buffer_page = element.section, element.page
        buffer.append(element.content)
    flush_text_buffer()

    return chunks


if __name__ == "__main__":
    from src.ingest.parser import parse_pdf

    elements = parse_pdf("data/2022_Q3_AAPL.pdf")
    chunks = build_chunks(elements)

    kinds = {}
    for c in chunks:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1

    print("Chunk counts:", kinds)
    print("Total chunks:", len(chunks))
    print("\nSample text chunk:\n")
    first_text = next((c for c in chunks if c.kind == "text"), None)
    if first_text:
        print(first_text.content)
