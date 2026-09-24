"""
tests/test_pipeline.py
-----------------------
Sanity tests for the ingestion pipeline (parsing + chunking), which run
fully offline with no API keys or model downloads required.

Retrieval/generation are integration-tested manually via
scripts/ingest.py + app/streamlit_app.py, since they require downloading
the embedding model and/or a live LLM backend (Groq API key or local
Ollama), which this repo intentionally does not require just to verify
the parsing logic is correct.

Run with:  PYTHONPATH=. pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.ingest.parser import parse_pdf, Element
from src.ingest.chunker import build_chunks, Chunk

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "2022_Q3_AAPL.pdf")


@pytest.fixture(scope="module")
def elements():
    return parse_pdf(PDF_PATH)


@pytest.fixture(scope="module")
def chunks(elements):
    return build_chunks(elements)


def test_pdf_parses_without_error(elements):
    assert len(elements) > 0


def test_extracts_all_three_content_types(elements):
    kinds = {e.kind for e in elements}
    assert "text" in kinds
    assert "table" in kinds
    assert "image" in kinds


def test_extracts_expected_number_of_tables(elements):
    # The filing has ~30 financial tables across statements + notes.
    tables = [e for e in elements if e.kind == "table"]
    assert 20 <= len(tables) <= 40


def test_income_statement_table_has_correct_figures(elements):
    """Regression test: verifies the core financial figures extracted from
    the Condensed Consolidated Statements of Operations match the source
    PDF exactly (net sales, net income for Q3 2022)."""
    income_tables = [
        e for e in elements
        if e.kind == "table" and "Total net sales" in e.content and "Net income" in e.content
    ]
    assert len(income_tables) >= 1
    table = income_tables[0]
    assert "82,959" in table.content  # Q3 2022 total net sales ($M)
    assert "19,442" in table.content  # Q3 2022 net income ($M)


def test_segment_table_has_correct_geography_figures(elements):
    """Regression test for Note 9 segment data (Greater China)."""
    seg_tables = [
        e for e in elements
        if e.kind == "table" and "Greater China" in e.content
    ]
    assert len(seg_tables) >= 1
    assert "14,604" in seg_tables[0].content  # Greater China Q3 2022 net sales ($M)


def test_section_tagging_assigns_known_sections(elements):
    sections = {e.section for e in elements if e.section}
    assert any("Item 1" in s for s in sections)
    assert any("Note 9" in s for s in sections)


def test_tables_are_not_duplicated_in_text_elements(elements):
    """Ensures the table-exclusion filter in parser.py is working -- a
    known table value should not also appear as a raw text element."""
    text_with_82959 = [e for e in elements if e.kind == "text" and "82,959" in e.content]
    assert len(text_with_82959) == 0


def test_chunks_preserve_all_tables_atomically(elements, chunks):
    n_tables_parsed = len([e for e in elements if e.kind == "table"])
    n_table_chunks = len([c for c in chunks if c.kind == "table"])
    assert n_tables_parsed == n_table_chunks  # 1:1, tables are never split


def test_table_chunks_include_source_citation_header(chunks):
    table_chunks = [c for c in chunks if c.kind == "table"]
    for c in table_chunks:
        assert c.content.startswith("[Source:")
        assert f"Page {c.page}" in c.content


def test_text_chunks_respect_max_size(chunks):
    text_chunks = [c for c in chunks if c.kind == "text"]
    # Allow some slack for the "[Source: ...]" header prefix
    oversized = [c for c in text_chunks if len(c.content) > 1200]
    assert len(oversized) == 0


def test_image_element_extracted(elements):
    images = [e for e in elements if e.kind == "image"]
    assert len(images) >= 1
    assert os.path.exists(images[0].content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
