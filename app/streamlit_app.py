"""
app/streamlit_app.py
---------------------
Streamlit UI for the Apple Q3 2022 10-Q RAG system.

Features:
- Groq (gpt-oss-120b) generation with grounded, cited answers.
- Hybrid BM25 + vector retrieval.
- Generic Markdown-table -> dataframe rendering for any retrieved table
  (works for any table shape, not hardcoded to specific tables in this
  filing -- see docs/approach_writeup.md for why that matters).
- Extracted figure/image display.
- An inspector panel showing exactly which chunks were retrieved, with
  both the rendered and raw form of each, for verifying retrieval quality
  separately from generation quality.

Run:
    streamlit run app/streamlit_app.py
"""

import html
import os
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.generation.qa_chain import answer_question
from src.ingest.chunker import build_chunks
from src.ingest.embed_store import EmbeddingStore
from src.ingest.parser import parse_pdf
from src.retrieval.retriever import HybridRetriever

load_dotenv()

st.set_page_config(page_title="Apple 10-Q RAG Assistant", page_icon="📊", layout="wide")


# --- Table parsing helpers ---

def make_unique_column_names(names):
    """Disambiguate duplicate column names (pandas requires unique columns)."""
    unique_names = []
    for index, name in enumerate(names):
        name = str(name).strip() or f"Column {index + 1}"
        original_name, counter = name, 2
        while name in unique_names:
            name = f"{original_name} ({counter})"
            counter += 1
        unique_names.append(name)
    return unique_names


def clean_extracted_table_row(cells):
    """Drop empty cells and merge a lone '%' fragment into the preceding
    number (e.g. ["3", "%"] -> ["3%"]), a common PDF-table artifact."""
    cells = [str(c).strip() for c in cells if str(c).strip()]
    cleaned, i = [], 0
    while i < len(cells):
        if i + 1 < len(cells) and cells[i + 1] == "%":
            cleaned.append(cells[i] + "%")
            i += 2
        else:
            cleaned.append(cells[i])
            i += 1
    return cleaned


def extract_table_rows(content):
    """Parse Markdown-style table rows out of a table chunk's text,
    skipping the '| --- | --- |' separator row."""
    normalized = content.replace("\\|", "|").replace("&#124;", "|")
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip().startswith("|")]

    rows = []
    for line in lines:
        line = line.strip().strip("|")
        if not line:
            continue
        cells = [c.strip() for c in line.split("|")]
        is_separator = all(not c or set(c.replace(":", "").strip()) <= {"-"} for c in cells)
        if is_separator:
            continue
        rows.append(clean_extracted_table_row(cells))
    return rows


def render_table_chunk(content):
    """
    Render a retrieved table chunk as a readable dataframe.

    Deliberately generic: normalizes every row to the same column count
    (padding short rows, truncating long ones) and uses the first row as
    the header. This works for any table shape found in any filing, rather
    than hardcoding column layouts for specific tables in this one
    document -- a hardcoded per-table renderer wouldn't generalize past
    this exact PDF, which defeats the point of a reusable RAG pipeline.
    Falls back to a raw code block if the content can't be parsed as a
    table at all.
    """
    rows = extract_table_rows(content)
    if len(rows) < 2:
        st.code(content, language="text")
        return

    max_columns = max(len(row) for row in rows)
    normalized_rows = []
    for row in rows:
        row = list(row)
        if len(row) < max_columns:
            row += [""] * (max_columns - len(row))
        else:
            row = row[:max_columns]
        normalized_rows.append(row)

    header = make_unique_column_names(normalized_rows[0])
    data = normalized_rows[1:]

    try:
        df = pd.DataFrame(data, columns=header)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.code(content, language="text")


# --- Image rendering ---

def render_image_chunk(result):
    """
    Display an extracted image, if the referenced file still exists.

    The image path is parsed back out of the chunk's descriptive text
    (chunker.py formats image chunks as "... File: <path>. Dimensions:
    ..."). This couples the two files by string format rather than a
    shared schema -- acceptable for a project this size, but a structured
    metadata field (rather than embedding the path in prose) would be a
    cleaner contract if this were extended further.
    """
    image_path = None
    marker = "File: "
    if marker in result.content:
        image_path = result.content.split(marker, 1)[1].split(". Dimensions:", 1)[0].strip()

    if not image_path:
        st.warning("Image path was not available.")
        return

    image_path = os.path.normpath(image_path)
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Extracted image from Page {result.page}", width=300)
        st.caption("Extracted directly from the PDF for visual inspection.")
    else:
        st.warning(f"Extracted image file was not found: {image_path}")


# --- Retrieved-chunk inspector ---

def render_retrieved_chunk(index: int, result) -> None:
    """Display one retrieved chunk: rendered form (table/image/text) plus
    the raw extracted content, so retrieval quality can be inspected
    independently of how well the LLM used it."""
    st.markdown(f"**#{index} [{result.kind.upper()}] Page {result.page} — "
                f"{result.section}** (fusion score: {result.score:.4f})")

    if result.kind == "table":
        st.markdown("**Rendered table**")
        render_table_chunk(result.content)
        with st.container(border=True):
            st.markdown("**Raw extracted table**")
            st.code(result.content, language="text")

    elif result.kind == "image":
        st.markdown("**Extracted figure/image**")
        render_image_chunk(result)
        with st.container(border=True):
            st.markdown("**Image metadata**")
            st.code(result.content, language="text")

    else:
        st.code(result.content, language="text")


# --- Pipeline loading ---

@st.cache_resource(show_spinner="Loading vector store and building retriever...")
def load_pipeline(pdf_path):
    """Parse, chunk, embed (if not already indexed), and build the
    retriever once per Streamlit session."""
    elements = parse_pdf(pdf_path)
    chunks = build_chunks(elements)

    store = EmbeddingStore()
    if store.count() == 0:
        store.index_chunks(chunks)

    return HybridRetriever(store, chunks)


# --- UI ---

st.title("📊 Apple Inc. Q3 2022 10-Q — RAG Q&A Assistant")
st.caption(
    "Ask questions about text, tables, or figures in Apple's Q3 2022 Form 10-Q. "
    "Answers are grounded in retrieved passages."
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Generation model: Groq gpt-oss-120b")
    top_k = st.slider("Number of chunks to retrieve", min_value=2, max_value=10, value=6)

    st.markdown("---")
    st.subheader("Try a sample question")
    sample_questions = [
        "What was Apple's total net sales in Q3 2022 and how does it compare to Q3 2021?",
        "What was Apple's net sales and operating income in Greater China for the nine months ended June 25, 2022?",
        "How much did Apple spend on research and development in Q3 2022?",
        "What is the Epic Games lawsuit about?",
        "How many shares of common stock did Apple repurchase during Q3 2022, and at what average price?",
        "What was Apple's gross margin percentage for Products vs Services in Q3 2022?",
        "What were Apple's Services net sales in Q3 2022 and Q3 2021?",
    ]
    for sample in sample_questions:
        if st.button(sample, use_container_width=True):
            st.session_state["question"] = sample

PDF_PATH = os.path.join(PROJECT_ROOT, "data", "2022_Q3_AAPL.pdf")
retriever = load_pipeline(PDF_PATH)

question = st.text_input(
    "Ask a question about the filing:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What were Apple's Services net sales in Q3 2022?",
)

if st.button("Ask", type="primary") and question.strip():
    clean_question = question.strip()

    with st.spinner("Retrieving relevant passages..."):
        retrieved = retriever.retrieve(clean_question, k=top_k)

    with st.spinner("Generating grounded answer..."):
        try:
            answer = answer_question(clean_question, retrieved)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            answer = None

    if answer:
        st.markdown("### 💬 Answer")
        safe_answer = html.escape(answer.strip())
        # Compact rendering: collapse newlines so blank lines in the
        # generated answer don't create large vertical gaps in the UI.
        st.markdown(
            f"""
            <div style="white-space: normal; overflow-wrap: anywhere; word-break: break-word;
                        width: 100%; line-height: 1.55; margin-top: 0; margin-bottom: 0.5rem;">
            {safe_answer.replace(chr(10), ' ')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(f"🔍 Retrieved context ({len(retrieved)} chunks) — click to inspect"):
        for index, result in enumerate(retrieved, start=1):
            render_retrieved_chunk(index, result)
