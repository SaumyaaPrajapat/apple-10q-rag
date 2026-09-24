"""
scripts/ingest.py
------------------
One-shot ingestion pipeline: PDF -> parsed elements -> chunks -> embeddings
persisted to local ChromaDB. Run this once before using the Streamlit app.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --pdf data/2022_Q3_AAPL.pdf
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest.parser import parse_pdf
from src.ingest.chunker import build_chunks
from src.ingest.embed_store import EmbeddingStore


def main():
    parser = argparse.ArgumentParser(description="Ingest a 10-Q PDF into the RAG vector store.")
    parser.add_argument("--pdf", default="data/2022_Q3_AAPL.pdf", help="Path to the PDF file.")
    args = parser.parse_args()

    print(f"[1/3] Parsing PDF: {args.pdf}")
    elements = parse_pdf(args.pdf)
    kinds = {}
    for e in elements:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print(f"      Extracted elements: {kinds}")

    print("[2/3] Chunking elements...")
    chunks = build_chunks(elements)
    print(f"      Built {len(chunks)} chunks.")

    print("[3/3] Embedding + indexing into ChromaDB (downloads bge-small on first run)...")
    store = EmbeddingStore()
    store.index_chunks(chunks)
    print(f"      Done. Vector store now has {store.count()} vectors.")
    print("\nIngestion complete. You can now run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
