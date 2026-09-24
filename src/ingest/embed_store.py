"""
embed_store.py
---------------
Embeds Chunks with a local, free sentence-transformer model and persists
them into a local ChromaDB collection.

Why BAAI/bge-small-en-v1.5:
- Free, runs fully locally (no API key, no per-call cost) -- important for
  a reproducible take-home submission that a grader can run offline.
- Small (~130MB, 384-dim) so it embeds this whole filing in seconds on CPU.
- Punches well above its size on retrieval benchmarks (MTEB) relative to
  other sub-100M-parameter embedding models, which is why it's a common
  default in production RAG stacks despite being free/open.
See docs/approach_writeup.pdf for the full model comparison and reasoning.
"""

import os
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingest.chunker import Chunk

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "aapl_10q_q3_2022"


class EmbeddingStore:
    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 32) -> None:
        """Embed and upsert all chunks into the Chroma collection."""
        # bge models recommend prefixing passages (not needed for queries)
        # with nothing special for retrieval passages; the "query:" prefix
        # convention applies to the *query* side, handled in retriever.py.
        texts = [c.content for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [{"kind": c.kind, "page": c.page, "section": c.section or ""} for c in chunks]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            embeddings = self.model.encode(batch_texts, normalize_embeddings=True).tolist()
            self.collection.upsert(
                ids=batch_ids, embeddings=embeddings, documents=batch_texts, metadatas=batch_meta
            )

    def embed_query(self, query: str):
        # bge-small was trained with an instruction prefix for queries,
        # which measurably improves retrieval quality vs. an unprefixed query.
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.model.encode([prefixed], normalize_embeddings=True)[0].tolist()

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    from src.ingest.parser import parse_pdf
    from src.ingest.chunker import build_chunks

    print("Parsing PDF...")
    elements = parse_pdf("data/2022_Q3_AAPL.pdf")
    print(f"Parsed {len(elements)} elements.")

    print("Chunking...")
    chunks = build_chunks(elements)
    print(f"Built {len(chunks)} chunks.")

    print("Embedding + indexing (this downloads the model on first run)...")
    store = EmbeddingStore()
    store.index_chunks(chunks)
    print(f"Indexed. Collection now has {store.count()} vectors.")
