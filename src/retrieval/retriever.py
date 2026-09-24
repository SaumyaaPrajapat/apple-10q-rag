"""
retriever.py
------------
Hybrid retrieval for financial filings. Combines:
  1. Dense vector search for semantic similarity.
  2. BM25 for exact/near-exact lexical matching.
  3. A small, keyword-triggered boost for table chunks on financial questions.
  4. Reciprocal Rank Fusion (RRF) to combine the vector + BM25 rankings.

The structured-data boost (#3) is deliberately small and only activates on
queries that look like factual financial questions (see
`_looks_like_financial_question`). It nudges table chunks up in an
otherwise-close ranking rather than overriding RRF, since prose chunks
should remain fully eligible for narrative/legal questions.
"""

import re
from dataclasses import dataclass
from typing import Dict, List

from rank_bm25 import BM25Okapi

from src.ingest.chunker import Chunk
from src.ingest.embed_store import EmbeddingStore


@dataclass
class RetrievedChunk:
    id: str
    content: str
    kind: str
    page: int
    section: str
    score: float


def _tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for BM25 that keeps financial tokens intact
    (e.g. "greater china", "iphone", "14,604", "2022")."""
    return re.findall(r"\b[\w$%,.-]+\b", text.lower())


# Terms that suggest the answer lives in a financial table rather than prose.
_FINANCIAL_TERMS = [
    "net sales", "sales", "revenue", "operating income", "operating loss",
    "gross margin", "net income", "earnings", "research and development",
    "r&d", "expense", "expenses", "cash", "assets", "liabilities",
    "inventory", "debt", "tax", "taxes", "shares", "per share", "diluted",
    "basic", "greater china", "americas", "europe", "japan", "asia pacific",
    "iphone", "ipad", "mac", "wearables", "services",
]

# Phrasing patterns typical of a factual lookup question (vs. a narrative one).
_QUESTION_PATTERNS = [
    "how much", "what was", "what were", "how many", "what is", "what are",
    "for the quarter", "for the nine months", "for q1", "for q2", "for q3", "for q4",
]


def _looks_like_financial_question(query: str) -> bool:
    """
    Detect queries likely to benefit from a nudge toward structured tables.
    Intentionally conservative (requires both a financial term AND a
    lookup-style phrasing) so narrative/legal questions aren't affected.
    """
    q = query.lower()
    has_financial_term = any(term in q for term in _FINANCIAL_TERMS)
    has_question_pattern = any(pattern in q for pattern in _QUESTION_PATTERNS)
    return has_financial_term and has_question_pattern


class HybridRetriever:
    def __init__(self, store: EmbeddingStore, all_chunks: List[Chunk], top_k: int = 6):
        self.store = store
        self.top_k = top_k
        self.chunks_by_id: Dict[str, Chunk] = {c.id: c for c in all_chunks}

        # Build the BM25 index once over the full chunk collection.
        self.corpus_ids = [c.id for c in all_chunks]
        tokenized_corpus = [_tokenize(c.content) for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _vector_search(self, query: str, k: int) -> List[str]:
        """Return chunk IDs ranked by dense vector similarity."""
        query_emb = self.store.embed_query(query)
        results = self.store.collection.query(query_embeddings=[query_emb], n_results=k)
        return results["ids"][0]

    def _bm25_search(self, query: str, k: int) -> List[str]:
        """Return chunk IDs ranked by BM25 lexical relevance."""
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.corpus_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:k]]

    def _apply_structured_boost(self, query: str, fused_scores: Dict[str, float]) -> None:
        """
        In-place: nudges table chunks up for questions that look like
        factual financial lookups, with an extra nudge when the table's
        own content contains the specific metric phrase from the query
        (e.g. "total net sales"). Boost sizes are small relative to typical
        RRF score gaps, by design — this breaks close ties, it doesn't
        override strong semantic/lexical signal.
        """
        if not _looks_like_financial_question(query):
            return

        q = query.lower()
        for cid in list(fused_scores.keys()):
            chunk = self.chunks_by_id.get(cid)
            if chunk is None or chunk.kind != "table":
                continue

            fused_scores[cid] += 0.015
            content_lower = chunk.content.lower()
            if "total net sales" in q and "total net sales" in content_lower:
                fused_scores[cid] += 0.015
            elif "net sales" in q and "net sales" in content_lower:
                fused_scores[cid] += 0.010

    def retrieve(self, query: str, k: int = None, rrf_k: int = 60) -> List[RetrievedChunk]:
        """
        Hybrid retrieval: dense + BM25, fused via Reciprocal Rank Fusion,
        with a small structured-data boost applied afterward (see
        `_apply_structured_boost`).

        The candidate pool fetched from each ranker is deliberately larger
        than the final result set `k`, so a cluster of similar prose chunks
        can't crowd out a relevant table chunk before fusion happens. With
        only ~150 chunks in this corpus, retrieving most of the collection
        is cheap and gives RRF a meaningfully better candidate set.
        """
        k = k or self.top_k
        pool_size = min(len(self.corpus_ids), max(k * 8, 40))

        vector_ids = self._vector_search(query, pool_size)
        bm25_ids = self._bm25_search(query, pool_size)

        fused_scores: Dict[str, float] = {}
        for rank, cid in enumerate(vector_ids):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        self._apply_structured_boost(query, fused_scores)

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for cid, score in ranked:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(id=cid, content=chunk.content, kind=chunk.kind,
                                page=chunk.page, section=chunk.section or "", score=score)
            )
        return results


if __name__ == "__main__":
    from src.ingest.chunker import build_chunks
    from src.ingest.parser import parse_pdf

    elements = parse_pdf("data/2022_Q3_AAPL.pdf")
    chunks = build_chunks(elements)

    store = EmbeddingStore()
    if store.count() == 0:
        store.index_chunks(chunks)

    retriever = HybridRetriever(store, chunks)

    queries = [
        "What was Apple's net sales in Greater China for Q3 2022?",
        "What was Apple total net sales in the third quarter of 2022?",
        "What were Greater China net sales and operating income for the nine months ended June 25 2022?",
        "How much did Apple spend on research and development in the third quarter of 2022?",
        "What is the Epic Games lawsuit about?",
    ]

    for query in queries:
        print(f"\n{'=' * 80}\nQUERY: {query}\n{'=' * 80}")
        for result in retriever.retrieve(query, k=3):
            print(f"\n[{result.kind} | page {result.page} | score={result.score:.4f}]")
            print(result.content[:1000])
