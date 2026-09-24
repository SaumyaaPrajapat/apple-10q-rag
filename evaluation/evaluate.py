"""
evaluation/evaluate.py
----------------------
Evaluates retrieval quality for the Apple Q3 2022 10-Q RAG system.

Metrics: Hit@1, Hit@3, Hit@6, Mean Reciprocal Rank (MRR).

Questions with an empty `relevant_pages` list are treated as unanswerable
(deliberately, to test abstention) and are excluded from retrieval metrics.

Usage:
    python evaluation/evaluate.py
"""

import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "2022_Q3_AAPL.pdf")
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "questions.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "results.csv")
TOP_K = 6

# Make project imports work when running `python evaluation/evaluate.py`.
sys.path.insert(0, PROJECT_ROOT)

from src.ingest.chunker import build_chunks
from src.ingest.embed_store import EmbeddingStore
from src.ingest.parser import parse_pdf
from src.retrieval.retriever import HybridRetriever


# --- Utility functions ---

def load_questions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_answerable(question: Dict) -> bool:
    """A question is answerable for retrieval scoring if it has at least
    one expected relevant page."""
    return bool(question.get("relevant_pages"))


def hit_at_k(retrieved_pages: List[int], relevant_pages: List[int], k: int) -> int:
    """1 if any relevant page appears in the top-k retrieved pages, else 0."""
    return int(any(page in relevant_pages for page in retrieved_pages[:k]))


def first_relevant_rank(retrieved_pages: List[int], relevant_pages: List[int]) -> Optional[int]:
    """1-based rank of the first relevant result, or None if not retrieved."""
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return rank
    return None


def reciprocal_rank(retrieved_pages: List[int], relevant_pages: List[int]) -> float:
    rank = first_relevant_rank(retrieved_pages, relevant_pages)
    return 1.0 / rank if rank else 0.0


def preview_text(text: str, max_length: int = 280) -> str:
    """Compact one-line preview for terminal output."""
    text = " ".join(text.split())
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


# --- Per-question evaluation ---

def evaluate_question(retriever: HybridRetriever, question: Dict) -> Tuple[Dict, List]:
    """Run retrieval for one evaluation question; returns (result_dict, retrieved_chunks)."""
    question_id = question["id"]
    query = question["question"]
    relevant_pages = question.get("relevant_pages", [])

    retrieved = retriever.retrieve(query, k=TOP_K)
    retrieved_pages = [c.page for c in retrieved]

    if not relevant_pages:
        # Deliberately unanswerable question -- excluded from retrieval
        # metrics since there's no expected page to score against.
        return {
            "id": question_id, "question": query, "answerable": False,
            "expected_pages": "", "hit_at_1": "", "hit_at_3": "", "hit_at_6": "",
            "mrr": "", "first_relevant_rank": "", "status": "UNANSWERABLE",
        }, retrieved

    h1 = hit_at_k(retrieved_pages, relevant_pages, 1)
    h3 = hit_at_k(retrieved_pages, relevant_pages, 3)
    h6 = hit_at_k(retrieved_pages, relevant_pages, 6)
    mrr = reciprocal_rank(retrieved_pages, relevant_pages)
    rank = first_relevant_rank(retrieved_pages, relevant_pages)

    result = {
        "id": question_id, "question": query, "answerable": True,
        "expected_pages": ",".join(str(p) for p in relevant_pages),
        "hit_at_1": h1, "hit_at_3": h3, "hit_at_6": h6,
        "mrr": round(mrr, 4), "first_relevant_rank": rank if rank is not None else "",
        "status": "HIT" if h6 else "MISS",
    }
    return result, retrieved


def print_failure_details(question: Dict, retrieved: List) -> None:
    """Print detailed retrieval info for a failed answerable question."""
    print(f"\n  {'-' * 66}\n  FAILED QUESTION\n  {'-' * 66}")
    print(f"  ID: {question['id']}")
    print(f"  Question: {question['question']}")
    print(f"  Expected pages: {question.get('relevant_pages', [])}")
    print("\n  Retrieved results:")
    for rank, chunk in enumerate(retrieved, start=1):
        print(f"    #{rank} | Page {chunk.page} | {chunk.kind} | Score {chunk.score:.4f}")
        print(f"         {preview_text(chunk.content)}")
    print(f"  {'-' * 66}\n")


def save_results(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["id", "question", "answerable", "expected_pages", "hit_at_1",
                  "hit_at_3", "hit_at_6", "mrr", "first_relevant_rank", "status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# --- Main ---

def main():
    print("=" * 70)
    print("Apple Q3 2022 10-Q RAG Evaluation")
    print("=" * 70)

    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} evaluation questions.")

    print("\n[1/3] Loading PDF and chunks...")
    elements = parse_pdf(PDF_PATH)
    chunks = build_chunks(elements)
    print(f"      Loaded {len(chunks)} chunks.")

    print("[2/3] Loading embedding store...")
    store = EmbeddingStore()
    print(f"      Vector store contains {store.count()} vectors.")

    retriever = HybridRetriever(store=store, all_chunks=chunks, top_k=TOP_K)

    print("\n[3/3] Running evaluation...\n")
    results = []
    for question in questions:
        result, retrieved = evaluate_question(retriever, question)
        results.append(result)

        if not result["answerable"]:
            print(f" {result['id']} | Unanswerable | Excluded from retrieval metrics")
            continue

        print(f" {result['id']} | Hit@1={result['hit_at_1']} | Hit@3={result['hit_at_3']} | "
              f"Hit@6={result['hit_at_6']} | MRR={result['mrr']:.3f} | {result['status']}")

        if result["status"] == "MISS":
            print_failure_details(question, retrieved)

    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    num_answerable = len(answerable)

    if num_answerable > 0:
        hit1 = sum(r["hit_at_1"] for r in answerable) / num_answerable
        hit3 = sum(r["hit_at_3"] for r in answerable) / num_answerable
        hit6 = sum(r["hit_at_6"] for r in answerable) / num_answerable
        mrr = sum(r["mrr"] for r in answerable) / num_answerable
    else:
        hit1 = hit3 = hit6 = mrr = 0.0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions       : {len(questions)}")
    print(f"Answerable questions  : {num_answerable}")
    print(f"Unanswerable questions: {len(unanswerable)}")
    print(f"\nHit@1                 : {hit1:.3f}")
    print(f"Hit@3                 : {hit3:.3f}")
    print(f"Hit@6                 : {hit6:.3f}")
    print(f"MRR                   : {mrr:.3f}")

    save_results(results, RESULTS_PATH)
    print(f"\nDetailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
