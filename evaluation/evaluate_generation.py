"""
evaluation/evaluate_generation.py
---------------------------------
End-to-end evaluation of the Apple Q3 2022 10-Q RAG system: question ->
hybrid retrieval -> top-K context -> Groq LLM -> generated answer.

For each question, records:
  - retrieval success (did the top-K include an expected page?)
  - the generated answer
  - numerical answer matching (do the expected figures appear in the answer?)
  - citation presence and whether the cited page matches retrieved evidence
  - abstention behavior on deliberately unanswerable questions

Usage:
    python evaluation/evaluate_generation.py

Requirements:
    GROQ_API_KEY must be available in .env.
"""

import csv
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "2022_Q3_AAPL.pdf")
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "questions.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "generation_results.csv")
TOP_K = 6

sys.path.insert(0, PROJECT_ROOT)

from src.generation.qa_chain import answer_question
from src.ingest.chunker import build_chunks
from src.ingest.embed_store import EmbeddingStore
from src.ingest.parser import parse_pdf
from src.retrieval.retriever import HybridRetriever


# --- Question loading ---

def load_questions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Numerical matching ---

def extract_expected_numbers(text: str) -> List[str]:
    """Extract monetary values, percentages, and plain numbers from the
    expected answer, deduplicated while preserving order."""
    patterns = [
        r"\$[\d,]+(?:\.\d+)?",
        r"\b\d[\d,]*(?:\.\d+)?%",
        r"\b\d[\d,]*(?:\.\d+)?\b",
    ]
    values = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    seen, result = set(), []
    for value in values:
        normalized = value.lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(value)
    return result


def number_is_present(expected_number: str, answer: str) -> bool:
    """Check whether an expected numeric value appears in the generated
    answer, tolerant of common formatting differences ($, commas, spacing)."""
    expected = expected_number.strip().lower()
    answer_lower = answer.lower()

    if "%" in expected:
        expected_value = expected.replace("%", "").replace(",", "").replace(" ", "")
        pattern = r"(?<!\d)" + re.escape(expected_value) + r"\s*%"
        return bool(re.search(pattern, answer_lower))

    expected_numeric = re.sub(r"[^0-9.]", "", expected)
    if not expected_numeric:
        return False

    answer_numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", answer_lower)
    return any(re.sub(r"[^0-9.]", "", c) == expected_numeric for c in answer_numbers)


def numerical_match_score(expected_answer: str, generated_answer: str) -> float:
    """Fraction of expected numerical values found in the generated answer
    (1.0 if the expected answer has no numbers to check, e.g. an
    abstention-style expected answer)."""
    expected_numbers = extract_expected_numbers(expected_answer)
    if not expected_numbers:
        return 1.0
    matches = sum(number_is_present(n, generated_answer) for n in expected_numbers)
    return matches / len(expected_numbers)


# --- Retrieval checks ---

def retrieved_pages(retrieved) -> List[int]:
    """Unique retrieved pages, in ranking order."""
    pages, seen = [], set()
    for chunk in retrieved:
        if chunk.page not in seen:
            seen.add(chunk.page)
            pages.append(chunk.page)
    return pages


def page_retrieved(retrieved, relevant_pages: List[int]) -> bool:
    """Whether at least one expected page appears among the retrieved chunks."""
    if not relevant_pages:
        return False
    return any(chunk.page in relevant_pages for chunk in retrieved)


# --- Citation checks ---

def citation_from_answer(answer: str) -> str:
    """Extract the '(Source: Page X, Section Name)' citation, if present."""
    match = re.search(r"\(Source:.*?\)", answer, flags=re.IGNORECASE | re.DOTALL)
    return match.group(0) if match else ""


def citation_page(answer: str) -> Optional[int]:
    """Extract the cited page number, if any."""
    citation = citation_from_answer(answer)
    if not citation:
        return None
    match = re.search(r"Page\s+(\d+)", citation, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def citation_matches_retrieval(answer: str, retrieved) -> bool:
    """Whether the page cited by the answer is among the retrieved pages
    (a proxy for citation faithfulness -- catches the model citing a page
    it never actually saw)."""
    page = citation_page(answer)
    if page is None:
        return False
    return any(chunk.page == page for chunk in retrieved)


# --- Abstention check ---

_ABSTENTION_PHRASES = [
    "not available", "not provided", "not reported", "not disclosed",
    "cannot be determined", "does not provide", "does not report",
    "not present in the retrieved context", "not present in the filing",
    "information is unavailable", "information is not available",
]


def answer_appears_to_abstain(answer: str) -> bool:
    """Lightweight keyword signal for whether the model declined to answer,
    rather than a full semantic check."""
    text = answer.lower()
    return any(phrase in text for phrase in _ABSTENTION_PHRASES)


# --- Generation with retry on rate limiting ---

def generate_with_retry(question: str, retrieved, max_attempts: int = 3) -> str:
    """Call the Groq-backed answer_question, retrying with backoff on a
    429 (rate limit) response. Returns an '[GENERATION ERROR] ...' string
    on non-retryable failures so evaluation can continue past one bad
    question instead of aborting the whole run."""
    try:
        for attempt in range(max_attempts):
            try:
                return answer_question(question, retrieved)
            except Exception as exc:
                if "429" not in str(exc):
                    raise
                wait_seconds = 8 * (attempt + 1)
                print(f"      Groq rate limit reached. Waiting {wait_seconds}s...")
                time.sleep(wait_seconds)
        raise RuntimeError("Groq generation failed after retries.")
    except Exception as exc:
        print(f"      Generation error: {exc}")
        return f"[GENERATION ERROR] {exc}"


# --- Per-question result builders ---

def build_unanswerable_result(question_id: str, question: str, expected_answer: str,
                               answer: str, pages: List[int], citation: str,
                               cited_page: Optional[int], citation_valid: bool) -> Dict:
    abstained = answer_appears_to_abstain(answer)
    return {
        "id": question_id, "question": question, "expected_answer": expected_answer,
        "generated_answer": answer, "answerable": False, "relevant_pages": "",
        "retrieved_pages": ",".join(str(p) for p in pages),
        "retrieval_hit": "", "numerical_match": "", "citation": citation,
        "cited_page": cited_page if cited_page is not None else "",
        "citation_matches_retrieval": citation_valid,
        "abstained": abstained,
        "status": "ABSTAINED" if abstained else "DID_NOT_ABSTAIN",
    }


def build_answerable_result(question_id: str, question: str, expected_answer: str,
                             answer: str, relevant_pages: List[int], pages: List[int],
                             retrieval_hit: bool, numerical_score: float, citation: str,
                             cited_page: Optional[int], citation_valid: bool) -> Dict:
    return {
        "id": question_id, "question": question, "expected_answer": expected_answer,
        "generated_answer": answer, "answerable": True,
        "relevant_pages": ",".join(str(p) for p in relevant_pages),
        "retrieved_pages": ",".join(str(p) for p in pages),
        "retrieval_hit": retrieval_hit, "numerical_match": round(numerical_score, 3),
        "citation": citation, "cited_page": cited_page if cited_page is not None else "",
        "citation_matches_retrieval": citation_valid, "abstained": "",
        "status": "OK" if retrieval_hit else "RETRIEVAL_MISS",
    }


def evaluate_one_question(retriever: HybridRetriever, question_data: Dict) -> Dict:
    """Run retrieval + generation for one question and score the result."""
    question_id = question_data["id"]
    question = question_data["question"]
    expected_answer = question_data["expected_answer"]
    relevant_pages = question_data.get("relevant_pages", [])

    retrieved = retriever.retrieve(question, k=TOP_K)
    pages = retrieved_pages(retrieved)
    retrieval_hit = page_retrieved(retrieved, relevant_pages)

    answer = generate_with_retry(question, retrieved)

    numerical_score = numerical_match_score(expected_answer, answer)
    citation = citation_from_answer(answer)
    cited_page = citation_page(answer)
    citation_valid = citation_matches_retrieval(answer, retrieved)

    if not relevant_pages:
        result = build_unanswerable_result(question_id, question, expected_answer, answer,
                                            pages, citation, cited_page, citation_valid)
        print(f"      Retrieved pages: {pages}")
        print("      Abstention: " + ("YES" if result["abstained"] else "NO"))
    else:
        result = build_answerable_result(question_id, question, expected_answer, answer,
                                          relevant_pages, pages, retrieval_hit,
                                          numerical_score, citation, cited_page, citation_valid)
        print(f"      Retrieved pages: {pages}")
        print("      Retrieval hit: " + ("YES" if retrieval_hit else "NO"))
        print(f"      Numerical match: {numerical_score:.3f}")
        print("      Citation: " + (citation if citation else "NONE"))
        print("      Citation matches retrieved: " + ("YES" if citation_valid else "NO"))

    print(f"      Answer: {answer}\n")
    return result


def save_results(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["id", "question", "expected_answer", "generated_answer", "answerable",
                  "relevant_pages", "retrieved_pages", "retrieval_hit", "numerical_match",
                  "citation", "cited_page", "citation_matches_retrieval", "abstained", "status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: List[Dict]) -> None:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    retrieval_rate = (sum(bool(r["retrieval_hit"]) for r in answerable) / len(answerable)
                       if answerable else 0.0)
    avg_numerical = (sum(r["numerical_match"] for r in answerable) / len(answerable)
                      if answerable else 0.0)
    citation_rate = (sum(bool(r["citation_matches_retrieval"]) for r in answerable) / len(answerable)
                      if answerable else 0.0)
    abstention_rate = (sum(bool(r["abstained"]) for r in unanswerable) / len(unanswerable)
                        if unanswerable else 0.0)

    print("=" * 70)
    print("END-TO-END EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions        : {len(results)}")
    print(f"Answerable questions   : {len(answerable)}")
    print(f"Unanswerable questions : {len(unanswerable)}")
    print(f"\nRetrieval Hit@6        : {retrieval_rate:.3f}")
    print(f"Avg numerical match    : {avg_numerical:.3f}")
    print(f"Citation match rate    : {citation_rate:.3f}")
    print(f"Abstention rate        : {abstention_rate:.3f}")
    print(f"\nResults saved to: {RESULTS_PATH}")


def main():
    print("=" * 70)
    print("Apple Q3 2022 10-Q End-to-End RAG Evaluation")
    print("=" * 70)

    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} evaluation questions.")

    print("\n[1/4] Loading PDF and chunks...")
    elements = parse_pdf(PDF_PATH)
    chunks = build_chunks(elements)
    print(f"      Loaded {len(chunks)} chunks.")

    print("[2/4] Loading embedding store...")
    store = EmbeddingStore()
    print(f"      Vector store contains {store.count()} vectors.")

    retriever = HybridRetriever(store=store, all_chunks=chunks, top_k=TOP_K)

    print("[3/4] Checking Groq configuration...")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY was not found. Make sure it is present in your .env file.")
    print("      GROQ_API_KEY found.")

    print("[4/4] Running end-to-end evaluation...\n")
    results = []
    for index, question_data in enumerate(questions, start=1):
        print(f"[{index:02d}/{len(questions):02d}] {question_data['id']}: {question_data['question']}")
        results.append(evaluate_one_question(retriever, question_data))

    save_results(results, RESULTS_PATH)
    print_summary(results)


if __name__ == "__main__":
    main()
