"""
qa_chain.py
-----------
Grounded question answering for Apple's Q3 2022 Form 10-Q.

Flow: receive a question + retrieved chunks -> build a grounded prompt ->
call Groq -> return a concise, cited answer.

Only Groq is used in the final application (see docs/approach_writeup.md
for why: it reliably outperformed a local 8B model on the numeric/table
reasoning this filing requires, and Groq's free tier makes it a defensible
default with no cost trade-off).
"""

import os
from typing import List

from dotenv import load_dotenv
from groq import Groq

from src.retrieval.retriever import RetrievedChunk

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """
You answer questions about Apple Inc.'s Q3 2022 Form 10-Q.

Use ONLY the retrieved context provided by the application.

Instructions:

- Give a concise, factual answer.
- Use exact values from the retrieved context.
- Do not invent facts.
- Do not use outside knowledge.
- For numerical questions, carefully inspect ALL retrieved tables.
- Prefer a table when it directly contains the requested value.
- Carefully match values to the correct row and column.
- Pay attention to whether the question asks about:
  - three months / quarter
  - nine months
  - 2022
  - 2021
  - percentage
  - dollar amount
  - shares
- For comparison questions, provide all requested values.
- If the filing reports several period-specific values instead of one
  combined value, report those values rather than inventing a combined value.
- Do not perform calculations unless they are directly supported by the
  retrieved context and necessary to answer the question.
- If the answer is not present in the retrieved context, say:
  The information is not available in the retrieved context.
- Do not claim that information is unavailable if it appears anywhere in
  the retrieved passages.
- Do not use Markdown.
- Do not use bullet points.
- Do not use numbered lists.
- Do not use tables.
- Write the answer as one short paragraph.
- End with exactly one citation in this format:

(Source: Page X, Section Name)

Use the actual page number and section name from the retrieved context.

Return ONLY the answer and citation.
"""


def build_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    """Convert retrieved chunks into a clearly delimited context block."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(
            f"\n--- Retrieved Passage {i} ---\n"
            f"Type: {chunk.kind}\n"
            f"Page: {chunk.page}\n"
            f"Section: {chunk.section}\n"
            f"Content:\n{chunk.content}\n"
            f"--- End Passage {i} ---\n"
        )
    return "\n".join(parts)


def answer_with_groq(question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Generate a grounded answer using Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    context = build_context(retrieved_chunks)
    user_prompt = f"\nQuestion:\n{question}\n\nRetrieved context:\n{context}\n"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def answer_question(question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Generate an answer from retrieved context using Groq."""
    return answer_with_groq(question, retrieved_chunks)
