# Apple Q3 2022 10-Q — RAG Information Retrieval System

A Retrieval-Augmented Generation (RAG) system that answers questions about
Apple Inc.'s **Form 10-Q for the third quarter of fiscal 2022**. It handles
**narrative text** and **financial tables**, extracts **embedded figures**, and
returns short answers that are grounded in the filing and carry a page-level
citation.

- **Source document:** `data/2022_Q3_AAPL.pdf` (28 pages; from the Docugami KG-RAG dataset)
- **Methodology write-up:** `docs/Approach_and_Methodology.pdf`
- **Interface:** Streamlit web app

---

## What it does

Ask a question in plain English and the app will:

1. Retrieve the most relevant passages and tables from the filing (hybrid search).
2. Send them to an LLM with a strict "answer only from this context" prompt.
3. Show a concise answer with a citation, e.g. `(Source: Page 18, Item 2. Management's Discussion ...)`.
4. Let you inspect every retrieved chunk — tables are rendered as real tables, and extracted images are displayed.

Example questions:

| Type         | Question                                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Table        | What was Apple's total net sales in Q3 2022 and how does it compare to Q3 2021?                                         |
| Table        | What were Greater China net sales and operating income for the nine months ended June 25, 2022?                         |
| Table        | What was the gross margin percentage for Products vs Services in Q3 2022?                                               |
| Text         | What is the Epic Games lawsuit about?                                                                                   |
| Text + table | How many shares did Apple repurchase in Q3 2022, and at what average price?                                             |
| Abstention   | What was Apple's revenue from a product category not reported in the 10-Q? → _"not available in the retrieved context"_ |

---

## Models used

| Role                        | Model                              | Where it runs                   |
| --------------------------- | ---------------------------------- | ------------------------------- |
| **Answer generation (LLM)** | `openai/gpt-oss-120b`              | Groq API (needs a free API key) |
| **Embeddings**              | `BAAI/bge-small-en-v1.5` (384-dim) | Locally, on CPU — no API key    |
| **Keyword retrieval**       | BM25 (`rank-bm25`, BM25Okapi)      | Locally                         |

Generation runs at `temperature=0` for reproducible answers.

---

## Design

```
                        ┌──────────── OFFLINE: python scripts/ingest.py ────────────┐
 2022_Q3_AAPL.pdf ─▶ parser.py ─▶ chunker.py ─▶ embed_store.py ─▶ ChromaDB (data/chroma_db)
                     text/tables/   164 chunks    bge-small          persistent vectors
                     images         (800/100)     embeddings
                        └────────────────────────────────────────────────────────────┘

                        ┌──────────── ONLINE: streamlit run app/streamlit_app.py ───┐
 Question ─▶ retriever.py ──────────────────────────────▶ qa_chain.py ─▶ Answer + citation
             ├─ dense search (ChromaDB)  ┐                 Groq gpt-oss-120b
             ├─ BM25 keyword search      ├─ RRF fusion     grounded prompt
             └─ small table boost        ┘ → top-k chunks
                        └────────────────────────────────────────────────────────────┘
```

**Key design decisions**

- **Table-aware parsing.** `pdfplumber` finds tables and converts them to clean Markdown. Tables are kept as single, unsplit chunks, and the few text lines just above a table (often its column headers, such as "Three Months Ended … 2022 2021") are attached to it.
- **Section- and page-aware chunking.** Text is grouped by 10-Q section (`Item N.`, `Note N –`) and by page, then split into ~800-character chunks with 100 characters of overlap. Grouping by page preserves page metadata so generated answers can cite the source page.
- **Hybrid retrieval.** Dense vectors capture meaning; BM25 captures exact tokens such as "Greater China" or "14,604". The two rankings are merged with **Reciprocal Rank Fusion (RRF)**. A small boost nudges table chunks up for factual financial lookups.
- **Grounded generation.** The prompt forbids outside knowledge, requires care with periods (3-month vs 9-month, 2022 vs 2021) and units, requires an abstention message when the context lacks the answer, and requires exactly one citation.
- **Local embeddings + hosted LLM.** Retrieval needs no paid API and is reproducible offline; only the final answer step calls Groq.

---

## Setup

**Requirements:** Python 3.10 or newer (3.10–3.12 recommended), ~1 GB free disk space (PyTorch + embedding model), internet for the first run.

### 1. Clone and install

```bash
git clone <your-repo-url>
cd rag-pipeline

# create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows (cmd / PowerShell)

pip install -r requirements.txt
```

### 2. Add your Groq API key

Get a free key at <https://console.groq.com>, then:

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

Open `.env` and set:

```
GROQ_API_KEY=your_groq_api_key_here
```

> `.env` is git-ignored. Never commit it. If a key is ever exposed, revoke it in the Groq console and create a new one.

### 3. Build the index (run once)

```bash
python scripts/ingest.py
```

This parses the PDF (646 text elements, 31 tables, 1 image), builds 164 chunks, downloads the embedding model on first run (~130 MB), and writes 164 vectors to `data/chroma_db/`. You may see harmless ChromaDB "Failed to send telemetry event" warnings.

### 4. Launch the app

```bash
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>), click a sample question or type your own, and press **Ask**. Use the sidebar slider to change how many chunks are retrieved (default 6).

---

## Tests and evaluation

```bash
# Offline tests for parsing + chunking (no API key needed)
python -m pytest tests/ -v

# Retrieval-only metrics (no LLM calls)
python evaluation/evaluate.py

# End-to-end evaluation (retrieval + Groq generation); needs GROQ_API_KEY
python evaluation/evaluate_generation.py
```

### Results on the 25-question evaluation set

`evaluation/questions.json` has 25 questions: 24 answerable and 1 deliberately unanswerable (to test abstention).

| Metric                                      | Result        |
| ------------------------------------------- | ------------- |
| Retrieval Hit@1                             | 16/24 (0.667) |
| Retrieval Hit@3                             | 21/24 (0.875) |
| Retrieval Hit@6                             | 23/24 (0.958) |
| Retrieval MRR                               | 0.771         |
| Average numerical match                     | 0.798         |
| Citation points to a retrieved page         | 24/24 (1.000) |
| Correct abstention on unanswerable question | 1/1 (1.000)   |

Notes on reading these numbers:

- Retrieval "hit" means a chunk from an expected **page** was retrieved; it does not check that the exact chunk contained the answer.
- Numerical match is a strict automatic string check on the numbers in the reference answer. It can under-score correct answers (e.g. the share-repurchase question, where the filing reports three sub-period average prices and no single overall average).
- The set is small and is a sanity check, not a benchmark.
- Groq's free tier can return a rate-limit error (HTTP 429); the evaluation script waits and retries.

---

## Project structure

```
rag-pipeline/
├── app/
│   └── streamlit_app.py          # Web UI: ask, answer, inspect retrieved chunks
├── data/
│   └── 2022_Q3_AAPL.pdf          # Source filing
│   (chroma_db/, extracted_images/ are generated and git-ignored)
├── docs/
│   └── Approach_and_Methodology.pdf      # Methodology write-up
├── evaluation/
│   ├── questions.json            # 25 labeled questions
│   ├── evaluate.py               # Retrieval metrics (Hit@k, MRR)
│   ├── evaluate_generation.py    # End-to-end metrics
│   ├── results.csv               # Output of evaluate.py
│   └── generation_results.csv    # Output of evaluate_generation.py
├── scripts/
│   └── ingest.py                 # PDF -> chunks -> vector store (run once)
├── src/
│   ├── ingest/
│   │   ├── parser.py             # PDF -> Elements (text / table / image)
│   │   ├── chunker.py            # Elements -> Chunks
│   │   └── embed_store.py        # Embeddings + ChromaDB
│   ├── retrieval/
│   │   └── retriever.py          # Hybrid BM25 + dense search with RRF
│   └── generation/
│       └── qa_chain.py           # Prompt + Groq call
├── tests/
│   └── test_pipeline.py          # Offline parsing/chunking tests
├── .env.example                  # Template for GROQ_API_KEY
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Limitations

- **Figures are extracted, not "understood".** Embedded images are pulled out of the PDF and shown in the UI, but the image bytes are **not** passed to the LLM, so the system does not do visual question answering. (This particular filing contains only one image, the Apple logo.) Text and table content drives all answers.
- **Single-document system.** Section patterns, the table-boost keyword list, and the prompt are tuned for SEC 10-Q filings and this Apple filing.
- **Tables from `pdfplumber` can be imperfect** when the PDF layout is unusual; the UI shows the raw extracted text alongside the rendered table so this can be checked.
- **No cross-encoder reranker** and no numeric/aggregation layer; questions that need arithmetic across several tables are the weakest area.
- **Hosted LLM.** Answer generation needs internet access and a Groq key, and is subject to free-tier rate limits.

## Possible improvements

- Pass extracted images to a vision-capable model for true figure Q&A.
- Add a cross-encoder reranker on the top ~20 fused candidates.
- Move tunable values (chunk size, top-k, model names, boost weights) into a single config file.
- Enlarge the evaluation set and add answer-level (LLM- or human-judged) scoring.
- Generalize to multiple filings with per-document metadata filters.
