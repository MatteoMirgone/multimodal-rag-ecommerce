# Multimodal Conversational AI for E-commerce

A vision-language RAG chatbot for product Q&A on the Amazon Product Dataset 2020. The system accepts text queries, image uploads, or both, retrieves relevant products from a CLIP-based vector index, and generates grounded responses using Llama 3.1.

Implementation of the class assignment *"Multimodal Conversational AI for E-commerce: A Vision-Language Approach"* for MSc Applied Data Science, University of Chicago.

## Architecture

Two pipelines that meet at the vector database:

**Offline indexing** (runs once)
1. Load Amazon Product Dataset 2020 (Kaggle, promptcloud)
2. Clean, dedupe, stratified-sample to ~2K products
3. Cache primary product images locally
4. Encode text descriptions and images with CLIP (ViT-B/32)
5. L2-normalize and write to ChromaDB (two collections: `products_text`, `products_image`)

**Online query** (runs per user message)
1. Streamlit UI accepts text and/or image (combined into one chat-input pill)
2. **Hybrid retrieval** — CLIP vector search + keyword-name search merged via Reciprocal Rank Fusion, then a lexical-anchor + image-exact-match override
3. Confidence is scored from top similarity + text-overlap heuristic (HIGH / MEDIUM / LOW)
4. Llama 3.1 generates a grounded response using retrieved product metadata and the confidence tag
5. UI displays the answer with a colored confidence badge plus retrieved product cards (image-first, with cosine similarity pills)


## Project structure

```
GenAI Final Project/
├── README.md                            you are here
├── requirements.txt
├── .env.example                         template for the GROQ_API_KEY (copy to .env)
├── amazon_products.csv                  (in notebooks/) raw Kaggle dataset
├── notebooks/
│   ├── build_index_colab.ipynb          Phases 1–2 combined: clean + cache + CLIP + index (run on Colab T4)
│   └── 03_retrieval_eval.ipynb          Phase 3: Recall@k notebook (interactive variant of scripts/run_eval.py)
├── scripts/
│   ├── rebuild_prepared_data.py         Reconstruct prepared_data parquet from an existing chroma_db
│   ├── cache_images.py                  Download primary product images locally
│   ├── run_eval.py                      Phase 3 as a script — writes eval_results/
│   └── smoke_test.py                    End-to-end pipeline test (no LLM key required)
├── app/
│   ├── rag_chain.py                     CLIP encoding, chroma retrieval, LLM grounding (importable)
│   └── app.py                           Streamlit chat UI
├── chroma_db/                           persistent vector store (built by Phase 2)
├── prepared_data/                       cleaned catalog + cached images (built by Phase 1)
├── eval_results/                        Recall@k CSVs + chart (built by Phase 3)
└── docs/
    └── REPORT.md                        research report
```

## Setup

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS/Linux
# or: .venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

A GPU is helpful for Phase 2 (CLIP encoding) but not required — Apple Silicon MPS works, CPU is slow but viable. The provided `chroma_db/` is already built, so most users will only need GPU/MPS if they re-index.

### 2. Dataset

The CSV is included at `notebooks/amazon_products.csv` (~19 MB). If you re-download from Kaggle:

```
https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020
```

### 3. LLM API key

Llama 3.1 via Groq (free tier, sub-second responses):

1. Sign up at https://console.groq.com
2. Get an API key
3. Either export it, or copy `.env.example` to `.env` and paste it there:

```bash
cp .env.example .env
# edit .env, set GROQ_API_KEY=gsk_your_key
```

The app auto-loads `.env` from the project root.

Alternative: Together AI for Mixtral or other open models — set `LLM_PROVIDER=together` and `TOGETHER_API_KEY`.

## Running the project

The fastest path — the vector index is already built and committed under `chroma_db/`. Everything else is reproducible from there.

### Quick start (uses the prebuilt index)

```bash
# 1. Rebuild prepared_data parquet from chroma_db
python scripts/rebuild_prepared_data.py

# 2. Cache product images locally (~1 minute, 99.8% success rate)
python scripts/cache_images.py

# 3. Smoke test the pipeline (no GROQ key needed)
python scripts/smoke_test.py

# 4. Run the Recall@1/5/10 evaluation (~30 sec on Apple Silicon)
python scripts/run_eval.py

# 5. Launch the chatbot
cd app
streamlit run app.py
```

Open the URL Streamlit prints (typically http://localhost:8501).

### Full rebuild (re-index from scratch)

Run `notebooks/build_index_colab.ipynb` on Google Colab with a T4 GPU runtime. It does Phase 1 (data cleanup + image caching) and Phase 2 (CLIP encoding + ChromaDB indexing) end-to-end. Download the generated `chroma_db.zip` and `prepared_data/` and drop them into the project root.

For the evaluation step, you can run either the notebook (`notebooks/03_retrieval_eval.ipynb`) or the script (`scripts/run_eval.py`) — they produce the same artifacts.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TARGET_SIZE` (notebook) | 2000 | Catalog size after sampling |
| `MAX_IMAGE_SIDE` (notebook + cache_images) | 512 | Image resize maximum side |
| `MODEL_NAME` (notebook + eval) | `openai/clip-vit-base-patch32` | CLIP variant |
| `BATCH_SIZE` (notebook) | 64 | Text encoding batch size |
| `N_EVAL` (run_eval.py) | 500 | Eval set size |
| `GROQ_API_KEY` (env / .env) | — | Required for the LLM call |
| `LLM_PROVIDER` (env) | `groq` | `groq` or `together` |
| `LLM_MODEL` (env) | `llama-3.1-8b-instant` | LLM model name |

The Streamlit sidebar additionally exposes `top_k` (1–10) and the prompt strategy (zero-shot, few-shot, multi-shot) at query time.

## Design decisions

**Why CLIP ViT-B/32, not ViT-L/14?** B/32 is ~3× faster and quality is sufficient at this catalog size — partial-text Recall@5 = 0.998 (see `docs/REPORT.md` §5). ViT-L/14 is a drop-in upgrade if you want to ablate.

**Why hybrid retrieval (CLIP + keyword + RRF)?** CLIP-only retrieval is weak on short generic queries like *"skates"* or *"roller blades"* because cosine similarity favors verbose product descriptions over precise lexical matches. Adding a keyword lane (in-memory token index over product names, with super-linear scoring for multi-keyword matches) and merging via Reciprocal Rank Fusion fixes this entire class of failures with no model retrain. A **lexical anchor** (all-query-keywords-in-name → position 0) and **image-exact-match override** (image cosine ≥ 0.95 → position 0) handle the long-tail cases where the right product is retrieved but at the wrong rank.

**Why ChromaDB, not Vertex AI Vector Search?** The project spec mentions Vertex as the example vector DB. ChromaDB runs locally with zero infrastructure, persists to a 24 MB SQLite file, and uses the same cosine-similarity HNSW algorithm as Vertex. Swapping is ~50 lines if required. Note: changing the vector store does **not** improve retrieval accuracy — embeddings are identical regardless of where they live. The hybrid layer is what moves the numbers.

**Why Llama 3.1 8B Instant via Groq, not 70B?** Grounded RAG (where the answer is in the retrieved context) doesn't need a frontier model. 8B is fast and accurate for this use case. 70B is available on Groq as `llama-3.1-70b-versatile` if you want to swap.

**Why separate text and image collections?** Both embeddings live in the same 512-dim CLIP space, so technically one collection works. Keeping them separate makes ablation cleaner and matches the typical "query each modality independently, then merge" pattern. The merge logic — RRF, lexical anchor, image-exact-match override — lives in `MultimodalRAG.retrieve()`.

**Zero/few/multi-shot prompting.** All three strategies are implemented in `app/rag_chain.py::MultimodalRAG.build_prompt`. Exemplars use abstract placeholder product names (`{PRODUCT_A}`, `{PRODUCT_B}`) inside the system message — not as separate chat turns — so the LLM cannot confuse them with real conversation history. Switch between strategies live in the sidebar segmented control.

**Why a confidence tag in the prompt instead of just relying on cosine?** Naked cosine similarity is unintuitive for LLMs; a categorical HIGH/MEDIUM/LOW label produces much more consistent response styles. Text-overlap is folded into the confidence calculation (3+ query words appearing in the top product's name escalates to HIGH even at moderate cosine), which catches the case where CLIP scores a real lexical match conservatively.

## Evaluation summary

From `eval_results/recall_summary.csv`:

| Query type | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| text → text (self) | 1.000 | 1.000 | 1.000 |
| image → image (self) | 1.000 | 1.000 | 1.000 |
| partial-text → text | 0.984 | 0.998 | 0.998 |
| image → text | 0.568 | 0.816 | 0.894 |
| text → image | 0.556 | 0.810 | 0.868 |

Full breakdown — per-category, per-query, and discussion — in `docs/REPORT.md`.

## Known limitations

- **77-token cap**: CLIP truncates text at 77 tokens. The `desc_full` variant is mostly truncated in practice; `desc_standard` is the one indexed. CLIP architecture limit, not a bug.
- **Catalog skew**: 60.6% of the indexed catalog is Toys & Games. Out-of-domain queries (electronics, apparel) retrieve the closest toy rather than failing gracefully — the UI's exposed similarity score lets the user judge match quality.
- **Out-of-distribution images**: CLIP works best on catalog-style images. User-uploaded photos with cluttered backgrounds, weird angles, or partial views retrieve worse.
- **No conversation memory across turns**: The LLM is called fresh each turn. Streamlit retains the message log; threading prior turns into the prompt is a few-line extension.
- **No fine-tuning**: CLIP is used zero-shot. Fine-tuning on Amazon product data would likely close part of the ~18-point gap between in-modality and cross-modal Recall@5.

## License & attribution

Class project for MSADS at the University of Chicago. Built on Hugging Face `transformers`, ChromaDB, Streamlit, and the OpenAI CLIP model. Llama 3.1 inference via Groq.
