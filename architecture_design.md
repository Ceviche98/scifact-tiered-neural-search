# SciFact Search Engine — Architecture Design

**Queen Mary University of London — Information Retrieval (ECS736P) — Coursework II**

Authors: Oscar Gabriel Moreno Veliz, Min Thant Kyaw, Mohamed Amiin Mohamed, Andres Julian Herrera Medina

---

## 1. Project Overview

A scientific claim retrieval system over the **BEIR SciFact** corpus (5,183 abstracts). The system exposes two retrieval tiers with very different speed/accuracy trade-offs, a real-time analytics page for benchmarking both tiers, and a Streamlit-based web UI.

The corpus is fixed; no user documents are indexed at runtime. All heavy pre-computation (FAISS indexes, corpus embeddings, HyDE hypothetical documents) is done offline and loaded at startup.

---

## 2. Directory Structure

```
scifact_search_engine/
│
├── app_ui.py                   # Main Streamlit page — search interface
├── config.py                   # Central config: paths, hyperparameters, example queries
├── architecture_design.md      # This document
│
├── pages/
│   └── Stats.py                # Streamlit multi-page: analytics / benchmarking
│
├── search/
│   ├── __init__.py
│   ├── embedder.py             # Shared embed() function (both tiers)
│   ├── hyde.py                 # HyDE generation (Qwen) + GOLFer filter
│   ├── loaders.py              # Cached data/asset loaders
│   ├── models.py               # Model load/unload helpers
│   └── pipelines.py            # one_stage_search(), two_stage_search()
│
├── user_interface/
│   ├── __init__.py             # neumorphic_switch() Streamlit custom component
│   ├── info_icon.png           # Icon embedded into the sidebar Info button
│   └── frontend/
│       └── index.html          # Self-contained HTML/JS toggle component
│
├── data_fast_tier/
│   └── data/
│       └── mrl_index/
│           ├── index.faiss     # 256-dim MRL FAISS index (Stella 400M embeddings)
│           └── id_map.json     # int → corpus doc_id mapping
│
├── data_pro_tier/
│   ├── mrl_index/
│   │   ├── index.faiss         # 256-dim MRL FAISS index (Stella 1.5B embeddings)
│   │   └── id_map.json
│   ├── mrl_cache.pkl           # Full 1536-dim embeddings for Stage-2 rerank
│   └── hyde_cache.json         # Pre-computed HyDE docs keyed by query_id
│
├── scifact_data/
│   ├── corpus.jsonl            # 5,183 document abstracts (BEIR format)
│   ├── queries.jsonl           # Test queries
│   └── qrels/
│       └── test.tsv            # Relevance judgements (query-id, corpus-id, score)
│
└── models/
    └── hub/                    # HuggingFace offline model cache (TRANSFORMERS_OFFLINE=1)
        ├── models--NovaSearch--stella_en_400M_v5/
        ├── models--NovaSearch--stella_en_1.5B_v5/
        └── models--Qwen--Qwen2.5-1.5B-Instruct/
```

---

## 3. Tech Stack

| Layer | Library / Version |
|---|---|
| UI framework | Streamlit |
| Deep learning | PyTorch (CPU or CUDA, fp16 on GPU) |
| Transformers | HuggingFace `transformers` |
| Vector search | FAISS (`faiss-cpu` or `faiss-gpu`) |
| Generation (HyDE) | Qwen 2.5 1.5B Instruct (CausalLM) |
| Analytics charts | Plotly (dark theme) |
| Data wrangling | Pandas, NumPy |
| Runtime | Python 3.11, fully offline (`TRANSFORMERS_OFFLINE=1`) |

---

## 4. Models

### 4.1 Stella 400M (`NovaSearch/stella_en_400M_v5`)
- **Role**: Fast-tier bi-encoder for query and document embedding.
- **Embedding dim**: 1536 (full), truncated to **256** (MRL) for FAISS lookup.
- **Precision**: fp16 on GPU, fp32 on CPU.
- **Special handling**: `use_memory_efficient_attention=False`, `unpad_inputs=False` to avoid custom CUDA kernels.
- **Query prefix**: `"Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: <text>"` — applied automatically in `embed()`.
- **Pooling**: Mean pooling over token embeddings, followed by L2 normalisation.

### 4.2 Stella 1.5B (`NovaSearch/stella_en_1.5B_v5`)
- **Role**: Pro-tier bi-encoder — higher capacity for better ranking quality.
- **Embedding dim**: 1536 (full), truncated to **256** (MRL) for Stage-1 FAISS, full 1536 for Stage-2 rerank.
- **Architecture**: Based on Qwen2, loaded with `torch_dtype=torch.float16`.
- **Runtime patches** applied at load time (in `models.py`):
  - Disables Flash Attention 2 (`is_flash_attn_2_available → False`).
  - Patches `DynamicCache.get_usable_length` to prevent AttributeError.
  - Patches `Qwen2Config.__getattribute__` to supply a default `rope_theta=1_000_000.0` when missing.
- **Pooling**: Same mean-pool + L2 norm as 400M.

### 4.3 Qwen 2.5 1.5B Instruct (`Qwen/Qwen2.5-1.5B-Instruct`)
- **Role**: HyDE document generator — produces a fake scientific abstract from a query.
- **Usage**: Only loaded for live HyDE (novel queries not in cache). Loaded on demand, immediately deleted after generation to free RAM before Stella 1.5B loads.
- **Generation config**: `max_new_tokens=200`, `temperature=0.3`, `do_sample=True`.
- **Prompt template**: chat template via `apply_chat_template`; system role = biomedical researcher.

### Memory management
Only **one model** lives in `st.session_state` at a time. `_unload_current_model()` runs `del` + `gc.collect()` + `torch.cuda.empty_cache()` before any new model loads. This is critical on machines without abundant VRAM/RAM.

---

## 5. Central Configuration (`config.py`)

```
BASE_DIR       = project root (Path)
MRL_DIM        = 256          # Truncated dimension used for FAISS lookup
STAGE1_K       = 50           # Candidates retrieved in Stage-1 (Pro tier)
TOP_K          = 10           # Final results returned to user
APP_VERSION    = "V3.29"
EXAMPLE_QUERIES = [10 pre-defined claims]
```

Environment variables set at import time (before torch/transformers load):
- `TRANSFORMERS_OFFLINE=1` — prevents any network calls.
- `HF_HUB_CACHE` → `models/hub/` — all models served locally.
- `KMP_DUPLICATE_LIB_OK=TRUE` — suppresses OpenMP conflicts on Windows.
- `TOKENIZERS_PARALLELISM=false` — avoids deadlock warnings with Streamlit threads.

---

## 6. Retrieval Architecture

### 6.1 Fast Tier — 1-Stage MRL Search

```
User query
    │
    ▼
embed(query, is_query=True)          ← Stella 400M, mean-pool, L2-norm → [1, 1536]
    │
    ▼
q_trunc = query_embed[:, :256]       ← MRL truncation to 256-dim
L2-normalise q_trunc
    │
    ▼
FAISS IndexFlatIP.search(q_trunc, k=10)   ← Inner product = cosine (normalised)
    │
    ▼
id_map[int_id] → doc_id              ← Map integer FAISS indices → corpus IDs
    │
    ▼
Return [(doc_id, score), …]  TOP_K=10
```

**Latency**: Typically < 100 ms after model is loaded (model load itself is the slow step).

---

### 6.2 Pro Tier — HyDE + GOLFer + 2-Stage MRL/Full Rerank

```
User query
    │
    ├─ [in HyDE cache?] ──YES──► load hyde_cache[qid]
    │                                     │
    └─ [novel query]   ──NO───► Qwen 2.5 1.5B generates hypothetical abstract (200 tokens)
                                          │
                                          ▼
                               GOLFer filter (sentence-level overlap)
                               Keeps only sentences sharing ≥2 non-stopword tokens with query
                               → filtered_text
                                          │
                          ┌───────────────┴──────────────────┐
                          ▼                                   ▼
              embed(query, is_query=True)       embed(filtered_text, is_query=False)
              q_embed [1, 1536]                 h_embed [1, 1536]
                          │                                   │
                          └─────────── Interpolate ───────────┘
                                    combined = 0.7 * q_embed + 0.3 * h_embed
                                    q_embed = L2_norm(combined)
                                          │
                                          ▼
                         ┌─── Stage 1: FAISS MRL Lookup ───┐
                         │  q_trunc = q_embed[:, :256]      │
                         │  FAISS.search(q_trunc, k=50)     │
                         │  → 50 candidate doc indices      │
                         └──────────────┬───────────────────┘
                                        │
                                        ▼
                         ┌─── Stage 2: Full-dim Rerank ────┐
                         │  doc_vecs = full_embeds[indices] │  ← 50 × 1536 fp16 tensors
                         │  scores = q_embed @ doc_vecs.T   │  ← dot product (already normed)
                         │  top-k = topk(scores, k=10)      │
                         └──────────────┬───────────────────┘
                                        │
                                        ▼
                              Return [(doc_id, score), …]  TOP_K=10
```

**Why 2-stage?** Stage-1 uses cheap 256-dim vectors for fast approximate retrieval. Stage-2 re-scores those 50 candidates using the full 1536-dim embeddings for precision — no model inference needed at Stage 2, just a matrix multiply.

**HyDE interpolation weight**: `α = 0.7` (query) + `0.3` (hypothetical doc). Fixed; not configurable from UI.

---

## 7. Module Breakdown

### `search/embedder.py`
Single function: `embed(texts, is_query)`.  
Reads model/tokenizer/device from `st.session_state`. Applies query prefix if `is_query=True`. Tokenises with `max_length=512`, runs forward pass with `torch.no_grad()`, mean-pools, L2-normalises, returns CPU tensor. Works for both Stella variants.

### `search/models.py`
- `_get_device()` — returns CUDA device if available, else CPU.
- `_unload_current_model()` — deletes model+tokenizer from session state, calls GC.
- `_ensure_400M_model()` — idempotent load of Stella 400M (skips if already loaded).
- `_ensure_1_5B_model()` — idempotent load of Stella 1.5B with runtime patches (see §4.2).

### `search/loaders.py`
All functions use Streamlit's caching decorators so assets are only loaded once per server session:
- `load_corpus()` → `@st.cache_data` — parses `corpus.jsonl` into `{doc_id: {title, abstract, metadata}}`.
- `load_queries_map()` → `@st.cache_data` — builds `{query_text: query_id}` lookup.
- `load_hyde_cache()` → `@st.cache_data` — loads `hyde_cache.json` (`{query_id: hyde_text}`).
- `load_fast_assets()` → `@st.cache_resource` — loads FAISS index + id_map for 400M tier.
- `load_pro_assets()` → `@st.cache_resource` — loads FAISS index + id_map + full 1536-dim embedding matrix (from `mrl_cache.pkl`).

### `search/hyde.py`
- `golfer_filter(query, hyde_text, min_overlap=2)` — sentence-level relevance filter. Splits the hypothetical abstract into sentences; keeps those with ≥2 non-stopword token overlaps with the query. Falls back to raw query if nothing passes.
- `generate_live_hyde_doc(query)` — loads Qwen, builds chat prompt, generates, decodes, aggressively unloads Qwen.

### `search/pipelines.py`
- `one_stage_search(query_embed, searcher, id_map)` — truncates to 256-dim, FAISS search, returns top-10.
- `two_stage_search(query_embed, searcher, id_map, full_embeds)` — Stage-1 FAISS (k=50), Stage-2 full-dim dot-product rerank, returns top-10.

---

## 8. User Interface

### 8.1 Search Page (`app_ui.py`)
- **Layout**: Wide layout. Sidebar for status/controls. Main area: example query pills → centered search card → results list → document panel.
- **Search card**: Custom CSS fuses a `st.text_input` and a `st.form_submit_button` into a single unified dark bar with red glow-on-focus.
- **Results panel**: Scrollable list (750px height when doc panel is open). Clicking "📄 Read" on a result opens a side panel with the full abstract. A DOM-breaker trick (`click_counter % 2`) forces Streamlit to reset the scroll position on each new document open.
- **Sidebar**: Index status (SVG neon check/cross icons), loaded model name, unload button, Info dialog (author credits), version badge.

### 8.2 Analytics Page (`pages/Stats.py`)
- **Purpose**: Full comparative evaluation of both tiers across all BEIR SciFact test queries.
- **Trigger**: "⚡ Fast vs Pro ▶" button in the sidebar.
- **Execution flow** (driven by `st.session_state.eval_phase`):
  1. `idle` → show placeholder.
  2. `fast_running` → run all N queries through Fast tier, append per-query metrics, update charts every N/40 queries.
  3. `pro_running` → run all N queries through Pro tier (HyDE cache only, no live Qwen generation), update charts.
  4. `complete` → show summary table with ✦ winner per metric, delta percentages, and a narrative insight.
- **Metrics computed per query**: nDCG@10, Recall@10, MRR@10, MAP@10, Precision@10, Latency (ms).
- **Charts**: 4 Plotly dark-theme charts (nDCG rolling average + per-query, latency bar + rolling average) — 2 per tier side-by-side, updated live.

### 8.3 Custom Toggle Component (`user_interface/`)
A Streamlit custom component built as a self-contained `index.html` (no npm/build step). Renders a neumorphic FAST ↔ PRO toggle using inline CSS and vanilla JS. Communicates state back to Python via `postMessage` (Streamlit component bridge). `neumorphic_switch(key, default)` wraps the component and returns a boolean.

### 8.4 Sidebar page renaming
Both pages use a `components.html()` `MutationObserver` injection to rename sidebar nav links from the filesystem names (`app ui`, `Stats`) to display names (`Search Engine`, `Analytics`) without renaming any files.

---

## 9. Data Assets

| Asset | Location | Description |
|---|---|---|
| Corpus | `scifact_data/corpus.jsonl` | 5,183 BEIR SciFact abstracts |
| Queries | `scifact_data/queries.jsonl` | Test queries with IDs |
| QRels | `scifact_data/qrels/test.tsv` | Ground truth relevance judgements (scores 0/1/2) |
| Fast FAISS index | `data_fast_tier/data/mrl_index/index.faiss` | 256-dim Flat IP index, Stella 400M embeddings |
| Fast ID map | `data_fast_tier/data/mrl_index/id_map.json` | `{int_index: doc_id}` |
| Pro FAISS index | `data_pro_tier/mrl_index/index.faiss` | 256-dim Flat IP index, Stella 1.5B embeddings |
| Pro ID map | `data_pro_tier/mrl_index/id_map.json` | `{int_index: doc_id}` |
| Pro full embeds | `data_pro_tier/mrl_cache.pkl` | Full 1536-dim Stella 1.5B embeddings, all 5,183 docs |
| HyDE cache | `data_pro_tier/hyde_cache.json` | `{query_id: hypothetical_abstract_text}` |

---

## 10. Caching Strategy

```
st.cache_data     ─── corpus, queries map, HyDE cache
                      (serialisable Python objects, safe to pickle across reruns)

st.cache_resource ─── FAISS indexes, full embedding matrix
                      (non-serialisable C++/Tensor objects, shared across sessions)

st.session_state  ─── loaded model, tokenizer, device, current results,
                      selected doc ID, click counter, pro_toggle state
```

Models are **not** cached via Streamlit decorators — they are stored in `session_state` and managed manually so they can be unloaded on demand.

---

## 11. Logging

Terminal logging is implemented via Python's standard `logging` module configured in `app_ui.py`:

```
Format:  HH:MM:SS  LEVEL     message
Logger:  "scifact"
Level:   INFO
```

Logged events per search:
- **Fast**: model load, index load, embedding, search, result count, timing summary.
- **Pro**: HyDE cache hit/miss, hypothetical abstract text, GOLFer filtered text, model load, asset load, embedding + interpolation, 2-stage search, result count, timing summary.

No logs appear in the Streamlit UI — all output goes to the terminal only.

---

## 12. Key Design Decisions

| Decision | Rationale |
|---|---|
| MRL (Matryoshka Representation Learning) truncation to 256-dim for FAISS | Reduces index size and search latency while retaining good recall for Stage-1 shortlisting |
| 2-stage retrieval (MRL coarse → full-dim rerank) | Avoids storing full 1536-dim FAISS index while still scoring final candidates at full precision |
| HyDE cache for known queries | Qwen generation takes several seconds; caching eliminates this cost for all test queries |
| GOLFer sentence filter | Prevents HyDE hallucinations from polluting the embedding — keeps only sentences with lexical overlap with the original query |
| Single model in RAM at a time | Stella 1.5B and Qwen are both large; loading both simultaneously would exceed typical RAM budgets |
| `TRANSFORMERS_OFFLINE=1` | Ensures reproducibility and prevents unintended network calls during evaluation |
| Streamlit `@st.cache_resource` for FAISS + embeddings | These objects are expensive to deserialise and should survive page reruns and multi-user sessions |
| Custom neumorphic toggle (no build step) | Pure HTML/CSS/JS component avoids Node.js toolchain while giving a polished custom widget |
