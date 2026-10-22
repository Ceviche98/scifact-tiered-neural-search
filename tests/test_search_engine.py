"""
Unit & Integration Test Suite for SciFact Tiered Neural Search Engine.

Tests:
1. GOLFer lexical grounding verification (`search.hyde.golfer_filter`)
2. 1-Stage Matryoshka 256d FAISS search (`search.pipelines.one_stage_search`)
3. 2-Stage Matryoshka 256d -> 1536d search & exact matrix reranking (`search.pipelines.two_stage_search`)
4. BEIR SciFact corpus, query map, and pre-computed HyDE + GOLFer cache loaders (`search.loaders`)
5. Pre-built FAISS index integrity (`FAST_INDEX` & `PRO_INDEX`)
6. BEIR IR evaluation metrics (nDCG@10, Recall@10, MRR@10, MAP@10, Precision@10)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FAST_IDMAP, FAST_INDEX, HYDE_CACHE, MRL_DIM, PRO_IDMAP, PRO_INDEX, TOP_K
from search.hyde import golfer_filter
from search.loaders import load_corpus, load_fast_assets, load_hyde_cache, load_queries_map
from search.pipelines import one_stage_search, two_stage_search


# ── 1. GOLFer Verification Tests ─────────────────────────────────────────────

def test_golfer_filter_keeps_grounded_and_drops_hallucinated_sentences():
    claim = "0-dimensional biomaterials show inductive properties."
    raw_hyde = (
        "Recent studies on 0-dimensional biomaterials demonstrate strong inductive properties in tissue regeneration. "
        "Quantum entanglement in superconducting qubits stabilizes cryogenic coherence. "
        "These inductive biomaterials promote osteogenic differentiation without exogenous growth factors."
    )
    filtered = golfer_filter(claim, raw_hyde, min_overlap=2)
    assert "0-dimensional biomaterials" in filtered
    assert "inductive biomaterials" in filtered
    assert "Quantum entanglement" not in filtered


def test_golfer_filter_fallback_when_all_sentences_ungrounded():
    claim = "Aspirin inhibits platelet cyclooxygenase-1 activity."
    ungrounded = "Solar neutrino oscillations were measured inside the Cherenkov detector."
    filtered = golfer_filter(claim, ungrounded, min_overlap=2)
    assert filtered == claim


# ── 2. One-Stage 256d MRL Search Tests ───────────────────────────────────────

def test_one_stage_search_retrieves_exact_nearest_neighbor():
    torch.manual_seed(42)
    n_docs, full_dim = 60, 1024
    doc_embs = F.normalize(torch.randn(n_docs, full_dim), p=2, dim=1)
    doc_256 = F.normalize(doc_embs[:, :MRL_DIM], p=2, dim=1).numpy().astype(np.float32)

    index = faiss.IndexFlatIP(MRL_DIM)
    index.add(doc_256)
    id_map = {str(i): f"doc_{i}" for i in range(n_docs)}

    target_idx = 17
    query_embed = doc_embs[target_idx : target_idx + 1]

    results = one_stage_search(query_embed, index, id_map)

    assert len(results) == TOP_K
    assert results[0][0] == f"doc_{target_idx}"
    assert np.isclose(results[0][1], 1.0, atol=1e-4)


# ── 3. Two-Stage 256d -> 1536d Reranking Tests ───────────────────────────────

def test_two_stage_search_reranks_candidates_with_full_1536d_vectors():
    torch.manual_seed(7)
    n_docs, full_dim = 80, 1536
    full_embeds = F.normalize(torch.randn(n_docs, full_dim), p=2, dim=1)

    # Make doc_5 slightly higher in the first 256 dims, while doc_12 is an exact match across all 1536 dims
    query_embed = full_embeds[12:13].clone()
    full_embeds[5, :MRL_DIM] = query_embed[0, :MRL_DIM]
    full_embeds[5, MRL_DIM:] = -query_embed[0, MRL_DIM:]
    full_embeds[5] = F.normalize(full_embeds[5], p=2, dim=0)

    doc_256 = F.normalize(full_embeds[:, :MRL_DIM], p=2, dim=1).numpy().astype(np.float32)
    index = faiss.IndexFlatIP(MRL_DIM)
    index.add(doc_256)
    id_map = {str(i): f"doc_{i}" for i in range(n_docs)}

    results = two_stage_search(query_embed, index, id_map, full_embeds)

    assert len(results) == TOP_K
    assert results[0][0] == "doc_12"
    assert np.isclose(results[0][1], 1.0, atol=1e-4)


# ── 4. Real BEIR SciFact Corpus, Queries & Pre-computed HyDE Cache Tests ─────

def test_real_scifact_corpus_and_hyde_caches_integrity():
    corpus = load_corpus()
    assert len(corpus) == 5183, "BEIR SciFact corpus must contain exactly 5,183 scientific abstracts"

    q_map = load_queries_map()
    assert len(q_map) >= 300, "BEIR SciFact queries map must contain at least 300 claims"

    hyde_cache = load_hyde_cache()
    assert len(hyde_cache) >= 300, "Pre-computed HyDE + GOLFer cache must contain at least 300 test queries"


# ── 5. Pre-built FAISS Index Integrity Tests ─────────────────────────────────

def test_prebuilt_faiss_indices_dimensions_and_counts():
    fast_index, fast_id_map = load_fast_assets()
    assert fast_index.d == 256
    assert fast_index.ntotal == 5183
    assert len(fast_id_map) == 5183

    pro_index = faiss.read_index(str(PRO_INDEX))
    with open(PRO_IDMAP, encoding="utf-8") as f:
        pro_id_map = json.load(f)
    assert pro_index.d == 256
    assert pro_index.ntotal == 5183
    assert len(pro_id_map) == 5183


# ── 6. BEIR IR Ranking Metrics Verification ──────────────────────────────────

def test_beir_ranking_metrics_calculation():
    from pipelines_offline.evaluate_cli import evaluate_predictions

    qrels = {
        "q1": {"d1": 1, "d2": 1},
        "q2": {"d10": 1},
    }
    results = {
        "q1": {"d1": 0.95, "d3": 0.80, "d2": 0.75},
        "q2": {"d10": 0.99, "d11": 0.50},
    }
    ndcg, _map, recall, precision = evaluate_predictions(qrels, results, k_values=[10])
    assert 0.85 <= ndcg["NDCG@10"] <= 1.0
    assert np.isclose(recall["Recall@10"], 1.0)
    assert np.isclose(precision["P@10"], 0.15)
