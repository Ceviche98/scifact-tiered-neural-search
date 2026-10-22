"""
Stats Page — Search Analytics
===================================
Comparative evaluation of Fast (Stella 400M, 1-stage) vs Pro (Stella 1.5B + HyDE, 2-stage)
over the full BEIR SciFact test query set with real-time charts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # ensure project root is on sys.path

from config import (  # must be first: executes env vars before torch/transformers load
    BASE_DIR, APP_VERSION, EXAMPLE_QUERIES,
    FAST_INDEX, PRO_INDEX, HYDE_CACHE,
    QUERIES_FILE, QRELS_FILE,
)

import base64
import json
import math
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch.nn.functional as F

from search.embedder import embed
from search.hyde import golfer_filter
from search.loaders import load_fast_assets, load_hyde_cache, load_pro_assets
from search.models import _ensure_400M_model, _ensure_1_5B_model
from search.pipelines import one_stage_search, two_stage_search

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SciFact Analytics", page_icon="📊", layout="wide")

@st.dialog("Authors & Info")
def show_info():
    st.markdown("""
**Queen Mary University of London**
**Information Retrieval (ECS736P)**
**Coursework II**

### Authors:
- Oscar Gabriel Moreno Veliz (240121150)
- Min Thant Kyaw (250968910)
- Mohamed Amiin Mohamed (250174186)
- Andres Julian Herrera Medina (250371774)

*All rights reserved 2026*
""")

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {
    "model": None, "tokenizer": None, "loaded_model_name": None, "device": None,
    "eval_phase": "idle",      # idle | fast_running | pro_running | complete
    "fast_per_query": [],      # list of per-query metric dicts
    "pro_per_query":  [],
    "fast_summary":   None,
    "pro_summary":    None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""<style>
:root {
    --accent:   #ff4b4b;
    --accent2:  #7b61ff;
    --bar-bg:   #262730;
    --fs-body:  0.92rem;
    --fs-title: 1.9rem;
}

.stMarkdown p, .stMarkdown li, .stMarkdown small,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] label { font-size: var(--fs-body) !important; }

.stHeaderActionElements, [data-testid="stHeaderActionElements"],
h1 a, h2 a, h3 a { display: none !important; }

[data-testid="stMainBlockContainer"] { padding-top: 1.2rem !important; }

/* ── Sidebar nav item: the eval trigger button ── */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background:    transparent !important;
    border:        1px solid rgba(255,255,255,0.1) !important;
    color:         #ccc !important;
    text-align:    left !important;
    font-weight:   500 !important;
    border-radius: 8px !important;
    transition:    background 0.18s, color 0.18s, border-color 0.18s !important;
    padding:       10px 14px !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background:   rgba(255, 75, 75, 0.18) !important;
    border-color: rgba(255, 75, 75, 0.5)  !important;
    color:        #ff4b4b !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:disabled {
    background:   rgba(255, 75, 75, 0.12) !important;
    border-color: rgba(255, 75, 75, 0.4)  !important;
    color:        #ff4b4b !important;
    opacity:      1 !important;
    cursor:       default !important;
}

/* ── Progress bar in red ── */
[data-testid="stProgressBar"] > div {
    background: var(--accent) !important;
}

/* ── Stats button in top-right of main screen ── */
[data-testid="stPageLink"] a {
    background:    var(--bar-bg)  !important;
    border:        1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    padding:       6px 14px !important;
    color:         #ccc !important;
    font-size:     var(--fs-body) !important;
    font-weight:   600 !important;
    transition:    background 0.2s, color 0.2s !important;
    text-decoration: none !important;
}
[data-testid="stPageLink"] a:hover {
    background: rgba(255,75,75,0.18) !important;
    color:      #ff4b4b !important;
    border-color: rgba(255,75,75,0.5) !important;
}

/* ── Tier label headers ── */
.tier-label-fast {
    color: #ff4b4b; font-weight: 700; font-size: 1rem; margin-bottom: 8px;
    padding: 8px 12px; background: rgba(255,75,75,0.08);
    border-left: 3px solid #ff4b4b; border-radius: 0 6px 6px 0;
}
.tier-label-pro {
    color: #7b61ff; font-weight: 700; font-size: 1rem; margin-bottom: 8px;
    padding: 8px 12px; background: rgba(123,97,255,0.08);
    border-left: 3px solid #7b61ff; border-radius: 0 6px 6px 0;
}

/* Shrink sidebar divider margins so everything fits without scrolling */
[data-testid="stSidebar"] hr { margin: 0.4em 0 !important; }
.app-version { font-size: 0.68rem !important; color: #555; letter-spacing: 0.5px; }
/* Reduce title margin */
[data-testid="stSidebar"] h1 { margin-top: 0 !important; margin-bottom: 0.2rem !important; padding-top: 0 !important; }
/* Tighten paragraph spacing in sidebar */
[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.3rem !important; }
/* Shrink button height in sidebar */
[data-testid="stSidebar"] .stButton button { padding-top: 0.3rem !important; padding-bottom: 0.3rem !important; }
</style>""", unsafe_allow_html=True)

# ── Dynamic Sidebar Renamer (Without Renaming Files) ──
components.html(
    """<script>
    const doc = window.parent.document;
    const rename = () => {
        doc.querySelectorAll('[data-testid="stSidebarNavLink"] span').forEach(el => {
            if (el.textContent === 'app ui') el.textContent = 'Search Engine';
            if (el.textContent === 'Stats')  el.textContent = 'Analytics';
        });
    };
    new MutationObserver(rename).observe(doc.body, {childList: true, subtree: true});
    rename();
    </script>""",
    height=0, width=0,
)

# ── SVG icons (matching app_ui.py) ─────────────────────────────────────────────
_neon_check = '<svg viewBox="0 0 24 24" style="width:20px;height:20px;"><circle cx="12" cy="12" r="9" fill="none" stroke="#39ff14" stroke-width="2"/><path d="M7 12.5l3 3 7-7" fill="none" stroke="#39ff14" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
_neon_cross = '<svg viewBox="0 0 24 24" style="width:20px;height:20px;"><circle cx="12" cy="12" r="9" fill="none" stroke="#ff073a" stroke-width="2"/><path d="M9 9l6 6M15 9l-6 6" fill="none" stroke="#ff073a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'

def _status_row(label, ok):
    icon = _neon_check if ok else _neon_cross
    return (f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'margin-bottom:12px;padding-right:10px;"><span>{label}</span>{icon}</div>')

# ── Metrics ────────────────────────────────────────────────────────────────────
def _metrics(retrieved, relevant_docs):
    from config import TOP_K
    k = TOP_K
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant_docs)
    prec   = hits / k
    recall = hits / len(relevant_docs) if relevant_docs else 0.0
    ap, h = 0.0, 0
    for i, d in enumerate(top_k):
        if d in relevant_docs:
            h += 1; ap += h / (i + 1)
    ap /= len(relevant_docs) if relevant_docs else 1.0
    mrr = next((1.0 / (i + 1) for i, d in enumerate(top_k) if d in relevant_docs), 0.0)
    dcg  = sum((2**relevant_docs[d] - 1) / math.log2(i + 2)
               for i, d in enumerate(top_k) if d in relevant_docs)
    idcg = sum((2**r - 1) / math.log2(i + 2)
               for i, r in enumerate(sorted(relevant_docs.values(), reverse=True)[:k]))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"ndcg": ndcg, "recall": recall, "mrr": mrr, "map": ap, "precision": prec}

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_eval_data():
    queries = []
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries.append({"id": str(d["_id"]), "text": d["text"].strip()})
    df = pd.read_csv(QRELS_FILE, sep="\t")
    qrels = {}
    for _, row in df.iterrows():
        qid, docid, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if score > 0:
            qrels.setdefault(qid, {})[docid] = score
    queries = [q for q in queries if q["id"] in qrels]
    return queries, qrels

# ── Chart builders (dark Plotly) ───────────────────────────────────────────────
_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#bbb", size=11),
    margin=dict(l=48, r=16, t=44, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

def _dark_layout(**overrides):
    """Merge _DARK with per-chart overrides to avoid duplicate keyword errors."""
    return {**_DARK, **overrides}

def _ndcg_chart(per_query, title, color):
    n = len(per_query)
    _y01 = dict(range=[0, 1], gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(**_dark_layout(
            title=dict(text=title, font=dict(color=color, size=13)),
            xaxis_title="Query #", yaxis_title="nDCG@10",
            yaxis=_y01))
        return fig
    xs = list(range(1, n + 1))
    instant = [r["ndcg"] for r in per_query]
    rolling = [sum(instant[:i+1]) / (i+1) for i in range(n)]
    fig = go.Figure([
        go.Scatter(x=xs, y=instant, mode="lines", name="Per-query",
                   line=dict(color="rgba(200,200,200,0.25)", width=1)),
        go.Scatter(x=xs, y=rolling, mode="lines", name="Rolling avg",
                   line=dict(color=color, width=2.5)),
    ])
    fig.update_layout(**_dark_layout(
        title=dict(text=title, font=dict(color=color, size=13)),
        xaxis_title="Query #", yaxis_title="nDCG@10",
        yaxis=_y01))
    return fig

def _latency_chart(per_query, title, color):
    n = len(per_query)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(**_dark_layout(
            title=dict(text=title, font=dict(color=color, size=13)),
            xaxis_title="Query #", yaxis_title="Latency (ms)"))
        return fig
    xs = list(range(1, n + 1))
    lats = [r["latency_ms"] for r in per_query]
    rolling = [sum(lats[:i+1]) / (i+1) for i in range(n)]
    fig = go.Figure([
        go.Bar(x=xs, y=lats, name="Per-query (ms)",
               marker_color="rgba(150,150,180,0.3)"),
        go.Scatter(x=xs, y=rolling, mode="lines", name="Rolling avg",
                   line=dict(color=color, width=2)),
    ])
    fig.update_layout(**_dark_layout(
        title=dict(text=title, font=dict(color=color, size=13)),
        xaxis_title="Query #", yaxis_title="Latency (ms)",
        barmode="overlay"))
    return fig

def _multi_metric_chart(per_query, title, color):
    """Rolling-average line chart: nDCG@10, Recall@10, MRR@10, MAP@10."""
    n = len(per_query)
    _y01 = dict(range=[0, 1], gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(**_dark_layout(
            title=dict(text=title, font=dict(color=color, size=13)),
            xaxis_title="Query #", yaxis_title="Score",
            yaxis=_y01))
        return fig
    xs = list(range(1, n + 1))
    metrics = [
        ("nDCG@10",   "ndcg",    color,      2.5),
        ("Recall@10", "recall",  "#44d7b6",  1.5),
        ("MRR@10",    "mrr",     "#f7c59f",  1.5),
        ("MAP@10",    "map",     "#a8d8ea",  1.5),
    ]
    traces = []
    for label, key, clr, w in metrics:
        vals = [r[key] for r in per_query]
        rolling = [sum(vals[:i+1]) / (i+1) for i in range(n)]
        traces.append(go.Scatter(x=xs, y=rolling, mode="lines",
                                 name=label, line=dict(color=clr, width=w)))
    fig = go.Figure(traces)
    fig.update_layout(**_dark_layout(
        title=dict(text=title, font=dict(color=color, size=13)),
        xaxis_title="Query #", yaxis_title="Score",
        yaxis=_y01))
    return fig

def _summarise(per_query):
    n = len(per_query)
    if n == 0: return None
    return {
        "nDCG@10":       sum(r["ndcg"]      for r in per_query) / n,
        "Recall@10":     sum(r["recall"]    for r in per_query) / n,
        "MRR@10":        sum(r["mrr"]       for r in per_query) / n,
        "MAP@10":        sum(r["map"]       for r in per_query) / n,
        "Precision@10":  sum(r["precision"] for r in per_query) / n,
        "Avg Latency (ms)": sum(r["latency_ms"] for r in per_query) / n,
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Analytics")
    st.markdown("Comparative evaluation over the full **BEIR SciFact** test query set.")
    st.divider()

    st.markdown('<p style="font-size:0.72rem;font-weight:700;color:#888;'
                'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 8px 4px;">'
                'Evaluations</p>', unsafe_allow_html=True)

    phase = st.session_state.eval_phase
    _running = phase in ("fast_running", "pro_running")
    _btn_label = ("⚡ Fast vs Pro  ▶" if phase == "idle"
                  else "⚡ Fast vs Pro  ↺" if phase == "complete"
                  else "⚡ Fast vs Pro  ⏳")

    if st.button(_btn_label, key="eval_btn", use_container_width=True,
                 type="secondary", disabled=_running):
        st.session_state.update({
            "eval_phase": "fast_running",
            "fast_per_query": [], "pro_per_query": [],
            "fast_summary": None, "pro_summary": None,
        })
        st.rerun()

    st.divider()
    st.markdown("**Index Status**")
    st.markdown(_status_row("Fast 400M Index", FAST_INDEX.exists()), unsafe_allow_html=True)
    st.markdown(_status_row("Pro 1.5B Index",  PRO_INDEX.exists()),  unsafe_allow_html=True)
    st.markdown(_status_row("HyDE Cache",      HYDE_CACHE.exists()), unsafe_allow_html=True)
    st.divider()

    _loaded = st.session_state.get("loaded_model_name") or "None"
    _short  = _loaded.split("/")[-1] if _loaded != "None" else "None"
    st.markdown(f"**Loaded model:** `{_short}`")
    st.divider()

    _icon_path = BASE_DIR / "user_interface" / "info_icon.png"
    _icon_b64 = base64.b64encode(_icon_path.read_bytes()).decode() if _icon_path.exists() else ""
    if _icon_b64:
        st.markdown(f"""
<style>
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button p {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    line-height: 1 !important;
    white-space: nowrap !important;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button p::before {{
    content: "";
    display: block;
    width: 28px;
    height: 28px;
    background: url("data:image/png;base64,{_icon_b64}") no-repeat center / contain;
    flex-shrink: 0;
    margin-right: 6px;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
    justify-content: center;
}}
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center; margin-bottom:8px;">
Queen Mary University of London<br>
Information Retrieval (ECS736P)<br>
Coursework II
</div>
""", unsafe_allow_html=True)

    # Info button — wide centre column so text never wraps
    c1, c2, c3 = st.columns([0.5, 2, 0.5])
    with c2:
        if st.button("Info", type="secondary", key="info_btn", use_container_width=True):
            show_info()

    st.markdown(f"""
<div style="text-align:center; margin-top:10px;">
    <span class="app-version">Version {APP_VERSION}</span>
</div>
""", unsafe_allow_html=True)

# ── Page heading ──────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-size:var(--fs-title);font-weight:700;margin:0 0 0.3rem 0;">'
            'Search Analytics</h1>', unsafe_allow_html=True)

st.markdown("Benchmark the **Fast** and **Pro** retrieval tiers across all test queries. "
            "Metrics: nDCG@10 · Recall@10 · MRR@10 · MAP@10 · Precision@10 · Avg Latency.")

st.divider()

# ── Main content ───────────────────────────────────────────────────────────────
phase = st.session_state.eval_phase

if phase == "idle":
    st.markdown("""
<div style="text-align:center;padding:5rem 2rem;color:#555;">
  <div style="font-size:3.5rem;margin-bottom:1.2rem;">📊</div>
  <div style="font-size:1.15rem;font-weight:700;color:#777;margin-bottom:0.6rem;">
    No evaluation running
  </div>
  <div style="font-size:0.92rem;color:#555;line-height:1.7;">
    Click <strong style="color:#ff4b4b;">⚡ Fast vs Pro</strong> in the left panel
    to start a full comparative evaluation<br>
    across all BEIR SciFact test queries.
  </div>
</div>""", unsafe_allow_html=True)

else:
    # ── Load queries + qrels (cached) ─────────────────────────────────────────
    queries, qrels = _load_eval_data()
    N = len(queries)

    # ── Status strip ──────────────────────────────────────────────────────────
    status_ph = st.empty()
    prog_ph   = st.empty()

    # ── Chart layout: 2 columns, 2 rows each ──────────────────────────────────
    col_f, col_p = st.columns(2)

    with col_f:
        st.markdown('<div class="tier-label-fast">⚡ Fast Tier — Stella 400M (1-stage FAISS)</div>',
                    unsafe_allow_html=True)
        f_metric_ph  = st.empty()
        f_latency_ph = st.empty()

    with col_p:
        st.markdown('<div class="tier-label-pro">🧠 Pro Tier — Stella 1.5B + HyDE (2-stage)</div>',
                    unsafe_allow_html=True)
        p_metric_ph  = st.empty()
        p_latency_ph = st.empty()

    summary_ph = st.empty()

    # ── Render current chart state ─────────────────────────────────────────────
    # Counter incremented on every _redraw() call so each plotly_chart call
    # within a single script execution gets a unique key (prevents auto-ID clash).
    _draw_count = [0]

    def _redraw():
        _draw_count[0] += 1
        c = _draw_count[0]
        fq = st.session_state.fast_per_query
        pq = st.session_state.pro_per_query
        f_metric_ph.plotly_chart(_multi_metric_chart(fq, "Metrics — Fast", "#ff4b4b"),
                                 use_container_width=True, key=f"f_m_{c}")
        f_latency_ph.plotly_chart(_latency_chart(fq, "Latency — Fast", "#ff4b4b"),
                                  use_container_width=True, key=f"f_l_{c}")
        p_metric_ph.plotly_chart(_multi_metric_chart(pq, "Metrics — Pro", "#7b61ff"),
                                 use_container_width=True, key=f"p_m_{c}")
        p_latency_ph.plotly_chart(_latency_chart(pq, "Latency — Pro", "#7b61ff"),
                                  use_container_width=True, key=f"p_l_{c}")

    _redraw()

    # ── Run Fast tier ──────────────────────────────────────────────────────────
    if phase == "fast_running" and len(st.session_state.fast_per_query) < N:
        status_ph.info("⚡ Loading Stella 400M…")
        _ensure_400M_model()
        searcher_f, id_map_f = load_fast_assets()
        STEP = max(1, N // 40)   # update charts ~40 times across the run

        for i, q in enumerate(queries):
            t0 = time.time()
            q_emb = embed([q["text"]], is_query=True)
            results = one_stage_search(q_emb, searcher_f, id_map_f)
            ids = [doc_id for doc_id, _ in results]
            lat_ms = (time.time() - t0) * 1000
            m = _metrics(ids, qrels[q["id"]])
            m["latency_ms"] = lat_ms
            st.session_state.fast_per_query.append(m)

            prog = (i + 1) / N
            prog_ph.progress(prog)
            rolling_ndcg = sum(r["ndcg"] for r in st.session_state.fast_per_query) / (i + 1)
            status_ph.info(
                f"⚡ Fast Tier — query {i+1}/{N} "
                f"| rolling nDCG@10: **{rolling_ndcg:.4f}** "
                f"| lat: {lat_ms:.0f} ms"
            )
            if (i + 1) % STEP == 0 or i == N - 1:
                _redraw()

        st.session_state.fast_summary = _summarise(st.session_state.fast_per_query)
        st.session_state.eval_phase = "pro_running"
        st.rerun()

    # ── Run Pro tier ───────────────────────────────────────────────────────────
    if phase == "pro_running" and len(st.session_state.pro_per_query) < N:
        hyde_cache = load_hyde_cache()
        # Build text → id lookup
        qtext_id = {}
        with open(QUERIES_FILE, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                qtext_id[d["text"].strip()] = str(d["_id"])

        status_ph.info("🧠 Loading Stella 1.5B for Pro tier…")
        _ensure_1_5B_model()
        searcher_p, id_map_p, full_embeds = load_pro_assets()
        HYDE_ALPHA = 0.7
        STEP = max(1, N // 40)

        for i, q in enumerate(queries):
            t0 = time.time()
            q_emb = embed([q["text"]], is_query=True)

            # HyDE interpolation (skip live generation; use cache only)
            qid = qtext_id.get(q["text"])
            if qid and qid in hyde_cache:
                filtered = golfer_filter(q["text"], hyde_cache[qid])
                h_emb    = embed([filtered], is_query=False)
                combined = HYDE_ALPHA * q_emb + (1 - HYDE_ALPHA) * h_emb
                q_emb    = F.normalize(combined.float(), p=2, dim=1)

            results = two_stage_search(q_emb, searcher_p, id_map_p, full_embeds)
            ids = [doc_id for doc_id, _ in results]
            lat_ms = (time.time() - t0) * 1000
            m = _metrics(ids, qrels[q["id"]])
            m["latency_ms"] = lat_ms
            st.session_state.pro_per_query.append(m)

            prog = (i + 1) / N
            prog_ph.progress(prog)
            rolling_ndcg = sum(r["ndcg"] for r in st.session_state.pro_per_query) / (i + 1)
            status_ph.info(
                f"🧠 Pro Tier — query {i+1}/{N} "
                f"| rolling nDCG@10: **{rolling_ndcg:.4f}** "
                f"| lat: {lat_ms:.0f} ms"
            )
            if (i + 1) % STEP == 0 or i == N - 1:
                _redraw()

        st.session_state.pro_summary = _summarise(st.session_state.pro_per_query)
        st.session_state.eval_phase = "complete"
        st.rerun()

    # ── Complete: summary table ────────────────────────────────────────────────
    if phase == "complete":
        prog_ph.progress(1.0)
        status_ph.success("✅ Evaluation complete — both tiers finished!")
        _redraw()

        fs = st.session_state.fast_summary
        ps = st.session_state.pro_summary

        if fs and ps:
            with summary_ph.container():
                st.markdown("---")
                st.markdown("### 📋 Summary — Fast vs Pro")

                METRIC_KEYS = ["nDCG@10", "Recall@10", "MRR@10",
                               "MAP@10", "Precision@10", "Avg Latency (ms)"]
                rows = []
                for mk in METRIC_KEYS:
                    fv, pv = fs[mk], ps[mk]
                    fast_wins = (fv < pv) if mk == "Avg Latency (ms)" else (fv >= pv)
                    fstr = f"{fv:.4f}" if mk != "Avg Latency (ms)" else f"{fv:.1f} ms"
                    pstr = f"{pv:.4f}" if mk != "Avg Latency (ms)" else f"{pv:.1f} ms"

                    if mk == "Avg Latency (ms)":
                        if fv < pv and pv > 0:
                            pct = (pv - fv) / pv * 100
                            fstr += f" (+{pct:.2f}%)"
                        elif pv < fv and fv > 0:
                            pct = (fv - pv) / fv * 100
                            pstr += f" (+{pct:.2f}%)"
                    else:
                        if fv > pv and pv > 0:
                            pct = (fv - pv) / pv * 100
                            fstr += f" (+{pct:.2f}%)"
                        elif pv > fv and fv > 0:
                            pct = (pv - fv) / fv * 100
                            pstr += f" (+{pct:.2f}%)"

                    rows.append({
                        "Metric": mk,
                        "⚡ Fast (400M)":       ("✦ " if fast_wins else "  ") + fstr,
                        "🧠 Pro (1.5B + HyDE)": ("✦ " if not fast_wins else "  ") + pstr,
                    })

                df = pd.DataFrame(rows).set_index("Metric")

                # Issue 3: half-width table
                tbl_col, _ = st.columns([0.5, 0.5])
                with tbl_col:
                    st.dataframe(df, use_container_width=True,
                                 column_config={
                                     "⚡ Fast (400M)":       st.column_config.TextColumn(width="medium"),
                                     "🧠 Pro (1.5B + HyDE)": st.column_config.TextColumn(width="medium"),
                                 })

                ndcg_delta = ps["nDCG@10"] - fs["nDCG@10"]
                lat_delta  = ps["Avg Latency (ms)"] - fs["Avg Latency (ms)"]
                pct_ndcg   = ndcg_delta / fs["nDCG@10"] * 100 if fs["nDCG@10"] > 0 else 0
                pct_lat    = lat_delta  / fs["Avg Latency (ms)"] * 100 if fs["Avg Latency (ms)"] > 0 else 0

                if ndcg_delta > 0:
                    st.info(
                        f"🧠 **Pro** achieves **{pct_ndcg:+.1f}% nDCG@10** over Fast, "
                        f"at **{pct_lat:+.0f}% higher latency** "
                        f"({fs['Avg Latency (ms)']:.0f} ms \u2192 {ps['Avg Latency (ms)']:.0f} ms)."
                    )
                else:
                    st.info(
                        f"⚡ **Fast** matches or exceeds Pro on nDCG@10 "
                        f"({pct_ndcg:+.1f}%) while being "
                        f"**{abs(pct_lat):.0f}% faster** "
                        f"({fs['Avg Latency (ms)']:.0f} ms vs {ps['Avg Latency (ms)']:.0f} ms)."
                    )

    # Issue 2: Re-run button rendered OUTSIDE placeholder so click is reliable
    if phase == "complete" and st.session_state.fast_summary and st.session_state.pro_summary:
        if st.button("🔄 Re-run Evaluation", key="rerun_btn", type="secondary"):
            st.session_state.update({
                "eval_phase": "fast_running",
                "fast_per_query": [], "pro_per_query": [],
                "fast_summary": None, "pro_summary": None,
            })
            st.rerun()
