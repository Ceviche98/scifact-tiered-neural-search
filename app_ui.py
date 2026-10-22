"""
Streamlit Search UI - Unified (Fast 1-Stage & Pro 2-Stage)
====================
Fast Track  — Stella 400M + MRL (256d FAISS 1-stage lookup)
Pro Track   — HyDE cache + GOLFer + Stella 1.5B + MRL (256d FAISS → 1536d rerank)
"""

from config import (  # must be first: executes env vars before torch/transformers load
    BASE_DIR, APP_VERSION, EXAMPLE_QUERIES,
    FAST_INDEX, PRO_INDEX, HYDE_CACHE,
)

import base64
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scifact")

import streamlit as st
import streamlit.components.v1 as components
import torch.nn.functional as F
from user_interface import neumorphic_switch
from search.embedder import embed
from search.hyde import generate_live_hyde_doc, golfer_filter
from search.loaders import load_corpus, load_fast_assets, load_hyde_cache, load_pro_assets, load_queries_map
from search.models import _ensure_1_5B_model, _ensure_400M_model, _unload_current_model
from search.pipelines import one_stage_search, two_stage_search

# ── Dialogs ────────────────────────────────────────────────────────────────────
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

# ── Initialization ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="SciFact Search", page_icon="🔬", layout="wide")



for key, val in [("model", None), ("tokenizer", None), ("loaded_model_name", None),
                 ("device", None), ("query_box", ""), ("selected_doc_id", None),
                 ("click_counter", 0)]:
    if key not in st.session_state: st.session_state[key] = val

doc_map = load_corpus()
qtext_to_id = load_queries_map()
hyde_cache = load_hyde_cache()

# ── Sidebar Info Panel ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 SciFact Search")
    st.markdown("Scientific claim retrieval over the **BEIR SciFact** corpus (5,183 abstracts).")
    st.divider()

    st.markdown("**Index Status**")

    neon_check = '''<svg viewBox="0 0 24 24" style="width: 20px; height: 20px;">
<circle cx="12" cy="12" r="9" fill="none" stroke="#39ff14" stroke-width="2"/>
<path d="M7 12.5l3 3 7-7" fill="none" stroke="#39ff14" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''
    neon_cross = '''<svg viewBox="0 0 24 24" style="width: 20px; height: 20px;">
<circle cx="12" cy="12" r="9" fill="none" stroke="#ff073a" stroke-width="2"/>
<path d="M9 9l6 6M15 9l-6 6" fill="none" stroke="#ff073a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''

    def status_row(label, is_ok):
        icon = neon_check if is_ok else neon_cross
        return f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; padding-right: 10px;"><span>{label}</span>{icon}</div>'

    st.markdown(status_row("Fast 400M Index", FAST_INDEX.exists()), unsafe_allow_html=True)
    st.markdown(status_row("Pro 1.5B Index", PRO_INDEX.exists()), unsafe_allow_html=True)
    st.markdown(status_row("HyDE Cache", HYDE_CACHE.exists()), unsafe_allow_html=True)

    st.divider()

    # Dynamic Placeholder that gets updated from the bottom of the script
    model_display_placeholder = st.empty()
    _tmp_model = st.session_state.get("loaded_model_name") or "None"
    _tmp_short = _tmp_model.split("/")[-1] if _tmp_model != "None" else "None"
    model_display_placeholder.markdown(f"**Loaded model:** `{_tmp_short}`")

    if st.button("Unload model (free RAM)"):
        _unload_current_model()
        st.success("Model unloaded.")
        st.rerun()

    st.divider()
    st.markdown(f"""
<div style="text-align: center;">
Queen Mary University of London<br>
Information Retrieval (ECS736P)<br>
Coursework II
</div>
""", unsafe_allow_html=True)

    _icon_path = BASE_DIR / "user_interface" / "info_icon.png"
    _icon_b64 = base64.b64encode(_icon_path.read_bytes()).decode() if _icon_path.exists() else ""

    if _icon_b64:
        # key insight: button[kind=...] is a React prop, NOT an HTML attr — CSS can't see it.
        # Instead, target the <p> tag inside the button that lives in stHorizontalBlock
        # (columns). The only button in a stHorizontalBlock in the sidebar is this one.
        # Make <p> a flex container so ::before icon + text both align to center precisely.
        st.markdown(f"""
<style>
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button p {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    line-height: 1 !important;
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

    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        if st.button("Info", type="secondary", key="info_btn", use_container_width=True):
            show_info()

    # Version badge at the very bottom of the sidebar
    st.markdown(f"""
<div style="text-align:center; margin-top:18px;">
    <span class="app-version">Version {APP_VERSION}</span>
</div>
""", unsafe_allow_html=True)

# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown("""<style>
:root {
    --fs-title:   1.9rem;
    --fs-body:    0.92rem;
    --fs-version: 0.68rem;
    --bar-h:      3.8rem;
    --bar-bg:     #262730;
    --bar-r:      16px; /* Increased from 10px to 16px */
    --btn-ins:    6px;
}

.stMarkdown p, .stMarkdown li, .stMarkdown small,
.stMarkdown label, .stTextInput label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] label { font-size: var(--fs-body) !important; }
.app-version { font-size: var(--fs-version) !important; color: #555; letter-spacing: 0.5px; }

/* Shrink sidebar divider margins so everything fits without scrolling */
[data-testid="stSidebar"] hr { margin: 0.85em 0 !important; }

.stHeaderActionElements, [data-testid="stHeaderActionElements"],
h1 a, h2 a, h3 a { display: none !important; }

[data-testid="stMainBlockContainer"] { padding-top: 1.2rem !important; }


/* Pull the search form up tight under the heading row */
[data-testid="stMainBlockContainer"] [data-testid="stForm"] {
    margin-top: -1.3rem !important;
}

/* ── Unified dark bar: stHorizontalBlock becomes the container ── */
div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    background:    var(--bar-bg) !important;
    border-radius: var(--bar-r) !important;
    border:        1px solid rgba(250,250,250,0.15) !important;
    height:        var(--bar-h) !important;
    gap:           0 !important;
    align-items:   center !important;
    overflow:      hidden !important;
    padding-right: var(--btn-ins) !important;
    box-sizing:    border-box !important;
    transition:    border-color 0.2s, box-shadow 0.2s;
}

/* When the inner input is focused, make the ENTIRE unified bar glow red */
div[data-testid="stForm"] [data-testid="stHorizontalBlock"]:focus-within {
    border-color: #ff4b4b !important;
    box-shadow: 0 0 0 1px #ff4b4b !important;
}

/* Input wrappers: transparent so bar background shows through */
div[data-testid="stForm"] [data-testid="stTextInput"],
div[data-testid="stForm"] [data-testid="stTextInput"] > div,
div[data-testid="stForm"] [data-testid="stTextInput"] > div > div {
    background:    transparent !important;
    height:        var(--bar-h) !important;
    margin-bottom: 0 !important;
    border:        none !important;
    box-shadow:    none !important; /* Turns off the disjointed inner red border */
}

/* Input element: invisible, stretches full bar height */
div[data-testid="stForm"] [data-testid="stTextInput"] input {
    background:     transparent !important;
    border:         none !important;
    box-shadow:     none !important;
    outline:        none !important;
    height:         var(--bar-h) !important;
    font-size:      var(--fs-body) !important;
    padding-top:    0 !important;
    padding-bottom: 6px !important; /* Pushes text slightly up */
    color:          inherit !important;
}

/* Position the helper text at the bottom right and hide original text */
div[data-testid="stForm"] div[data-testid="InputInstructions"] {
    position: absolute !important;
    top: 65% !important;  /* Forces text down since 'bottom' can float unpredictably */
    right: 20px !important; /* Pushed further inward from the edge */
    bottom: auto !important;
    transform: none !important;
    font-size: 0 !important; /* Hides original 'Press Enter to submit form' */
}

/* Inject new custom phrase and style it darker + italic */
div[data-testid="stForm"] div[data-testid="InputInstructions"]::after {
    content: "Press Enter to submit the query";
    font-size: 0.75rem !important; /* Restores readable font size */
    color: #555555 !important;     /* Dark grey, less visible */
    font-style: italic !important; /* Requested italic font */
}

/* Button wrapper: full bar height, inset padding floats button inside bar */
div[data-testid="stForm"] [data-testid="stFormSubmitButton"],
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] > div {
    height:      var(--bar-h) !important;
    display:     flex !important;
    align-items: center !important;
    padding:     var(--btn-ins) 0 var(--btn-ins) var(--btn-ins) !important;
    box-sizing:  border-box !important;
}

/* Button: all 4 corners rounded, fills its inset area */
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    height:         100% !important;
    min-height:     0 !important;
    border-radius:  10px !important; /* Outer bar 16px - inset 6px = 10px for perfect contour */
    width:          100% !important;
    font-size:      var(--fs-body) !important;
    margin:         0 !important;
    padding-top:    0 !important;
    padding-bottom: 0 !important;
    box-sizing:     border-box !important;
}
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

# ── Page title + Stats nav button ──────────────────────────────────────────────────────
st.markdown('<h1 style="font-size:var(--fs-title);font-weight:700;margin:0 0 0.3rem 0;">Scientific Claim Search Engine</h1>',
            unsafe_allow_html=True)

# ── Example queries: label + 2×5 pills (full width) ──
st.markdown('<p style="font-size:var(--fs-body);font-weight:600;margin:0 0 6px 0;">Example queries:</p>',
            unsafe_allow_html=True)
pill_cols = st.columns(5)
for i, eq in enumerate(EXAMPLE_QUERIES):
    if pill_cols[i % 5].button(eq[:55] + ("…" if len(eq) > 55 else ""), key=f"pill_{i}",
                               use_container_width=True, help=eq):
        st.session_state["query_box"] = eq

# ── Centered search card (~68% width) ──
if "pro_toggle" not in st.session_state:
    st.session_state.pro_toggle = False

_pad, search_card, _pad2 = st.columns([0.16, 0.68, 0.16])
with search_card:
    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
    # Row 1: heading left, toggle right — SAME split as form row below (0.58 / 0.42)
    hcol, tcol = st.columns([0.58, 0.42])
    with hcol:
        st.markdown(
            '<h2 style="font-size:var(--fs-title);font-weight:700;margin:0.6rem 0 0 10px;">'
            'Enter a scientific claim</h2>',
            unsafe_allow_html=True)
    with tcol:
        is_pro_track = neumorphic_switch(key="pro_toggle", default=st.session_state.pro_toggle)

    # Row 2: fused search bar — input wider, button ~half the original width
    with st.form(key="search_form", border=False):
        icol, bcol = st.columns([0.79, 0.21])
        with icol:
            query = st.text_input("", placeholder="e.g. Bariatric surgery has a positive impact...",
                                  key="query_box", label_visibility="collapsed")
        with bcol:
            search_clicked = st.form_submit_button("Search Database", type="primary",
                                                   use_container_width=True)


if search_clicked:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        st.session_state["selected_doc_id"] = None # Reset document panel on new search
        t0 = time.time()

        if not is_pro_track:
            log.info("=== FAST SEARCH  query=%r ===", query)
            log.info("Loading Stella 400M model…")
            with st.spinner("Loading Stella 400M…"): _ensure_400M_model()
            t_model_loaded = time.time()
            log.info("Stella 400M ready  (%.2fs)", t_model_loaded - t0)

            log.info("Loading fast-tier FAISS index…")
            searcher, id_map = load_fast_assets()
            log.info("Index loaded")

            log.info("Computing query embedding…")
            with st.spinner("Executing 1-Stage Search…"):
                q_embed = embed([query], is_query=True)
                log.info("Query embedded — running 1-stage MRL lookup…")
                results = one_stage_search(q_embed, searcher, id_map)
            log.info("Retrieved %d result(s)", len(results))

        else:
            log.info("=== PRO SEARCH  query=%r ===", query)
            filtered = query
            hyde_alpha = 0.7  # Fixed default interpolation as requested

            qid = qtext_to_id.get(query.strip())

            if qid and qid in hyde_cache:
                log.info("Query found in HyDE cache — loading pre-computed hypothetical abstract")
                hyde_text = hyde_cache[qid]
            else:
                log.info("Novel query — generating hypothetical abstract with Qwen (live HyDE)…")
                st.info("Novel query detected. Spinning up Qwen for live HyDE generation...")
                hyde_text = generate_live_hyde_doc(query)
                log.info("Hypothetical abstract generated")

            log.info("--- Hypothetical abstract ---\n%s\n---", hyde_text)

            log.info("Applying GOLFer filter…")
            filtered = golfer_filter(query, hyde_text)
            log.info("--- Filtered text passed to embedder ---\n%s\n---", filtered)

            log.info("Loading Stella 1.5B model…")
            with st.spinner("Loading Stella 1.5B…"): _ensure_1_5B_model()
            t_model_loaded = time.time()
            log.info("Stella 1.5B ready  (%.2fs)", t_model_loaded - t0)

            log.info("Loading Pro-tier search assets…")
            searcher, id_map, full_embeds = load_pro_assets()
            log.info("Assets loaded")

            log.info("Computing query & hypothetical doc embeddings…")
            with st.spinner("Executing 2-Stage Search…"):
                q_embed = embed([query], is_query=True)
                h_embed = embed([filtered], is_query=False)
                log.info("Interpolating embeddings  alpha=%.2f", hyde_alpha)
                combined = hyde_alpha * q_embed + (1 - hyde_alpha) * h_embed
                q_embed = F.normalize(combined.float(), p=2, dim=1)

                log.info("Executing 2-stage search (MRL coarse → full-dim rerank)…")
                results = two_stage_search(q_embed, searcher, id_map, full_embeds)
            log.info("Retrieved %d result(s)", len(results))

        t_end = time.time()
        t_load      = t_model_loaded - t0
        t_inference = t_end - t_model_loaded
        t_total     = t_end - t0
        log.info("=== DONE  load=%.2fs  inference=%.2fs  total=%.2fs ===",
                 t_load, t_inference, t_total)
        st.success(
            f"⏱️ Search completed — "
            f"**Model load:** {t_load:.2f}s  |  "
            f"**Inference:** {t_inference:.2f}s  |  "
            f"**Total:** {t_total:.2f}s"
        )
        st.session_state["current_results"] = results

# ── Split Layout for Results and Abstract Viewer ────────────────────────────────
results = st.session_state.get("current_results", [])
if results:
    if st.session_state["selected_doc_id"] is not None:
        col_list, col_doc = st.columns([0.67, 0.33])
    else:
        col_list, col_doc = st.columns([1, 0.0001]) # Effectively single column

    with col_list:
        st.subheader("Results")

        selected_id = st.session_state.get("selected_doc_id")

        # Inject CSS OUTSIDE the scroll container so it applies globally.
        list_container = st.container(height=750) if selected_id else st.container()

        with list_container:
            for rank, (doc_id, score) in enumerate(results, 1):
                info = doc_map.get(doc_id, {})
                title = info.get("title", "Unknown Title")
                words = info.get("abstract", "").split()
                snippet = " ".join(words[:40]) + ("…" if len(words) > 40 else "")
                is_selected = (doc_id == selected_id)

                if is_selected:
                    # Selected card: render as pure HTML div with red border.
                    # CSS :has() is unreliable across Streamlit versions; this is guaranteed.
                    st.markdown(f"""<div style="
                        border: 2px solid rgba(220, 50, 50, 0.85);
                        border-radius: 8px;
                        padding: 14px 16px 10px 16px;
                        margin-bottom: 4px;
                        background: rgba(220, 50, 50, 0.03);
                    ">
                        <p style="font-weight:700; margin:0 0 4px 0;">{rank}. {title}</p>
                        <small style="color:gray;">Score: {score:.4f} | ID: {doc_id}</small>
                        <p style="margin: 10px 0 0 0; font-size: 0.9em;">{snippet}</p>
                    </div>""", unsafe_allow_html=True)
                    st.divider()
                else:
                    with st.container():
                        rc1, rc2 = st.columns([0.85, 0.15])
                        with rc1:
                            st.markdown(f"**{rank}. {title}**\n\n<small style='color:gray;'>Score: {score:.4f} | ID: {doc_id}</small>", unsafe_allow_html=True)
                            st.write(snippet)
                        with rc2:
                            if st.button("📄 Read", key=f"read_{doc_id}"):
                                st.session_state["selected_doc_id"] = doc_id
                                st.session_state["click_counter"] += 1
                                st.rerun()
                        st.divider()

    with col_doc:
        selected_id = st.session_state.get("selected_doc_id")
        if selected_id:
            # Ghost subheader to vertically align with the left side's "Results" header
            st.subheader(" ")

            # 1) Top section (Fixed, no scroll bar, no frame)
            c1, c2 = st.columns([0.75, 0.25])
            with c2:
                if st.button("✖ Close", key="close_panel", use_container_width=True):
                    st.session_state["selected_doc_id"] = None
                    st.rerun()

            # 2) Abstract section (Scrollable, has standard Streamlit frame)
            # React DOM unmount hack: Alternating the wrapper depth forces Streamlit
            # to definitively destroy the old DOM node and reset the scrollbar to the top.
            dom_breaker = st.container()
            if st.session_state.get("click_counter", 0) % 2 == 1:
                dom_breaker = dom_breaker.container()

            with dom_breaker:
                with st.container(height=680):
                    info = doc_map.get(selected_id, {})

                    st.markdown(f"### {info.get('title', 'Unknown Title')}")
                    st.write("---")

                    st.markdown("**Full Abstract**")
                    st.write(info.get("abstract", "No abstract available."))

# ── Dynamic Updates ────────────────────────────────────────────────────────────
# Now that execution is finished, gracefully update the sidebar with whatever model is loaded into RAM.
loaded_model = st.session_state.get("loaded_model_name") or "None"
short_model  = loaded_model.split("/")[-1] if loaded_model != "None" else "None"
model_display_placeholder.markdown(f"**Loaded model:** `{short_model}`")
