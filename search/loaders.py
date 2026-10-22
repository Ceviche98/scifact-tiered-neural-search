import json
import pickle

import faiss
import streamlit as st

from config import (
    CORPUS_FILE, QUERIES_FILE, HYDE_CACHE,
    FAST_INDEX, FAST_IDMAP,
    PRO_INDEX, PRO_IDMAP, PRO_CACHE,
)
from scripts.download_assets import ensure_pro_cache


@st.cache_data(show_spinner="Loading corpus…")
def load_corpus() -> dict:
    doc_map = {}
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc_map[str(d["_id"])] = {
                "title":    d.get("title", "").strip(),
                "abstract": d.get("text",  "").strip(),
                "metadata": d.get("metadata", {}),
            }
    return doc_map


@st.cache_data(show_spinner="Loading queries map…")
def load_queries_map() -> dict:
    qtext_to_id = {}
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                qtext_to_id[d["text"].strip()] = str(d["_id"])
    return qtext_to_id


@st.cache_data(show_spinner="Loading HyDE cache…")
def load_hyde_cache() -> dict:
    if HYDE_CACHE.exists():
        with open(HYDE_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner="Loading Fast index (400M)…")
def load_fast_assets():
    searcher = faiss.read_index(str(FAST_INDEX))
    with open(FAST_IDMAP, encoding="utf-8") as f:
        id_map = json.load(f)
    return searcher, id_map


@st.cache_resource(show_spinner="Loading Pro index (1.5B)…")
def load_pro_assets():
    if not PRO_CACHE.exists():
        ensure_pro_cache(quiet=True)
    searcher = faiss.read_index(str(PRO_INDEX))
    with open(PRO_IDMAP, encoding="utf-8") as f:
        id_map = json.load(f)
    with open(PRO_CACHE, "rb") as f:
        cache = pickle.load(f)
    return searcher, id_map, cache["embeddings"]
