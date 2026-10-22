"""
Offline Matryoshka Representation Learning (MRL) FAISS Index Builder & Two-Stage Evaluator.

Supports building both the Fast Tier (NovaSearch/stella_en_400M_v5, 256d 1-Stage)
and the Pro Tier (NovaSearch/stella_en_1.5B_v5, 256d Stage-1 FAISS + 1536d Stage-2 MRL cache).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
for p in (THIS_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_loader import load_scifact_local

device = "cuda" if torch.cuda.is_available() else "cpu"


class StellaMRLRetriever:
    """Dense encoder supporting Matryoshka (MRL) prefix slicing and L2 re-normalization."""

    def __init__(self, model_name: str = "NovaSearch/stella_en_1.5B_v5", full_dim: int = 1536):
        print(f"Loading {model_name} on {device}...")
        self.model_name = model_name
        self.full_dim = full_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 16, is_query: bool = False) -> np.ndarray:
        if is_query:
            prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
            texts = [prompt + t for t in texts]

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", disable=is_query and len(texts) < 10):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            outputs = self.model(**inputs)

            attention_mask = inputs["attention_mask"]
            last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)


def build_tier_index(tier: str = "pro"):
    corpus, queries, _ = load_scifact_local()

    if tier == "fast":
        model_name = "NovaSearch/stella_en_400M_v5"
        full_dim = 1024
        out_dir = ROOT_DIR / "data_fast_tier" / "data" / "mrl_index"
        cache_file = None
    else:
        model_name = "NovaSearch/stella_en_1.5B_v5"
        full_dim = 1536
        out_dir = ROOT_DIR / "data_pro_tier" / "mrl_index"
        cache_file = ROOT_DIR / "data_pro_tier" / "mrl_cache.pkl"

    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / "index.faiss"
    id_map_file = out_dir / "id_map.json"

    retriever = StellaMRLRetriever(model_name=model_name, full_dim=full_dim)

    doc_ids = list(corpus.keys())
    doc_texts = [f"{corpus[did].get('title', '')}. {corpus[did].get('text', '')}" for did in doc_ids]

    print(f"Encoding {len(doc_texts)} SciFact abstracts with {model_name}...")
    embeddings_full = retriever.encode(doc_texts, batch_size=8, is_query=False)

    # Matryoshka 256d prefix truncation + L2 re-normalization
    embeddings_256 = embeddings_full[:, :256].copy()
    faiss.normalize_L2(embeddings_256)

    index_256 = faiss.IndexFlatIP(256)
    index_256.add(embeddings_256)
    faiss.write_index(index_256, str(index_file))
    print(f"Saved 256d FAISS index to {index_file}")

    import json
    id_map = {str(i): str(did) for i, did in enumerate(doc_ids)}
    with open(id_map_file, "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    if tier == "pro" and cache_file is not None:
        with open(cache_file, "wb") as f:
            pickle.dump({"embeddings": torch.from_numpy(embeddings_full).to(torch.float16)}, f)
        print(f"Saved 1536d Stage-2 MRL cache to {cache_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Matryoshka FAISS indices for Fast or Pro tier.")
    parser.add_argument("--tier", choices=["fast", "pro"], default="pro", help="Target search tier to build.")
    args = parser.parse_args()
    build_tier_index(args.tier)
