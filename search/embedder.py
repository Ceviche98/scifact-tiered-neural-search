import streamlit as st
import torch
import torch.nn.functional as F


def _mean_pool(token_emb, attn_mask):
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).to(token_emb.dtype)
    return torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def embed(texts: list[str], is_query: bool) -> torch.Tensor:
    model      = st.session_state["model"]
    tokenizer  = st.session_state["tokenizer"]
    device     = st.session_state["device"]
    model_name = st.session_state["loaded_model_name"]

    if is_query:
        texts = [
            f"Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {t}"
            for t in texts
        ]

    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=False) if "1.5B" in model_name else model(**enc)
        emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb.float(), p=2, dim=1)
    return emb.cpu()
