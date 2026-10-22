import gc
import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from search.models import _get_device, _unload_current_model


_STOP = frozenset(
    "the a an is are was were be been being have has had do does did will would could "
    "should may might shall can need to of in for on with at by from as into through "
    "during before after above below between out off over under again further then once "
    "that this these those and but or nor not so yet both either neither each every all "
    "any few more most other some such no only own same than too very just because if "
    "when where how what which who whom why it its we they their them he she his her "
    "our you your".split()
)


def golfer_filter(query: str, hyde_text: str, min_overlap: int = 2) -> str:
    q_words = set(re.findall(r"\w+", query.lower())) - _STOP
    sentences = re.split(r"(?<=[.!?])\s+", hyde_text.strip())
    kept = [s for s in sentences
            if len(q_words & (set(re.findall(r"\w+", s.lower())) - _STOP)) >= min_overlap]
    return " ".join(kept) if kept else query


def generate_live_hyde_doc(query: str) -> str:
    """Load Qwen, generate HyDE text, then immediately unload to save memory."""
    qwen_name = "Qwen/Qwen2.5-1.5B-Instruct"
    _unload_current_model()
    device = _get_device()

    with st.spinner(f"Generating HyDE Document with {qwen_name} (this may take a moment)…"):
        # Explicitly use_fast=False to bypass rust enum ModelWrapper json parsing crashes
        tokenizer = AutoTokenizer.from_pretrained(qwen_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            qwen_name, torch_dtype=torch.float16
        ).to(device)
        model.eval()

        prompt = (
            "Please write a highly technical, factual scientific abstract "
            "that provides evidence to answer this claim: "
            f"{query}\n\nAbstract:\n"
        )
        messages = [
            {"role": "system", "content": "You are a brilliant biomedical researcher. Output only the requested abstract text."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        hyde_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Clean up Qwen aggressively before Stella loads back
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return hyde_text
