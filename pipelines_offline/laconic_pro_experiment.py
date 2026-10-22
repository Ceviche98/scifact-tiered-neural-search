"""
Phase 1/2 Experimental Pipeline: LACONIC-8B Sparse Neural Retrieval + Qwen2.5-3B HyDE + GOLFer.

Preserved from our Coursework 1 & 2 architectural investigation (`pro_pipeline.py`).
Demonstrates why LACONIC-8B's lexical calibration (1.3B training tokens) underperformed on
zero-shot scientific claims (0.6487 nDCG@10, ~804ms latency), motivating our pivot to
instruction-tuned Stella-1.5B + Matryoshka 2-Stage Reranking (0.8063 nDCG@10).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from data_loader import load_scifact_local

device = "cuda" if torch.cuda.is_available() else "cpu"


class HyDEGenerator:
    """Generates a hypothetical scientific abstract for a claim and applies GOLFer lexical verification."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"Loading HyDE Generator ({model_name}) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        ).eval()

    @torch.no_grad()
    def generate_hypothetical_abstract(self, claim: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a biomedical scientist. Write a concise, 3-sentence scientific abstract "
                    "that provides empirical evidence for the user's claim. Use precise clinical and "
                    "biological terminology. Do not include introductory conversational text."
                ),
            },
            {"role": "user", "content": f"Claim: {claim}"},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            inputs.input_ids,
            max_new_tokens=120,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def golfer_filter(claim: str, hyde_abstract: str) -> str:
        """GOLFer entity/factual grounding filter to strip ungrounded LLM hallucinations."""
        stopwords = {
            "the", "a", "an", "in", "on", "of", "and", "to", "for", "with",
            "is", "are", "was", "were", "by", "that", "this", "from", "as", "at", "be", "have", "has",
        }
        claim_words = set(re.findall(r"\b\w+\b", claim.lower())) - stopwords
        sentences = re.split(r"(?<=[.!?])\s+", hyde_abstract)
        valid_sentences = []
        for s in sentences:
            s_words = set(re.findall(r"\b\w+\b", s.lower())) - stopwords
            if len(claim_words.intersection(s_words)) >= 1:
                valid_sentences.append(s)
        if not valid_sentences:
            return claim
        return " ".join(valid_sentences)


class LaconicRetriever:
    """Sparse MLM projection layer based on Bottleneck-Masked Language Modeling (`LACONIC-8B`)."""

    def __init__(self, model_name: str = "mixedbread-ai/laconic-8b-v0"):
        print(f"Loading Sparse Retriever ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    @torch.no_grad()
    def encode_sparse(self, texts: list[str], top_k_tokens: int = 128) -> list[dict[int, float]]:
        sparse_vecs = []
        for text in texts:
            inputs = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            max_logits, _ = torch.max(logits * inputs["attention_mask"][0].unsqueeze(-1), dim=0)
            activated = torch.log1p(torch.relu(max_logits))
            top_vals, top_indices = torch.topk(activated, k=top_k_tokens)
            vec = {
                int(idx): round(float(val), 4)
                for idx, val in zip(top_indices.cpu().numpy(), top_vals.cpu().numpy())
                if val > 0
            }
            sparse_vecs.append(vec)
        return sparse_vecs
