"""
Data loader utilities for BEIR SciFact offline indexing, CSR autoencoder training,
and ablation benchmark scripts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "scifact_data"


def load_scifact_local(data_dir: str | Path = DEFAULT_DATA_DIR):
    """Load the BEIR SciFact corpus, test queries, and ground-truth relevance judgments (qrels)."""
    data_path = Path(data_dir)
    corpus_file = data_path / "corpus.jsonl"
    queries_file = data_path / "queries.jsonl"
    qrels_file = data_path / "qrels" / "test.tsv"

    print(f"Loading local BEIR SciFact dataset from {data_path}...")

    corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[str(doc["_id"])] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }

    queries = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[str(q["_id"])] = q.get("text", "")

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, docid, score = str(row[0]), str(row[1]), int(row[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = score

    test_queries = {qid: queries[qid] for qid in qrels if qid in queries}
    print(f"Loaded {len(corpus)} documents and {len(test_queries)} evaluated test queries.")
    return corpus, test_queries, qrels
