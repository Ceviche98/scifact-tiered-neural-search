"""
Command-Line BEIR Evaluation Utility (nDCG@10, Recall@10, MRR@10, MAP@10, Precision@10).
"""

from __future__ import annotations

import numpy as np


def evaluate_predictions(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int] = [1, 3, 5, 10, 100],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Compute official BEIR ranking metrics using `pytrec_eval` (or pure-Python fallback)."""
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval

        evaluator = EvaluateRetrieval()
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
        return ndcg, _map, recall, precision
    except ImportError:
        # Pure-Python fallback for environments without beir/pytrec_eval installed
        ndcg_10, map_10, recall_10, prec_10 = [], [], [], []
        for qid, rel_docs in qrels.items():
            rel_set = {str(did) for did, score in rel_docs.items() if score > 0}
            if not rel_set or qid not in results:
                continue
            ranked = [
                str(did) for did, _ in sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            hits = [1 if d in rel_set else 0 for d in ranked]
            n_rel = len(rel_set)

            prec_10.append(sum(hits) / 10.0)
            recall_10.append(sum(hits) / n_rel)

            ap, running = 0.0, 0
            for i, h in enumerate(hits):
                if h:
                    running += 1
                    ap += running / (i + 1)
            map_10.append(ap / min(n_rel, 10))

            dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_rel, 10)))
            ndcg_10.append(dcg / idcg if idcg > 0 else 0.0)

        return (
            {"NDCG@10": float(np.mean(ndcg_10))},
            {"MAP@10": float(np.mean(map_10))},
            {"Recall@10": float(np.mean(recall_10))},
            {"P@10": float(np.mean(prec_10))},
        )
