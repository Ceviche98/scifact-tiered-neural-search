import numpy as np
import torch
import torch.nn.functional as F

from config import MRL_DIM, STAGE1_K, TOP_K


def one_stage_search(query_embed: torch.Tensor, searcher, id_map) -> list[tuple[str, float]]:
    q_trunc = query_embed[:, :MRL_DIM]
    q_trunc = F.normalize(q_trunc, p=2, dim=1).numpy().astype(np.float32)

    distances, indices = searcher.search(q_trunc, k=TOP_K)
    int_ids = indices[0]
    scores  = distances[0]

    valid = int_ids != -1
    return [(id_map[str(iid)], float(s)) for iid, s in zip(int_ids[valid], scores[valid])]


def two_stage_search(query_embed: torch.Tensor, searcher, id_map, full_embeds) -> list[tuple[str, float]]:
    q_trunc = query_embed[:, :MRL_DIM]
    q_trunc = F.normalize(q_trunc, p=2, dim=1).numpy().astype(np.float32)

    _, indices = searcher.search(q_trunc, k=STAGE1_K)
    int_ids = indices[0]
    int_ids = int_ids[int_ids != -1]

    doc_vecs = full_embeds[int_ids].float()
    scores = torch.matmul(query_embed.float(), doc_vecs.T).squeeze()
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)

    top_scores, top_idx = torch.topk(scores, k=min(TOP_K, len(scores)))
    final_ids = int_ids[top_idx.numpy()]
    return [(id_map[str(iid)], float(s)) for iid, s in zip(final_ids, top_scores.tolist())]
