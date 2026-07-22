from __future__ import annotations

from collections import Counter

import numpy as np


def group_recall_at_k(
    embeddings: np.ndarray, group_keys: list[str], k: int = 10
) -> float:
    n = len(group_keys)
    if n < 2:
        return 0.0
    groups = np.asarray(group_keys)
    counts = Counter(group_keys)
    evaluable = [i for i in range(n) if counts[group_keys[i]] >= 2]
    if not evaluable:
        return 0.0

    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)  # never retrieve self

    kk = min(k, n - 1)
    total_recall = 0.0
    for i in evaluable:
        topk = np.argpartition(-sims[i], kth=kk - 1)[:kk]
        retrieved_siblings = int(np.sum(groups[topk] == group_keys[i]))
        n_siblings = counts[group_keys[i]] - 1
        total_recall += retrieved_siblings / n_siblings
    return total_recall / len(evaluable)
