"""Evaluate filename-group recall over the built index — the P0 quality gate."""
from __future__ import annotations

import sys
from collections import Counter

from pixgrep.config import load_config
from pixgrep.metrics import group_recall_at_k
from pixgrep.store import load_index


def main() -> int:
    cfg = load_config()
    paths, groups, emb = load_index(cfg.index_dir)
    counts = Counter(groups)
    multi = sum(1 for g in groups if counts[g] >= 2)
    print(f"Items indexed: {len(paths)}")
    print(f"Items with >=2 same-group members (evaluable): {multi}")
    print(f"Distinct groups: {len(counts)}")
    for k in (5, 10):
        r = group_recall_at_k(emb, groups, k=k)
        print(f"group_recall@{k}: {r:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
