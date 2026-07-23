"""Compute and save zero-shot junk scores for all indexed images.

Usage:
    python scripts/flag_junk.py --config config.local.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pixgrep.config import load_config
from pixgrep.junk import junk_scores, save_junk_scores
from pixgrep.store import load_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Score indexed images for junk-ness")
    parser.add_argument("--config", default="config.local.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loading index from {cfg.index_dir} ...")
    paths, _groups, emb = load_index(cfg.index_dir)
    n = len(paths)
    print(f"  {n} rows loaded.")

    print("Loading text embedder ...")
    embedder = cfg.make_embedder()

    print("Computing junk scores ...")
    scores = junk_scores(emb, embedder)

    print("Saving scores to DB ...")
    save_junk_scores(cfg.index_dir, scores)

    pcts = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    thresholds = [0.0, 0.05, 0.1, 0.15]

    print("\n--- Score distribution (N={}) ---".format(n))
    for p in pcts:
        print(f"  p{p:3d}: {np.percentile(scores, p):.4f}")

    print("\n--- Counts at or above threshold ---")
    for t in thresholds:
        count = int(np.sum(scores >= t))
        print(f"  >= {t:.2f}: {count}/{n}  ({100.0 * count / max(n, 1):.1f}%)")


if __name__ == "__main__":
    main()
