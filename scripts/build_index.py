"""Build the pixgrep index over the configured image_root."""
from __future__ import annotations

import argparse
import sys
import time

from pixgrep.config import load_config
from pixgrep.indexer import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the pixgrep image index.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Wipe any existing index and rebuild from scratch.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config JSON (default: config.local.json).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    print(f"Loading model {cfg.model_id} (engine={cfg.engine}) ...")
    embedder = cfg.make_embedder()
    resume = not args.no_resume
    print(f"Indexing images under: {cfg.image_root}  (resume={resume})")
    t0 = time.time()
    result = build_index(cfg, embedder, resume=resume)
    dt = time.time() - t0
    print(
        f"\nDone in {dt:.1f}s — indexed {result['indexed']}, "
        f"dupes {result['dupes']}, quarantined {result['quarantined']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
