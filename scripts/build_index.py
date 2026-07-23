"""Build the pixgrep index over the configured image_root."""
from __future__ import annotations

import sys
import time

from pixgrep.config import load_config
from pixgrep.indexer import build_index


def main() -> int:
    cfg = load_config()
    print(f"Loading model {cfg.model_id} (engine={cfg.engine}) ...")
    embedder = cfg.make_embedder()
    print(f"Indexing images under: {cfg.image_root}")
    t0 = time.time()
    result = build_index(cfg, embedder)
    dt = time.time() - t0
    print(
        f"Done in {dt:.1f}s — indexed {result['indexed']}, "
        f"skipped {result['skipped']}."
    )
    if result["skipped"]:
        print("First few skipped files:")
        for s in result["skipped_files"][:10]:
            print("  ", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
