"""Measure SigLIP 2 embedding throughput on a sample of the configured images."""
from __future__ import annotations

import sys

from pixgrep.config import load_config
from pixgrep.embedding import Embedder
from pixgrep.indexer import find_images, load_rgb
from pixgrep.throughput import measure


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cfg = load_config()
    files = find_images(cfg.image_root)[:n]
    imgs = [img for img in (load_rgb(f) for f in files) if img is not None]
    print(f"Loaded {len(imgs)} images; loading model ...")
    embedder = Embedder(cfg.model_id)

    def embed_fn(batch):
        return embedder.embed_images(batch)

    rate = measure(embed_fn, imgs, warmup=min(4, len(imgs)))
    print(f"Throughput: {rate:.1f} images/sec on {cfg.model_id} ({embedder.device})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
