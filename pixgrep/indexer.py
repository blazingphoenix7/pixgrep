from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

from .config import Config
from .filenames import group_key
from .store import save_index

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_rgb(path: Path):
    try:
        img = Image.open(path)
        img.draft("RGB", (256, 256))  # decode at reduced scale where possible
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def build_index(cfg: Config, embedder) -> dict:
    files = find_images(cfg.image_root)
    paths: list[str] = []
    groups: list[str] = []
    vecs: list[np.ndarray] = []
    skipped: list[str] = []

    batch_imgs: list = []
    batch_meta: list[tuple[str, str]] = []

    def flush():
        if not batch_imgs:
            return
        v = embedder.embed_images(batch_imgs)
        vecs.append(v)
        for path_str, gkey in batch_meta:
            paths.append(path_str)
            groups.append(gkey)
        batch_imgs.clear()
        batch_meta.clear()

    for f in files:
        img = load_rgb(f)
        if img is None:
            skipped.append(str(f))
            continue
        batch_imgs.append(img)
        batch_meta.append((str(f), group_key(f.name, cfg.group_strip_pattern)))
        if len(batch_imgs) >= cfg.batch_size:
            flush()
    flush()

    if vecs:
        embeddings = np.vstack(vecs).astype(np.float32)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)

    save_index(cfg.index_dir, paths, groups, embeddings)
    return {"indexed": len(paths), "skipped": len(skipped), "skipped_files": skipped}
