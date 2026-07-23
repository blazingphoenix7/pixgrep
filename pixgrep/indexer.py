from __future__ import annotations

import hashlib
import io
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

from .config import Config
from .filenames import group_key as make_group_key
from .store import (
    BIN_FILENAME,
    NPY_FILENAME,
    _meta_int,
    _truncate_file,
    append_vecs,
    get_path_index,
    get_sha1_index,
    open_db,
    overwrite_vec,
    set_meta,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_CHECKPOINT = 256
_THUMB_MAX = 384
_THUMB_QUALITY = 80

log = logging.getLogger(__name__)


def find_images(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_rgb(path: Path):
    try:
        img = Image.open(path)
        img.draft("RGB", (256, 256))
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _load_rgb_bytes(data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        img.draft("RGB", (256, 256))
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _save_thumb(thumbs_dir: Path, row: int, img: Image.Image) -> None:
    try:
        thumbs_dir.mkdir(exist_ok=True)
        t = img.copy()
        t.thumbnail((_THUMB_MAX, _THUMB_MAX), Image.LANCZOS)
        t.convert("RGB").save(thumbs_dir / f"{row}.jpg", "JPEG", quality=_THUMB_QUALITY)
    except Exception:
        pass


def build_index(cfg: Config, embedder, *, resume: bool = True) -> dict:
    index_dir = Path(cfg.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    bin_file = index_dir / BIN_FILENAME
    thumbs_dir = index_dir / "thumbs"
    log_file = index_dir / "build.log"

    con = open_db(index_dir)
    dim: int | None = _meta_int(con, "embedding_dim") if resume else None

    if not resume:
        con.execute("DELETE FROM images")
        con.execute("DELETE FROM duplicates")
        con.execute("DELETE FROM meta")
        con.commit()
        if bin_file.exists():
            bin_file.unlink()
        dim = None
    elif dim is not None and bin_file.exists():
        count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        bin_elems = bin_file.stat().st_size // 2  # float16 = 2 bytes
        if bin_elems > count * dim:
            _truncate_file(bin_file, count * dim * 2)

    npy_file = index_dir / NPY_FILENAME
    if (
        resume
        and not bin_file.exists()
        and npy_file.exists()
        and con.execute("SELECT COUNT(*) FROM images").fetchone()[0] > 0
    ):
        # One-time migration: resuming on top of an old .npy-format index
        # would otherwise leave sqlite rows without matching bin rows.
        old = np.load(str(npy_file))
        append_vecs(bin_file, old)
        dim = int(old.shape[1])
        set_meta(con, "embedding_dim", str(dim))
        set_meta(con, "schema_version", "2")
        con.commit()

    path_idx = get_path_index(con) if resume else {}
    sha1_idx = get_sha1_index(con) if resume else {}
    # Paths already recorded as duplicates: skip on resume without re-reading
    # (re-hashing them would also re-insert duplicate rows every resume).
    dupe_idx: dict[str, tuple[float | None, int | None]] = {}
    if resume:
        for p, sz, mt in con.execute(
            "SELECT path, size, mtime FROM duplicates"
        ).fetchall():
            dupe_idx[p] = (mt, sz)

    files = find_images(cfg.image_root)
    total = len(files)

    done = con.execute("SELECT COUNT(*) FROM images").fetchone()[0] if resume else 0
    dupes = con.execute("SELECT COUNT(*) FROM duplicates").fetchone()[0] if resume else 0
    quarantined = 0

    max_row_res = con.execute("SELECT MAX(row) FROM images").fetchone()[0]
    next_row = (max_row_res + 1) if max_row_res is not None else 0

    batch_imgs: list = []
    # (path, gkey, mtime, size, sha1, old_row_or_None)
    batch_meta: list[tuple[str, str, float, int, str, int | None]] = []
    batch_sha1s: set[str] = set()

    t0 = time.time()
    t_cp = t0

    stop_flag = False

    def _on_stop(sig, frame):
        nonlocal stop_flag
        stop_flag = True

    orig_int = signal.getsignal(signal.SIGINT)
    orig_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    def flush() -> None:
        nonlocal dim, next_row, done, t_cp

        if not batch_imgs:
            return

        vecs = embedder.embed_images(batch_imgs)

        if dim is None:
            dim = int(vecs.shape[1])
            set_meta(con, "embedding_dim", str(dim))
            set_meta(con, "schema_version", "2")

        updates: list[tuple[int, np.ndarray, str, float, int, str]] = []
        inserts: list[tuple[Image.Image, str, str, float, int, str, np.ndarray]] = []

        for i, (path, gkey, mtime, size, sha1, old_row) in enumerate(batch_meta):
            vec = vecs[i]
            if old_row is not None:
                updates.append((old_row, vec, path, mtime, size, sha1))
            else:
                inserts.append((batch_imgs[i], path, gkey, mtime, size, sha1, vec))

        # Embeddings written BEFORE sqlite commit for crash safety
        for old_row, vec, _, _, _, _ in updates:
            overwrite_vec(bin_file, old_row, dim, vec)

        if inserts:
            append_vecs(bin_file, np.vstack([d[-1] for d in inserts]))

        # Sqlite commit
        for old_row, _, path, mtime, size, sha1 in updates:
            old = con.execute("SELECT sha1 FROM images WHERE row=?", (old_row,)).fetchone()
            if old and old[0] in sha1_idx:
                del sha1_idx[old[0]]
            con.execute(
                "UPDATE images SET mtime=?, size=?, sha1=? WHERE row=?",
                (mtime, size, sha1, old_row),
            )
            path_idx[path] = (old_row, mtime, size)
            sha1_idx[sha1] = old_row

        for img, path, gkey, mtime, size, sha1, _ in inserts:
            con.execute(
                "INSERT INTO images (row, path, group_key, mtime, size, sha1) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (next_row, path, gkey, mtime, size, sha1),
            )
            path_idx[path] = (next_row, mtime, size)
            sha1_idx[sha1] = next_row
            _save_thumb(thumbs_dir, next_row, img)
            next_row += 1

        con.commit()

        # Save thumbs for updated files (row ids known)
        for old_row, _, path, _, _, _ in updates:
            idx = next((i for i, m in enumerate(batch_meta) if m[0] == path), None)
            if idx is not None:
                _save_thumb(thumbs_dir, old_row, batch_imgs[idx])

        done = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]

        t_now = time.time()
        interval = t_now - t_cp or 1e-6
        rate = len(batch_imgs) / interval
        remaining = max(total - done - dupes - quarantined, 0)
        eta = f"{int(remaining / rate)}s" if rate > 0 else "?"
        line = (
            f"indexed={done} dupes={dupes} quarantined={quarantined} "
            f"rate={rate:.1f}img/s ETA={eta}"
        )
        print(line, flush=True)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

        t_cp = t_now
        batch_sha1s.clear()
        batch_imgs.clear()
        batch_meta.clear()

    interrupted = False
    try:
        for f in files:
            if stop_flag:
                break

            path_str = str(f)

            try:
                st = f.stat()
            except OSError:
                quarantined += 1
                continue

            mtime, size = st.st_mtime, st.st_size

            old_row: int | None = None
            if path_str in path_idx:
                cached_row, cached_mtime, cached_size = path_idx[path_str]
                if cached_mtime == mtime and cached_size == size:
                    continue  # unchanged: skip
                old_row = cached_row  # changed file: re-index in place
            elif path_str in dupe_idx:
                d_mtime, d_size = dupe_idx[path_str]
                if d_mtime == mtime and d_size == size:
                    continue  # unchanged known duplicate: skip re-hash

            try:
                data = f.read_bytes()
            except OSError:
                quarantined += 1
                continue

            sha1 = _sha1(data)

            # SHA1 dedup: same content as another committed image
            if old_row is None and (sha1 in sha1_idx or sha1 in batch_sha1s):
                con.execute(
                    "INSERT INTO duplicates (path, size, mtime, sha1, duplicate_of) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (path_str, size, mtime, sha1, sha1_idx.get(sha1)),
                )
                con.commit()
                dupes += 1
                continue

            # For changed files that now match a different image's content
            if old_row is not None and sha1 in sha1_idx and sha1_idx[sha1] != old_row:
                con.execute(
                    "INSERT INTO duplicates (path, size, mtime, sha1, duplicate_of) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (path_str, size, mtime, sha1, sha1_idx[sha1]),
                )
                con.commit()
                dupes += 1
                continue

            img = _load_rgb_bytes(data)
            if img is None:
                log.warning("quarantine: %s", path_str)
                quarantined += 1
                continue

            gkey = make_group_key(f.name, cfg.group_strip_pattern)
            batch_imgs.append(img)
            batch_meta.append((path_str, gkey, mtime, size, sha1, old_row))
            batch_sha1s.add(sha1)

            if len(batch_imgs) >= _CHECKPOINT:
                flush()
                if stop_flag:
                    break

        flush()

    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        try:
            flush()
        except Exception:
            pass
    finally:
        con.close()
        signal.signal(signal.SIGINT, orig_int)
        signal.signal(signal.SIGTERM, orig_term)

    if interrupted or stop_flag:
        print("resumable — rerun to continue", flush=True)
        sys.exit(0)

    return {
        "indexed": done,
        "dupes": dupes,
        "quarantined": quarantined,
        "skipped": quarantined,
        "skipped_files": [],
    }
