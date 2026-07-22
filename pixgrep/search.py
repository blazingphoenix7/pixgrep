from __future__ import annotations

from pathlib import Path

import numpy as np

from .store import load_index


class SearchEngine:
    """Brute-force exact cosine search over the built index."""

    def __init__(self, index_dir: Path, embedder):
        self.paths, self.groups, self.emb = load_index(Path(index_dir))
        self.embedder = embedder

    @property
    def count(self) -> int:
        return len(self.paths)

    def path_for(self, row: int) -> str:
        if not 0 <= row < len(self.paths):
            raise IndexError(f"row {row} out of range")
        return self.paths[row]

    def text_search(self, query: str, k: int = 24) -> list[dict]:
        qv = self.embedder.embed_texts([query])[0]
        return self._rank(qv, k)

    def image_search(self, pil_image, k: int = 24) -> list[dict]:
        qv = self.embedder.embed_images([pil_image])[0]
        return self._rank(qv, k)

    def similar(self, row: int, k: int = 24) -> list[dict]:
        if not 0 <= row < len(self.paths):
            raise IndexError(f"row {row} out of range")
        qv = self.emb[row]
        return self._rank(qv, k, exclude=row)

    def _rank(self, qv: np.ndarray, k: int, exclude: int | None = None) -> list[dict]:
        sims = self.emb @ qv.astype(np.float32)
        if exclude is not None:
            sims[exclude] = -np.inf
        k = min(k, len(self.paths) - (1 if exclude is not None else 0))
        if k <= 0:
            return []
        top = np.argpartition(-sims, kth=k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [self._result(int(i), float(sims[i])) for i in top]

    def _result(self, row: int, score: float) -> dict:
        p = Path(self.paths[row])
        return {
            "row": row,
            "score": round(score, 4),
            "path": self.paths[row],
            "name": p.name,
            "group": self.groups[row],
            "folder": p.parent.name,
        }
