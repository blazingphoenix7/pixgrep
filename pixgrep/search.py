from __future__ import annotations

from pathlib import Path

import numpy as np

from .junk import load_junk_scores
from .store import load_index
from .tags import TagStore


class SearchEngine:
    """Brute-force exact cosine search over the built index."""

    def __init__(
        self,
        index_dir: Path,
        embedder,
        hybrid_weight: float = 0.08,
        junk_threshold: float = 0.0,
    ):
        self.index_dir = Path(index_dir)
        self.paths, self.groups, self.emb = load_index(self.index_dir)
        self.embedder = embedder
        self._hybrid_weight = hybrid_weight
        self._junk_threshold = junk_threshold
        self._tags = TagStore(self.index_dir)
        self._junk_scores = load_junk_scores(self.index_dir, len(self.paths))

    @property
    def count(self) -> int:
        return len(self.paths)

    def path_for(self, row: int) -> str:
        if not 0 <= row < len(self.paths):
            raise IndexError(f"row {row} out of range")
        return self.paths[row]

    def text_search(
        self,
        query: str,
        k: int = 24,
        min_ratio: float = 0.6,
        min_score: float = 0.05,
        filters: dict[str, str] | None = None,
        hybrid_weight: float | None = None,
    ) -> list[dict]:
        qv = self.embedder.embed_texts([query])[0]
        hw = hybrid_weight if hybrid_weight is not None else self._hybrid_weight
        lex = None
        if hw > 0 and self._tags.has_data:
            lex = self._tags.lexical_scores(query, len(self.paths))
        return self._rank(
            qv, k,
            min_ratio=min_ratio, min_score=min_score,
            filters=filters, lex_scores=lex, hybrid_weight=hw,
        )

    def image_search(
        self,
        pil_image,
        k: int = 24,
        min_ratio: float = 0.6,
        min_score: float = 0.05,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        qv = self.embedder.embed_images([pil_image])[0]
        return self._rank(
            qv, k,
            min_ratio=min_ratio, min_score=min_score,
            filters=filters,
        )

    def similar(
        self,
        row: int,
        k: int = 24,
        min_ratio: float = 0.6,
        min_score: float = 0.05,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        if not 0 <= row < len(self.paths):
            raise IndexError(f"row {row} out of range")
        qv = self.emb[row]
        return self._rank(
            qv, k, exclude=row,
            min_ratio=min_ratio, min_score=min_score,
            filters=filters,
        )

    def _rank(
        self,
        qv: np.ndarray,
        k: int,
        exclude: int | None = None,
        min_ratio: float = 0.6,
        min_score: float = 0.05,
        filters: dict[str, str] | None = None,
        lex_scores: np.ndarray | None = None,
        hybrid_weight: float = 0.0,
    ) -> list[dict]:
        sims = self.emb @ qv.astype(np.float32)

        if exclude is not None:
            sims[exclude] = -np.inf

        # Apply filter restriction: non-matching rows scored as -inf
        if filters and self._tags.has_data:
            matching = self._tags.rows_matching(filters)
            if matching is not None:
                if len(matching) == 0:
                    return []
                keep = np.zeros(len(self.paths), dtype=bool)
                keep[matching] = True
                sims[~keep] = -np.inf

        # Mask junk rows before top-k partition
        if self._junk_scores is not None and self._junk_threshold > 0:
            sims[self._junk_scores >= self._junk_threshold] = -np.inf

        n_valid = int(np.sum(np.isfinite(sims)))
        k = min(k, n_valid)
        if k <= 0:
            return []

        # Partition and sort by semantic score; convert to list immediately so
        # truth-value checks below always work regardless of cutoff conditions.
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top: list[int] = list(top_idx[np.argsort(-sims[top_idx])])

        # Relevance floors apply to SEMANTIC score only
        best = float(sims[top[0]])
        if min_score > 0:
            top = [i for i in top if float(sims[i]) >= min_score]
        if min_ratio > 0 and best > 0:
            top = [i for i in top if float(sims[i]) >= best * min_ratio]

        if not top:
            return []

        # Hybrid blend: re-rank survivors by semantic + w * lexical
        if lex_scores is not None and hybrid_weight > 0 and len(lex_scores) == len(self.paths):
            hybrid = {i: float(sims[i]) + hybrid_weight * float(lex_scores[i]) for i in top}
            top = sorted(top, key=lambda i: -hybrid[i])
            return [self._result(int(i), hybrid[i]) for i in top]

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
