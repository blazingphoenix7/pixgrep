from __future__ import annotations

from pathlib import Path

import numpy as np

from .junk import load_junk_scores
from .query_norm import normalize_query
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
        group_strip_pattern: str = "",
        near_dupe_cos: float = 0.985,
        lexical_inject_k: int = 50,
        junk_soft_weight: float = 1.0,
    ):
        self.index_dir = Path(index_dir)
        self.paths, self.groups, self.emb = load_index(self.index_dir)
        self.embedder = embedder
        self._hybrid_weight = hybrid_weight
        self._junk_threshold = junk_threshold
        self._group_strip_pattern = group_strip_pattern
        self._near_dupe_cos = near_dupe_cos
        self._lexical_inject_k = lexical_inject_k
        self._junk_soft_weight = junk_soft_weight
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
        query = normalize_query(query)
        qv = self.embedder.embed_texts([query])[0]
        hw = hybrid_weight if hybrid_weight is not None else self._hybrid_weight
        lex = None
        inject_rows = None
        if self._tags.has_data:
            if hw > 0 or self._lexical_inject_k > 0:
                lex = self._tags.lexical_scores(query, len(self.paths))
            if self._lexical_inject_k > 0:
                inject_rows = self._tags.strong_match_rows(query)
        return self._rank(
            qv, k,
            min_ratio=min_ratio, min_score=min_score,
            filters=filters, lex_scores=lex, hybrid_weight=hw,
            inject_rows=inject_rows,
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

    @property
    def _group_map(self) -> dict[str, list[int]]:
        try:
            return self._group_map_cache  # type: ignore[attr-defined]
        except AttributeError:
            idx: dict[str, list[int]] = {}
            for i, gk in enumerate(self.groups):
                idx.setdefault(gk, []).append(i)
            # A row's discarded exact-duplicate copies may carry different
            # filenames; any group key those names produce is an alias for
            # the row, so it also joins those groups.
            alt_keys: dict[int, set[str]] = {}
            if self._group_strip_pattern:
                import sqlite3

                from .filenames import group_key as make_group_key

                try:
                    con = sqlite3.connect(str(self.index_dir / "pixgrep.sqlite"))
                    try:
                        dupes = con.execute(
                            "SELECT path, duplicate_of FROM duplicates "
                            "WHERE duplicate_of IS NOT NULL"
                        ).fetchall()
                    finally:
                        con.close()
                except sqlite3.OperationalError:
                    dupes = []
                for p, row in dupes:
                    if not 0 <= row < len(self.groups):
                        continue
                    alt = make_group_key(Path(p).name, self._group_strip_pattern)
                    if alt and alt != self.groups[row]:
                        bucket = idx.setdefault(alt, [])
                        if row not in bucket:
                            bucket.append(row)
                        alt_keys.setdefault(row, set()).add(alt)
            self._group_map_cache = idx
            self._row_alt_keys = alt_keys
            return idx

    def group_members(self, row: int) -> list[dict]:
        if not 0 <= row < len(self.paths):
            raise IndexError(f"row {row} out of range")
        gmap = self._group_map
        keys = {self.groups[row]} | getattr(self, "_row_alt_keys", {}).get(row, set())
        members: set[int] = {row}
        for gk in keys:
            members.update(gmap.get(gk, []))
        return [self._result(r, 0.0) for r in sorted(members)[:60]]

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
        inject_rows: np.ndarray | None = None,
    ) -> list[dict]:
        sims = self.emb @ qv.astype(np.float32)

        # Soft junk penalty: proportional demotion, not a binary cut. Runs
        # before any masking so it shapes ranking among survivors; the
        # binary junk_threshold mask (below) still applies after this.
        if self._junk_scores is not None and self._junk_soft_weight > 0:
            sims = sims - self._junk_soft_weight * np.clip(self._junk_scores, 0.0, None)

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

        # Over-fetch when near-dupe collapse is on, so dropped duplicates
        # still leave room to backfill up to k distinct results.
        pool_k = k
        if self._near_dupe_cos > 0:
            pool_k = min(n_valid, max(3 * k, k + 20))

        # Partition and sort by semantic score; convert to list immediately so
        # truth-value checks below always work regardless of cutoff conditions.
        top_idx = np.argpartition(-sims, kth=pool_k - 1)[:pool_k]
        top: list[int] = list(top_idx[np.argsort(-sims[top_idx])])

        # Relevance floors apply to SEMANTIC score only
        best = float(sims[top[0]])
        if min_score > 0:
            top = [i for i in top if float(sims[i]) >= min_score]
        if min_ratio > 0 and best > 0:
            top = [i for i in top if float(sims[i]) >= best * min_ratio]

        if not top:
            return []

        # Lexical injection: pull in rows a strong tag match identifies for
        # this query but that the semantic floors excluded above. They must
        # still be finite (not filter/junk masked, not excluded) and carry
        # real lexical support for THIS query, not just any tag match.
        if (
            self._lexical_inject_k > 0
            and inject_rows is not None
            and len(inject_rows) > 0
            and lex_scores is not None
        ):
            top_set = set(top)
            candidates = [
                int(i) for i in inject_rows
                if int(i) not in top_set and np.isfinite(sims[int(i)])
            ]
            lex_max = float(lex_scores.max()) if len(lex_scores) else 0.0
            if candidates and lex_max > 0:
                min_lex = 0.5 * lex_max
                candidates = [i for i in candidates if float(lex_scores[i]) >= min_lex]
                if candidates:
                    candidates.sort(
                        key=lambda i: -(float(sims[i]) + hybrid_weight * float(lex_scores[i]))
                    )
                    top = top + candidates[: self._lexical_inject_k]

        # Hybrid blend: re-rank survivors by semantic + w * lexical
        if lex_scores is not None and hybrid_weight > 0 and len(lex_scores) == len(self.paths):
            hybrid = {i: float(sims[i]) + hybrid_weight * float(lex_scores[i]) for i in top}
            top = sorted(top, key=lambda i: -hybrid[i])
            scores = hybrid
        else:
            scores = {i: float(sims[i]) for i in top}

        if self._near_dupe_cos > 0:
            top = self._collapse_near_dupes(top)
        top = top[:k]

        return [self._result(int(i), scores[i]) for i in top]

    def _collapse_near_dupes(self, ranked: list[int]) -> list[int]:
        """Drop lower-ranked rows that are near-identical (same group, cos >
        threshold) to an already-kept higher-ranked row."""
        kept: list[int] = []
        for i in ranked:
            gi = self.groups[i]
            ei = self.emb[i]
            if any(
                gi == self.groups[j] and float(ei @ self.emb[j]) > self._near_dupe_cos
                for j in kept
            ):
                continue
            kept.append(i)
        return kept

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
