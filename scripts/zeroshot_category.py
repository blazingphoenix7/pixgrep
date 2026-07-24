"""Zero-shot category backfill for index rows with no ERP category tag.

Scores every indexed row against a fixed set of category text prompts using
the configured text encoder, then VALIDATES the zero-shot predictions against
rows that already carry an ERP "category" tag (predicted vs actual). Reports
overall agreement, top confusions, and accuracy/coverage at a sweep of
top1-top2 margin thresholds, and recommends the smallest margin that reaches
>=95% agreement.

Report-only by default. Pass --apply to write "category" tags (merge
semantics, via pixgrep.tags.import_tags) for rows that currently have none —
restricted to the recommended (or --margin-overridden) confidence threshold,
and only for predicted labels that already exist verbatim in the tag
vocabulary (a prediction like "watch" or "gemstone", which have no existing
category tag value, is never written).

Usage:
    python scripts/zeroshot_category.py --config config.local.json
    python scripts/zeroshot_category.py --config config.local.json --apply --margin 0.03
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pixgrep.config import load_config
from pixgrep.store import load_index
from pixgrep.tags import import_tags

# Canonical zero-shot label -> text prompt fed to the text encoder.
CATEGORY_PROMPTS: dict[str, str] = {
    "ring": "a photo of a ring",
    "earring": "a photo of a pair of earrings",
    "pendant": "a photo of a pendant necklace",
    "bracelet": "a photo of a bracelet",
    "necklace": "a photo of a necklace",
    "bangle": "a photo of a bangle",
    "watch": "a photo of a wrist watch",
    "brooch": "a photo of a brooch",
    "gemstone": "a photo of a loose gemstone",
    "chain": "a photo of a chain",
}

# Canonical label -> existing "category" tag values counted as a correct
# match during validation. Catalog values with no single-object equivalent
# (e.g. multi-piece sets) are left out on purpose and excluded from scoring.
# Every value on the right already exists verbatim in the tag vocabulary, so
# writing the canonical label back for --apply never introduces a new value.
CATEGORY_TAG_ALIASES: dict[str, set[str]] = {
    "ring": {"ring", "fashion ring", "gents ring's"},
    "earring": {"earring"},
    "pendant": {"pendant"},
    "bracelet": {"bracelet"},
    "necklace": {"necklace"},
    "bangle": {"bangle"},
    "brooch": {"brooch"},
    "chain": {"chain"},
}

MARGIN_THRESHOLDS: tuple[float, ...] = (0.00, 0.01, 0.02, 0.03, 0.05)
MIN_ACCURACY = 0.95


def tag_value_to_canonical() -> dict[str, str]:
    """Reverse CATEGORY_TAG_ALIASES: raw tag value -> canonical zero-shot label."""
    out: dict[str, str] = {}
    for canonical, values in CATEGORY_TAG_ALIASES.items():
        for v in values:
            out[v] = canonical
    return out


def embed_prompts(embedder, labels: list[str]) -> np.ndarray:
    """Embed CATEGORY_PROMPTS[label] for each label, L2-normalized (labels, D)."""
    prompts = [CATEGORY_PROMPTS[label] for label in labels]
    vecs = np.array(embedder.embed_texts(prompts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def score_rows(
    emb: np.ndarray, text_vecs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine-score every row against every prompt vector.

    emb: (N, D) image embeddings. text_vecs: (C, D) prompt embeddings.
    Neither needs to be pre-normalized here; both are normalized defensively.

    Returns (pred_idx, top1_score, margin), each shape (N,). margin is the
    top1-top2 cosine gap (confidence); with a single category margin is
    top1 itself.
    """
    e = emb.astype(np.float32)
    e = e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-8, None)
    t = text_vecs.astype(np.float32)
    t = t / np.linalg.norm(t, axis=1, keepdims=True).clip(1e-8, None)

    sims = e @ t.T  # (N, C)
    order = np.argsort(-sims, axis=1)
    pred_idx = order[:, 0]
    top1 = np.take_along_axis(sims, order[:, 0:1], axis=1)[:, 0]
    if sims.shape[1] > 1:
        top2 = np.take_along_axis(sims, order[:, 1:2], axis=1)[:, 0]
        margin = top1 - top2
    else:
        margin = top1.copy()
    return pred_idx, top1, margin


def sweep_margin_thresholds(
    pred: list[str],
    true: list[str],
    margin: np.ndarray,
    thresholds: tuple[float, ...] = MARGIN_THRESHOLDS,
) -> list[dict]:
    """Accuracy/coverage of a validation set at each margin threshold.

    pred/true/margin must all be aligned to the same validation subset
    (rows with a known, mapped ground-truth category). Coverage is relative
    to the full validation subset (len(true)); accuracy is computed only
    over rows surviving the threshold.
    """
    n_total = len(true)
    pred_arr = np.array(pred)
    true_arr = np.array(true)
    correct = pred_arr == true_arr

    out = []
    for t in thresholds:
        mask = margin >= t
        n = int(mask.sum())
        coverage = n / n_total if n_total else 0.0
        accuracy = float(correct[mask].mean()) if n else 0.0
        out.append(
            {"threshold": t, "n": n, "coverage": coverage, "accuracy": accuracy}
        )
    return out


def recommend_threshold(
    sweep: list[dict], min_accuracy: float = MIN_ACCURACY
) -> float | None:
    """Smallest (most-coverage) threshold in `sweep` reaching min_accuracy.

    Assumes `sweep` is ordered by ascending threshold. Returns None if no
    threshold in the sweep reaches min_accuracy.
    """
    for row in sweep:
        if row["accuracy"] >= min_accuracy:
            return row["threshold"]
    return None


def top_confusions(
    pred: list[str], true: list[str], top_n: int = 10
) -> list[tuple[tuple[str, str], int]]:
    """Most common (true, predicted) mismatch pairs, most frequent first."""
    counts: Counter = Counter()
    for p, t in zip(pred, true):
        if p != t:
            counts[(t, p)] += 1
    return counts.most_common(top_n)


def load_category_tags(index_dir: Path) -> dict[int, str]:
    """Return {row: tag_value} for every row with an existing 'category' tag."""
    con = sqlite3.connect(str(Path(index_dir) / "pixgrep.sqlite"))
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "tags" not in tables:
            return {}
        rows = con.execute(
            "SELECT row, value FROM tags WHERE field='category'"
        ).fetchall()
        return {r: v for r, v in rows}
    finally:
        con.close()


def build_report(
    labels: list[str],
    pred_canonical: list[str],
    margin: np.ndarray,
    category_tags: dict[int, str],
) -> dict:
    """Assemble the full validation report as plain data (no printing).

    Returns a dict with: n_total, n_with_tag, n_mapped, n_unmapped,
    unmapped_counts, overall_agreement, confusions, sweep, recommended,
    untagged_rows (row indices without a category tag).
    """
    n_total = len(pred_canonical)
    tag_to_canon = tag_value_to_canonical()

    mapped_rows: list[int] = []
    true_canonical: list[str] = []
    unmapped_counts: Counter = Counter()

    for row, raw_value in category_tags.items():
        canon = tag_to_canon.get(raw_value)
        if canon is None:
            unmapped_counts[raw_value] += 1
            continue
        mapped_rows.append(row)
        true_canonical.append(canon)

    pred_for_mapped = [pred_canonical[r] for r in mapped_rows]
    margin_for_mapped = margin[mapped_rows] if mapped_rows else np.array([])

    sweep = sweep_margin_thresholds(pred_for_mapped, true_canonical, margin_for_mapped)
    recommended = recommend_threshold(sweep)
    overall_agreement = sweep[0]["accuracy"] if sweep else 0.0

    all_rows = set(range(n_total))
    tagged_rows = set(category_tags.keys())
    untagged_rows = sorted(all_rows - tagged_rows)

    return {
        "n_total": n_total,
        "n_with_tag": len(category_tags),
        "n_mapped": len(mapped_rows),
        "n_unmapped": sum(unmapped_counts.values()),
        "unmapped_counts": unmapped_counts,
        "overall_agreement": overall_agreement,
        "confusions": top_confusions(pred_for_mapped, true_canonical),
        "sweep": sweep,
        "recommended": recommended,
        "untagged_rows": untagged_rows,
    }


def untagged_coverage_at_threshold(
    untagged_rows: list[int],
    pred_canonical: list[str],
    margin: np.ndarray,
    threshold: float,
) -> dict:
    """How many untagged rows would receive a category at `threshold`.

    Splits by whether the predicted label already exists in the tag
    vocabulary (writable by --apply) or not (would be skipped).
    """
    writable_labels = set(CATEGORY_TAG_ALIASES.keys())
    writable = 0
    skipped_non_vocab = 0
    for row in untagged_rows:
        if margin[row] < threshold:
            continue
        if pred_canonical[row] in writable_labels:
            writable += 1
        else:
            skipped_non_vocab += 1
    return {
        "n_untagged": len(untagged_rows),
        "writable": writable,
        "skipped_non_vocab": skipped_non_vocab,
    }


def apply_predictions(
    index_dir: Path,
    paths: list[str],
    untagged_rows: list[int],
    pred_canonical: list[str],
    margin: np.ndarray,
    threshold: float,
) -> dict:
    """Write 'category' tags for untagged rows at/above `threshold`.

    Only writes predictions whose canonical label already exists in the tag
    vocabulary (CATEGORY_TAG_ALIASES keys) — never introduces a new value.
    Uses pixgrep.tags.import_tags(merge=True) so the write path matches the
    codebase's existing merge semantics.
    """
    writable_labels = set(CATEGORY_TAG_ALIASES.keys())
    records = []
    for row in untagged_rows:
        if margin[row] < threshold:
            continue
        label = pred_canonical[row]
        if label not in writable_labels:
            continue
        records.append({"file": Path(paths[row]).name, "category": label})

    report = import_tags(
        index_dir,
        records,
        filename_key="file",
        field_keys={"category": "category"},
        text_keys=[],
        merge=True,
    )
    return {"attempted": len(records), "report": report}


def _print_report(report: dict) -> None:
    print(f"Total index rows:            {report['n_total']}")
    print(f"Rows with existing category: {report['n_with_tag']}")
    print(f"  usable for validation:     {report['n_mapped']}")
    print(f"  excluded (unmapped value): {report['n_unmapped']}")
    if report["unmapped_counts"]:
        for value, count in report["unmapped_counts"].most_common():
            print(f"    {value!r}: {count}")
    print(f"Rows with no category tag:   {len(report['untagged_rows'])}")

    print(f"\nOverall agreement (argmax, no margin filter): {report['overall_agreement']:.4f}")

    print("\nTop confusions (true -> predicted):")
    for (true_label, pred_label), count in report["confusions"]:
        print(f"  {true_label:>10} -> {pred_label:<10}  {count}")

    print("\nMargin sweep (top1-top2 confidence):")
    print(f"  {'margin':>7} {'n':>7} {'coverage':>9} {'accuracy':>9}")
    for row in report["sweep"]:
        print(
            f"  {row['threshold']:>7.2f} {row['n']:>7d} "
            f"{row['coverage']:>9.1%} {row['accuracy']:>9.4%}"
        )

    if report["recommended"] is None:
        print(f"\nNo margin threshold in the sweep reaches {MIN_ACCURACY:.0%} agreement.")
    else:
        print(f"\nRecommended margin threshold: {report['recommended']:.2f} (>= {MIN_ACCURACY:.0%} agreement)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot category backfill for untagged index rows"
    )
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write category tags for untagged rows (default: report only).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Margin threshold to use for --apply (default: the recommended threshold).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loading index from {cfg.index_dir} ...")
    paths, _groups, emb = load_index(cfg.index_dir)
    n = len(paths)
    print(f"  {n} rows loaded.")

    labels = list(CATEGORY_PROMPTS.keys())
    print("Loading text embedder ...")
    embedder = cfg.make_embedder()
    text_vecs = embed_prompts(embedder, labels)

    print("Scoring rows against category prompts ...")
    pred_idx, _top1, margin = score_rows(emb, text_vecs)
    pred_canonical = [labels[i] for i in pred_idx]

    category_tags = load_category_tags(cfg.index_dir)
    report = build_report(labels, pred_canonical, margin, category_tags)

    print()
    _print_report(report)

    threshold = report["recommended"] if report["recommended"] is not None else max(
        MARGIN_THRESHOLDS
    )
    coverage = untagged_coverage_at_threshold(
        report["untagged_rows"], pred_canonical, margin, threshold
    )
    print(
        f"\nAt margin >= {threshold:.2f}: {coverage['writable']}/{coverage['n_untagged']} "
        f"untagged rows would receive a (vocab-valid) category "
        f"({coverage['skipped_non_vocab']} more predicted a non-vocab label and would be skipped)."
    )

    if not args.apply:
        print("\nReport-only run (pass --apply to write tags). No changes made.")
        return

    apply_threshold = args.margin if args.margin is not None else threshold
    print(f"\nApplying at margin >= {apply_threshold:.2f} ...")
    result = apply_predictions(
        cfg.index_dir, paths, report["untagged_rows"], pred_canonical, margin, apply_threshold
    )
    print(f"Attempted: {result['attempted']}")
    print(result["report"])


if __name__ == "__main__":
    main()
