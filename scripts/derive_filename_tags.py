"""Derive metal tags from SKU filenames for rows without spreadsheet metal tags.

Metal is encoded in the suffix letters immediately after the base style number.
Rule (applied to the first letter of the suffix, uppercase):
  - Starts with T (T, TT, TY, TW, TR, ...) → two-tone
  - First letter W → white
  - First letter Y → yellow
  - First letter R → rose
  - Anything else  → skip (conservative)

The two-char combos WT/YT/WY/YW used to also fire two-tone, but the metal
audit (private/eval/results/metal_audit.json) found they over-fire on solid
one-tone pieces (e.g. "R55921WTY.JPG" is solid yellow, not two-tone) — only
T-leading tokens are trustworthy.

Some product lines (see ``skip_metal_prefixes`` in _local/styles_mapping.json)
use a trailing letter to encode item TYPE (Ring/Earring/Pendant), not metal —
metal derivation is skipped entirely for stems starting with those prefixes.

Usage:
    python scripts/derive_filename_tags.py --config config.local.json [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixgrep.config import load_config

_STYLES_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent / "_local" / "styles_mapping.json"
)


def _load_skip_prefixes(mapping_path: Path = _STYLES_MAPPING_PATH) -> list[str]:
    """Load product-line prefixes to skip metal derivation for.

    Returns an empty list if the mapping file or key is absent.
    """
    if not mapping_path.exists():
        return []
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    return list(data.get("skip_metal_prefixes", []))


def _first_alnum_run(stem: str) -> str:
    """Return first contiguous alphanumeric run (stop at space / # / -)."""
    m = re.match(r"[a-zA-Z0-9]+", stem)
    return m.group(0) if m else stem


def _derive_metal(suffix: str) -> str | None:
    """Return metal value from suffix first letter, or None to skip."""
    s = suffix.upper()
    if not s:
        return None
    first = s[0]

    # Two-tone: only trust a leading T (T, TT, TY, TW, TR, ...). The two-char
    # combos WT/YT/WY/YW over-fire on solid one-tone pieces (see audit).
    if first == "T":
        return "two-tone"
    if first == "W":
        return "white"
    if first == "Y":
        return "yellow"
    if first == "R":
        return "rose"
    return None


def derive_filename_tags(
    index_dir: Path,
    dry_run: bool = False,
    skip_prefixes: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Derive and insert metal tags from filenames.

    Reads group_key from the images table (already stripped by the indexer).
    Only processes rows without an existing metal tag. Rows whose SKU starts
    with one of ``skip_prefixes`` (case-insensitive; product lines that use a
    trailing letter for item type, not metal) are skipped entirely.
    Does NOT drop or recreate tags / tag_text tables.

    Returns counts: derived, skipped_ambiguous, skipped_line_prefix, already_tagged.
    """
    skip_prefixes_lower = tuple(p.lower() for p in (skip_prefixes or ()))
    db_path = Path(index_dir) / "pixgrep.sqlite"
    con = sqlite3.connect(str(db_path))
    try:
        db_rows = con.execute(
            "SELECT row, path, group_key FROM images ORDER BY row"
        ).fetchall()

        # Rows that already have a metal tag
        try:
            metal_rows = {
                r[0]
                for r in con.execute(
                    "SELECT row FROM tags WHERE field='metal'"
                ).fetchall()
            }
        except Exception:
            metal_rows = set()

        # Existing tag_text for append
        try:
            text_rows = {
                r[0]: r[1]
                for r in con.execute("SELECT row, text FROM tag_text").fetchall()
            }
        except Exception:
            text_rows = {}

        derived = 0
        skipped_ambiguous = 0
        skipped_line_prefix = 0
        already_tagged = 0
        insert_tags: list[tuple] = []
        upsert_text: list[tuple] = []

        for db_row, path, group_key in db_rows:
            if db_row in metal_rows:
                already_tagged += 1
                continue

            stem = Path(path).stem
            clean = _first_alnum_run(stem)

            if skip_prefixes_lower and clean.lower().startswith(skip_prefixes_lower):
                skipped_line_prefix += 1
                continue

            gk = (group_key or "").lower()
            if gk and clean.lower().startswith(gk):
                suffix = clean[len(gk):]
            else:
                suffix = clean

            metal = _derive_metal(suffix)
            if metal is None:
                skipped_ambiguous += 1
                continue

            derived += 1
            if not dry_run:
                insert_tags.append((db_row, "metal", metal))
                existing = text_rows.get(db_row, "")
                new_text = (existing + " " + metal).strip() if existing else metal
                upsert_text.append((db_row, new_text))

        if not dry_run and insert_tags:
            con.execute(
                "CREATE TABLE IF NOT EXISTS tags (row INTEGER, field TEXT, value TEXT)"
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_tags_row ON tags(row)")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_tags_field_value ON tags(field, value)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS tag_text (row INTEGER PRIMARY KEY, text TEXT)"
            )
            con.executemany(
                "INSERT INTO tags (row, field, value) VALUES (?, ?, ?)", insert_tags
            )
            for db_row, text in upsert_text:
                con.execute(
                    "INSERT INTO tag_text (row, text) VALUES (?, ?)"
                    " ON CONFLICT(row) DO UPDATE SET text=excluded.text",
                    (db_row, text),
                )
            con.commit()

        return {
            "derived": derived,
            "skipped_ambiguous": skipped_ambiguous,
            "skipped_line_prefix": skipped_line_prefix,
            "already_tagged": already_tagged,
        }
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive metal tags from filenames")
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print counts only, no DB writes"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    skip_prefixes = _load_skip_prefixes()
    counts = derive_filename_tags(
        cfg.index_dir, dry_run=args.dry_run, skip_prefixes=skip_prefixes
    )

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Skip prefixes:       {skip_prefixes or '(none)'}")
    print(f"{prefix}Derived:             {counts['derived']}")
    print(f"{prefix}Skipped/ambiguous:   {counts['skipped_ambiguous']}")
    print(f"{prefix}Skipped/line-prefix: {counts['skipped_line_prefix']}")
    print(f"{prefix}Already tagged:      {counts['already_tagged']}")


if __name__ == "__main__":
    main()
