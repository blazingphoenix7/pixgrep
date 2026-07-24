"""Recompute group_key for every images row using the fixed pixgrep.filenames
logic (annotation-tail pre-strip) + the configured group_strip_pattern.

Run this once after the group_key fix lands, before rebuilding the tag stack
(import_styles.py / derive_filename_tags.py both read group_key from the DB).

Usage:
    python scripts/regroup_index.py --config config.local.json [--dry-run]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixgrep.config import load_config
from pixgrep.filenames import group_key


def regroup_index(index_dir: Path, strip_pattern: str, dry_run: bool = False) -> dict:
    """Recompute group_key for every row from its basename.

    Returns counts: total, changed.
    """
    db_path = Path(index_dir) / "pixgrep.sqlite"
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT row, path, group_key FROM images ORDER BY row").fetchall()

        updates: list[tuple[str, int]] = []
        for row, path, old_gk in rows:
            new_gk = group_key(Path(path).name, strip_pattern)
            if new_gk != (old_gk or ""):
                updates.append((new_gk, row))

        if not dry_run and updates:
            con.executemany("UPDATE images SET group_key=? WHERE row=?", updates)
            con.commit()

        return {"total": len(rows), "changed": len(updates)}
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute group_key for every index row"
    )
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print counts only, no DB writes"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    counts = regroup_index(cfg.index_dir, cfg.group_strip_pattern, dry_run=args.dry_run)

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Total rows:   {counts['total']}")
    print(f"{prefix}Changed rows: {counts['changed']}")


if __name__ == "__main__":
    main()
