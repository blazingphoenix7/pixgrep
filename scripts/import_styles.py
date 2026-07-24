"""Import ERP style data into the pixgrep index.

Maps CSV columns to canonical tag fields via _local/styles_mapping.json.
Runs in merge mode — existing tables survive; matched rows have their
ERP-sourced fields replaced.

Usage:
    python scripts/import_styles.py --csv all-styles.csv \\
        --config _local/config.fullindex.json [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixgrep.config import load_config


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def _decode_category(code: str, cat_map: dict) -> str | None:
    return cat_map.get(code.strip().upper())


def _decode_metal_color(code: str, metal_color_map: dict) -> str | None:
    return metal_color_map.get(code.strip().upper())


def _decode_material(code: str, metal_typ_map: dict) -> str | None:
    return metal_typ_map.get(code.strip().upper())


def _decode_segment(code: str, seg_map: dict) -> str | None:
    code = code.strip()
    if not code:
        return None
    mapped = seg_map.get(code.upper())
    if mapped:
        return mapped
    if len(code) <= 8:
        return code.lower()
    return None


def _decode_design(code: str, design_map: dict) -> str | None:
    code = code.strip()
    if not code:
        return None
    mapped = design_map.get(code.upper())
    if mapped:
        return mapped
    return code.lower()


def _decode_carat(dw_str: str) -> str | None:
    try:
        dw = float(dw_str.strip())
    except (ValueError, AttributeError):
        return None
    if dw <= 0:
        return None
    if dw < 0.4:
        return "under 1/2 ct"
    if dw < 0.75:
        return "1/2 ct"
    if dw < 1.25:
        return "1 ct"
    if dw < 2.25:
        return "2 ct"
    return "2+ ct"


def _decode_record(raw: dict, maps: dict) -> dict:
    """Decode one raw ERP record into canonical field → value pairs (None = skip)."""
    result: dict[str, str] = {}

    cat = _decode_category(raw.get("Category", ""), maps["category"])
    if cat:
        result["category"] = cat

    metal = _decode_metal_color(raw.get("Metal Color", ""), maps["metal_color"])
    if metal:
        result["metal"] = metal

    material = _decode_material(raw.get("Metal Typ", ""), maps["metal_typ"])
    if material:
        result["material"] = material

    seg = _decode_segment(raw.get("Group 1", ""), maps["segment_map"])
    if seg:
        result["segment"] = seg

    design = _decode_design(raw.get("Group 3", ""), maps["design_map"])
    if design:
        result["design"] = design

    coll = raw.get("Group 5", "").strip().lower()
    if coll:
        result["collection"] = coll

    motif = raw.get("Group 4", "").strip().lower()
    if motif:
        result["motif"] = motif

    if raw.get("Group 6", "").strip().upper() == "LGD":
        result["lab_grown"] = "yes"

    status = raw.get("Status", "").strip()
    if status == "Delete":
        result["status"] = "discontinued"
    elif status == "Hold":
        result["status"] = "hold"

    carat = _decode_carat(raw.get("Diamond Wt", ""))
    if carat:
        result["carat"] = carat

    return result


def _decode_consensus(candidates: list[dict], maps: dict) -> dict:
    """For a base match with multiple records, include only fields where all agree."""
    decoded_list = [_decode_record(c, maps) for c in candidates]

    all_fields: set[str] = set()
    for d in decoded_list:
        all_fields.update(d.keys())

    result: dict[str, str] = {}
    for field in all_fields:
        values = [d.get(field) for d in decoded_list]
        non_none = [v for v in values if v is not None]
        if non_none and len(set(non_none)) == 1:
            result[field] = non_none[0]
    return result


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_erp_csv(csv_path: Path) -> list[dict]:
    with open(str(csv_path), newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Core import
# ---------------------------------------------------------------------------

_FIRST_TOKEN_RE = re.compile(r"[^\s#]+")


def import_styles_core(
    index_dir: Path,
    erp_rows: list[dict],
    maps: dict,
    strip_pattern: str,
    dry_run: bool = False,
) -> dict:
    """Match ERP rows to index images and write tags in merge mode.

    Returns a dict with match/miss counts and per-field tag counts.
    """
    db_path = Path(index_dir) / "pixgrep.sqlite"
    con = sqlite3.connect(str(db_path))
    try:
        # Build ERP lookups -------------------------------------------------
        full_sku: dict[str, dict] = {}   # lowercased Style# → record (last wins)
        base_lkp: dict[str, list[dict]] = {}  # stripped key → list of records

        for rec in erp_rows:
            style_raw = rec.get("Style #", "").strip()
            if not style_raw:
                continue
            sku = style_raw.lower()
            full_sku[sku] = rec
            base = re.sub(strip_pattern, "", sku)
            base_lkp.setdefault(base, []).append(rec)

        # Load index rows ---------------------------------------------------
        db_rows = con.execute(
            "SELECT row, path, group_key FROM images ORDER BY row"
        ).fetchall()

        matched_exact = 0
        matched_base = 0
        unmatched = 0
        field_counts: dict[str, int] = {}

        tag_inserts: list[tuple[int, str, str]] = []
        text_inserts: list[tuple[int, str]] = []
        delete_pairs: list[tuple[int, str]] = []

        for db_row, path, group_key in db_rows:
            stem = Path(path).stem
            m = _FIRST_TOKEN_RE.match(stem)
            first_token = m.group(0) if m else stem
            exact_key = first_token.lower()

            # 1. Try exact full-SKU match
            if exact_key in full_sku:
                matched_exact += 1
                decoded = _decode_record(full_sku[exact_key], maps)
                desc = full_sku[exact_key].get("Description", "").strip().lower()
            else:
                # 2. Try base lookup via group_key, then stripped stem
                stripped_stem = re.sub(strip_pattern, "", exact_key)
                candidates = None
                for key in (group_key, stripped_stem):
                    if key and key in base_lkp:
                        candidates = base_lkp[key]
                        break

                if candidates is None:
                    unmatched += 1
                    continue

                matched_base += 1
                if len(candidates) == 1:
                    decoded = _decode_record(candidates[0], maps)
                    desc = candidates[0].get("Description", "").strip().lower()
                else:
                    decoded = _decode_consensus(candidates, maps)
                    desc = ""

            # Build tag and text rows ---------------------------------------
            for field, value in decoded.items():
                tag_inserts.append((db_row, field, value))
                delete_pairs.append((db_row, field))
                field_counts[field] = field_counts.get(field, 0) + 1

            text_parts = [desc] if desc else []
            text_parts.extend(decoded.values())
            combined = " ".join(p for p in text_parts if p)
            if combined:
                text_inserts.append((db_row, combined))

        # Write to DB -------------------------------------------------------
        if not dry_run:
            con.execute(
                "CREATE TABLE IF NOT EXISTS tags "
                "(row INTEGER, field TEXT, value TEXT)"
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_tags_row ON tags(row)")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_tags_field_value "
                "ON tags(field, value)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS tag_text "
                "(row INTEGER PRIMARY KEY, text TEXT)"
            )
            for pair in delete_pairs:
                con.execute(
                    "DELETE FROM tags WHERE row=? AND field=?", pair
                )
            con.executemany(
                "INSERT INTO tags (row, field, value) VALUES (?, ?, ?)",
                tag_inserts,
            )
            for db_row, text in text_inserts:
                con.execute(
                    "INSERT OR REPLACE INTO tag_text (row, text) VALUES (?, ?)",
                    (db_row, text),
                )
            con.commit()

        return {
            "matched_exact": matched_exact,
            "matched_base": matched_base,
            "unmatched": unmatched,
            "field_counts": field_counts,
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import ERP style data into pixgrep index"
    )
    parser.add_argument("--csv", required=True, help="ERP CSV file (all-styles.csv)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config JSON (default: config.local.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count matches without writing to DB.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    # Load company decode maps (gitignored, kept in _local/)
    maps_path = Path(__file__).resolve().parent.parent / "_local" / "styles_mapping.json"
    maps = json.loads(maps_path.read_text(encoding="utf-8"))

    erp_rows = _read_erp_csv(Path(args.csv))
    print(f"Loaded {len(erp_rows)} ERP rows from {Path(args.csv).name}")

    result = import_styles_core(
        index_dir=cfg.index_dir,
        erp_rows=erp_rows,
        maps=maps,
        strip_pattern=cfg.group_strip_pattern,
        dry_run=args.dry_run,
    )

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Matched exact:       {result['matched_exact']}")
    print(f"{prefix}Matched base:        {result['matched_base']}")
    print(f"{prefix}Unmatched index rows:{result['unmatched']}")
    print(f"{prefix}Per-field tag counts:")
    for field, count in sorted(result["field_counts"].items()):
        print(f"  {field:20s}: {count}")


if __name__ == "__main__":
    main()
