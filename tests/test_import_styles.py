"""Tests for scripts/import_styles.py and merge mode in import_tags.

All data is synthetic — no real SKUs, style numbers, or company vocabulary.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixgrep.store import save_index
from pixgrep.tags import import_tags
from scripts.import_styles import (
    _decode_carat,
    _decode_category,
    _decode_consensus,
    _decode_design,
    _decode_metal_color,
    _decode_segment,
    import_styles_core,
)

# ---------------------------------------------------------------------------
# Synthetic decode maps (generic labels, no company vocab)
# ---------------------------------------------------------------------------

MAPS = {
    "category": {
        "RNG": "ring",
        "PND": "pendant",
        "EAR": "earring",
        "SET": "combo-set",
    },
    "metal_color": {
        "W": "white",
        "Y": "yellow",
        "TT": "two-tone",
        "TY": "two-tone",
        "YW": "two-tone",
        "WR": "two-tone",
        "R": "rose",
        "RS": "rose",
        "RW": "rose",
        "RR": "rose",
    },
    "metal_typ": {
        "14KT": "14k-gold",
        "10KT": "10k-gold",
        "SILV": "silver",
        "ALLO": "alloy",
    },
    "segment_map": {
        "WOM": "womens",
        "MEN": "mens",
    },
    "design_map": {
        "HOOP": "hoop",
        "BAND": "band",
        "STUD": "stud",
    },
}

STRIP_PATTERN = r"(?<=\d)[a-z]+\d*[a-z]*(?:[-_ ]?\d+)?$"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_index(tmp_path: Path, stems: list[str], group_keys: list[str]) -> None:
    paths = [f"folder/{s}.jpg" for s in stems]
    emb = np.eye(max(len(stems), 4), dtype=np.float32)[: len(stems)]
    save_index(tmp_path, paths, group_keys, emb)


def _erp_row(**kwargs) -> dict:
    """Build a minimal ERP-style row dict (generic column names)."""
    defaults = {
        "Style #": "",
        "Category": "",
        "Description": "",
        "Group 1": "",
        "Diamond Wt": "",
        "Metal Typ": "",
        "Metal Color": "",
        "Group 3": "",
        "Group 4": "",
        "Group 5": "",
        "Group 6": "",
        "Status": "Active",
    }
    defaults.update(kwargs)
    return defaults


def _tags(tmp_path: Path) -> list[tuple]:
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    rows = con.execute("SELECT row, field, value FROM tags ORDER BY row, field, value").fetchall()
    con.close()
    return rows


def _tag_text(tmp_path: Path, row: int) -> str | None:
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    r = con.execute("SELECT text FROM tag_text WHERE row=?", (row,)).fetchone()
    con.close()
    return r[0] if r else None


# ---------------------------------------------------------------------------
# Part 1 — merge mode in import_tags
# ---------------------------------------------------------------------------

def test_merge_preserves_other_rows_tags(tmp_path):
    _mk_index(tmp_path, ["itemA", "itemB"], ["itemA", "itemB"])
    # Set tags for both rows via fresh import
    import_tags(
        tmp_path,
        [
            {"fn": "itemA.jpg", "cat": "ring", "col": "white"},
            {"fn": "itemB.jpg", "cat": "pendant", "col": "yellow"},
        ],
        "fn",
        {"category": "cat", "color": "col"},
        [],
    )
    # Merge-import only itemA with a new color; itemB must survive unchanged
    import_tags(
        tmp_path,
        [{"fn": "itemA.jpg", "cat": "bracelet", "col": "rose"}],
        "fn",
        {"category": "cat", "color": "col"},
        [],
        merge=True,
    )
    rows = _tags(tmp_path)
    # itemA updated
    assert ("itemA.jpg" not in str(rows))  # sanity — we work with row ints
    row_map: dict[int, list[tuple]] = {}
    for r, f, v in rows:
        row_map.setdefault(r, []).append((f, v))
    # row 0 = itemA: should be bracelet / rose
    assert ("category", "bracelet") in row_map[0]
    assert ("color", "rose") in row_map[0]
    # row 1 = itemB: must still have pendant / yellow
    assert ("category", "pendant") in row_map[1]
    assert ("color", "yellow") in row_map[1]


def test_merge_replaces_same_field_value(tmp_path):
    _mk_index(tmp_path, ["item0"], ["item0"])
    import_tags(
        tmp_path,
        [{"fn": "item0.jpg", "cat": "ring"}],
        "fn",
        {"category": "cat"},
        [],
    )
    import_tags(
        tmp_path,
        [{"fn": "item0.jpg", "cat": "bracelet"}],
        "fn",
        {"category": "cat"},
        [],
        merge=True,
    )
    rows = _tags(tmp_path)
    values = [v for _, f, v in rows if f == "category"]
    # Only the new value; no duplicate
    assert values == ["bracelet"]


def test_merge_leaves_tables_when_empty_records(tmp_path):
    _mk_index(tmp_path, ["item0"], ["item0"])
    import_tags(
        tmp_path,
        [{"fn": "item0.jpg", "cat": "ring"}],
        "fn",
        {"category": "cat"},
        [],
    )
    # Merge with empty record list — tables must survive intact
    import_tags(tmp_path, [], "fn", {"category": "cat"}, [], merge=True)
    rows = _tags(tmp_path)
    assert any(f == "category" and v == "ring" for _, f, v in rows)


def test_merge_tag_text_replaced_when_incoming_has_text(tmp_path):
    _mk_index(tmp_path, ["item0"], ["item0"])
    import_tags(
        tmp_path,
        [{"fn": "item0.jpg", "cat": "ring", "note": "old note"}],
        "fn",
        {"category": "cat"},
        ["note"],
    )
    import_tags(
        tmp_path,
        [{"fn": "item0.jpg", "cat": "bracelet", "note": "new note"}],
        "fn",
        {"category": "cat"},
        ["note"],
        merge=True,
    )
    text = _tag_text(tmp_path, 0)
    assert text is not None
    assert "new note" in text
    assert "old note" not in text


def test_merge_tag_text_left_when_incoming_has_no_text(tmp_path):
    _mk_index(tmp_path, ["item0", "item1"], ["item0", "item1"])
    import_tags(
        tmp_path,
        [
            {"fn": "item0.jpg", "cat": "ring", "note": "keep me"},
            {"fn": "item1.jpg", "cat": "bangle", "note": ""},
        ],
        "fn",
        {"category": "cat"},
        ["note"],
    )
    # Merge for item1: truly no combined text (no text_keys, no field values)
    # item1 row gets an empty record with empty cat and empty note → no text → leave
    import_tags(
        tmp_path,
        [{"fn": "item1.jpg", "cat": "", "note": ""}],
        "fn",
        {"category": "cat"},
        ["note"],
        merge=True,
    )
    # item1 has no incoming text and no field values → existing tag_text must survive
    text1 = _tag_text(tmp_path, 1)
    # item1's original text was empty (note="") but category "bangle" was a field val
    # so tag_text = "bangle". Now incoming has nothing → tag_text stays as "bangle"
    assert text1 is not None and "bangle" in text1

    # item0 was not touched by the merge at all → its text also survives
    text0 = _tag_text(tmp_path, 0)
    assert text0 is not None
    assert "keep me" in text0


# ---------------------------------------------------------------------------
# Part 2 — decode helpers
# ---------------------------------------------------------------------------

def test_decode_category_known(tmp_path):
    assert _decode_category("RNG", MAPS["category"]) == "ring"
    assert _decode_category("PND", MAPS["category"]) == "pendant"


def test_decode_category_unknown_returns_none(tmp_path):
    assert _decode_category("GARBAGE", MAPS["category"]) is None
    assert _decode_category("", MAPS["category"]) is None


def test_decode_metal_white_yellow(tmp_path):
    assert _decode_metal_color("W", MAPS["metal_color"]) == "white"
    assert _decode_metal_color("Y", MAPS["metal_color"]) == "yellow"


def test_decode_metal_two_tone_family(tmp_path):
    for code in ("TT", "TY", "YW", "WR"):
        assert _decode_metal_color(code, MAPS["metal_color"]) == "two-tone", code


def test_decode_metal_rose_family(tmp_path):
    for code in ("R", "RS", "RW", "RR"):
        assert _decode_metal_color(code, MAPS["metal_color"]) == "rose", code


def test_decode_metal_unknown_returns_none(tmp_path):
    assert _decode_metal_color("NA", MAPS["metal_color"]) is None
    assert _decode_metal_color("", MAPS["metal_color"]) is None


def test_decode_design_known(tmp_path):
    assert _decode_design("HOOP", MAPS["design_map"]) == "hoop"
    assert _decode_design("BAND", MAPS["design_map"]) == "band"


def test_decode_design_unknown_kept_raw_lowercase(tmp_path):
    assert _decode_design("NOVELCODE", MAPS["design_map"]) == "novelcode"
    assert _decode_design("B&R", MAPS["design_map"]) == "b&r"


def test_decode_design_empty_returns_none(tmp_path):
    assert _decode_design("", MAPS["design_map"]) is None


def test_decode_segment_known(tmp_path):
    assert _decode_segment("WOM", MAPS["segment_map"]) == "womens"


def test_decode_segment_unknown_short_kept(tmp_path):
    # 5 chars → kept as raw lowercase
    assert _decode_segment("GENTS", MAPS["segment_map"]) == "gents"


def test_decode_segment_unknown_long_skipped(tmp_path):
    # >8 chars → skip
    assert _decode_segment("LONGCODEVALUE", MAPS["segment_map"]) is None


def test_decode_segment_empty_returns_none(tmp_path):
    assert _decode_segment("", MAPS["segment_map"]) is None


def test_carat_buckets(tmp_path):
    assert _decode_carat("0.1") == "under 1/2 ct"
    assert _decode_carat("0.39") == "under 1/2 ct"
    assert _decode_carat("0.4") == "1/2 ct"
    assert _decode_carat("0.5") == "1/2 ct"
    assert _decode_carat("0.74") == "1/2 ct"
    assert _decode_carat("0.75") == "1 ct"
    assert _decode_carat("1.0") == "1 ct"
    assert _decode_carat("1.24") == "1 ct"
    assert _decode_carat("1.25") == "2 ct"
    assert _decode_carat("2.0") == "2 ct"
    assert _decode_carat("2.24") == "2 ct"
    assert _decode_carat("2.25") == "2+ ct"
    assert _decode_carat("5.0") == "2+ ct"


def test_carat_zero_or_empty_returns_none(tmp_path):
    assert _decode_carat("0") is None
    assert _decode_carat("") is None
    assert _decode_carat("N/A") is None


# ---------------------------------------------------------------------------
# import_styles_core — integration
# ---------------------------------------------------------------------------

def test_exact_match_wins_over_base(tmp_path):
    """Exact SKU lookup is used when available; base records are ignored."""
    _mk_index(tmp_path, ["1000w", "1000y"], ["1000", "1000"])
    erp_rows = [
        _erp_row(**{"Style #": "1000W", "Metal Color": "W", "Category": "RNG"}),
        _erp_row(**{"Style #": "1000Y", "Metal Color": "Y", "Category": "RNG"}),
    ]
    result = import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN)
    assert result["matched_exact"] == 2
    assert result["matched_base"] == 0

    tags = _tags(tmp_path)
    row0_metal = [v for r, f, v in tags if r == 0 and f == "metal"]
    row1_metal = [v for r, f, v in tags if r == 1 and f == "metal"]
    assert row0_metal == ["white"]
    assert row1_metal == ["yellow"]


def test_base_match_used_when_no_exact(tmp_path):
    """A stripped-stem hit populates the row when no exact match exists."""
    _mk_index(tmp_path, ["2000wg100"], ["2000"])
    erp_rows = [
        _erp_row(**{"Style #": "2000", "Metal Color": "W", "Category": "PND"}),
    ]
    result = import_styles_core(tmp_path, MAPS, STRIP_PATTERN) if False else \
        import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN)
    assert result["matched_base"] == 1

    tags = _tags(tmp_path)
    assert any(f == "category" and v == "pendant" for _, f, v in tags)


def test_base_multi_disagree_metal_skipped(tmp_path):
    """Base match with multiple records that disagree on metal → no metal tag."""
    _mk_index(tmp_path, ["3000wg"], ["3000"])
    erp_rows = [
        _erp_row(**{"Style #": "3000", "Metal Color": "W", "Category": "RNG"}),
        _erp_row(**{"Style #": "3000", "Metal Color": "Y", "Category": "RNG"}),
    ]
    result = import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN)
    assert result["matched_base"] == 1

    tags = _tags(tmp_path)
    metal_tags = [v for _, f, v in tags if f == "metal"]
    assert metal_tags == []  # disagreement → no metal tag
    # category agreed → still tagged
    cat_tags = [v for _, f, v in tags if f == "category"]
    assert cat_tags == ["ring"]


def test_base_multi_agree_metal_included(tmp_path):
    """Base match where all records agree on metal → metal tag is written."""
    _mk_index(tmp_path, ["4000wg"], ["4000"])
    erp_rows = [
        _erp_row(**{"Style #": "4000", "Metal Color": "W", "Category": "RNG"}),
        _erp_row(**{"Style #": "4000", "Metal Color": "W", "Category": "RNG"}),
    ]
    result = import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN)
    tags = _tags(tmp_path)
    metal_tags = [v for _, f, v in tags if f == "metal"]
    assert metal_tags == ["white"]


def test_dry_run_no_writes(tmp_path):
    """dry_run=True reports counts but writes nothing to the DB."""
    _mk_index(tmp_path, ["5000wg"], ["5000"])
    erp_rows = [_erp_row(**{"Style #": "5000", "Metal Color": "W", "Category": "RNG"})]
    result = import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN, dry_run=True)
    assert result["matched_base"] + result["matched_exact"] > 0

    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    tables = {
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    metal_count = 0
    if "tags" in tables:
        metal_count = con.execute(
            "SELECT COUNT(*) FROM tags WHERE field='metal'"
        ).fetchone()[0]
    con.close()
    assert metal_count == 0


def test_unmatched_index_rows_counted(tmp_path):
    """Index rows with no ERP match are counted as unmatched."""
    _mk_index(tmp_path, ["9999wg"], ["9999"])
    erp_rows = [_erp_row(**{"Style #": "0001", "Metal Color": "W"})]
    result = import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN)
    assert result["unmatched"] == 1
    assert result["matched_exact"] + result["matched_base"] == 0


def test_import_styles_fresh_wipes_existing_tags(tmp_path):
    """fresh=True drops prior tags/tag_text entirely before rematching —
    unlike merge mode, rows from other sources do NOT survive."""
    _mk_index(tmp_path, ["6000wg", "7000rg"], ["6000", "7000"])
    import_tags(
        tmp_path,
        [{"fn": "7000rg.jpg", "gem": "diamond"}],
        "fn",
        {"gemstone": "gem"},
        [],
    )
    erp_rows = [_erp_row(**{"Style #": "6000w", "Metal Color": "W", "Category": "RNG"})]
    import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN, fresh=True)

    tags = _tags(tmp_path)
    # row 1's pre-existing gemstone tag from another source is gone
    assert not any(r == 1 and f == "gemstone" for r, f, _ in tags)
    # row 0 metal is freshly written
    assert any(r == 0 and f == "metal" for r, f, _ in tags)


def test_import_styles_fresh_dry_run_does_not_drop(tmp_path):
    """fresh + dry_run must not touch the DB at all."""
    _mk_index(tmp_path, ["7000rg"], ["7000"])
    import_tags(
        tmp_path,
        [{"fn": "7000rg.jpg", "gem": "diamond"}],
        "fn",
        {"gemstone": "gem"},
        [],
    )
    erp_rows = [_erp_row(**{"Style #": "7000r", "Metal Color": "R", "Category": "RNG"})]
    import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN, dry_run=True, fresh=True)

    tags = _tags(tmp_path)
    assert any(r == 0 and f == "gemstone" and v == "diamond" for r, f, v in tags)


def test_import_styles_is_merge(tmp_path):
    """import_styles_core operates in merge mode — pre-existing tags from other
    sources survive for rows it does not match."""
    _mk_index(tmp_path, ["6000wg", "7000rg"], ["6000", "7000"])
    # Pre-populate tags for row 1 (7000rg) via import_tags
    import_tags(
        tmp_path,
        [{"fn": "7000rg.jpg", "gem": "diamond"}],
        "fn",
        {"gemstone": "gem"},
        [],
    )
    # ERP only covers row 0 (6000wg)
    erp_rows = [_erp_row(**{"Style #": "6000w", "Metal Color": "W", "Category": "RNG"})]
    import_styles_core(tmp_path, erp_rows, MAPS, STRIP_PATTERN)

    tags = _tags(tmp_path)
    # row 1 gemstone must survive
    assert any(r == 1 and f == "gemstone" for r, f, _ in tags)
    # row 0 metal must be written
    assert any(r == 0 and f == "metal" for r, f, _ in tags)
