"""Tests for scripts/derive_filename_tags.py — derivation rules and merge logic."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixgrep.store import save_index
from pixgrep.tags import import_tags
from scripts.derive_filename_tags import (
    _derive_metal,
    _first_alnum_run,
    derive_filename_tags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_index(
    tmp_path, filenames: list[str], group_keys: list[str]
) -> None:
    n = len(filenames)
    paths = [f"folder/{fn}" for fn in filenames]
    emb = np.eye(max(n, 4), dtype=np.float32)[:n]
    save_index(tmp_path, paths, group_keys, emb)


# ---------------------------------------------------------------------------
# Unit: _first_alnum_run
# ---------------------------------------------------------------------------


def test_alnum_run_plain():
    assert _first_alnum_run("1234YW") == "1234YW"


def test_alnum_run_space_separator():
    assert _first_alnum_run("1234YW #front") == "1234YW"


def test_alnum_run_hash_separator():
    assert _first_alnum_run("1234WG#2") == "1234WG"


def test_alnum_run_dash_separator():
    assert _first_alnum_run("1234RG-alt") == "1234RG"


def test_alnum_run_empty():
    assert _first_alnum_run("") == ""


# ---------------------------------------------------------------------------
# Unit: _derive_metal
# ---------------------------------------------------------------------------


def test_derive_white():
    assert _derive_metal("WG100") == "white"


def test_derive_yellow():
    assert _derive_metal("YG100") == "yellow"


def test_derive_rose():
    assert _derive_metal("RG100") == "rose"


def test_derive_two_tone_T_prefix():
    assert _derive_metal("TT100") == "two-tone"


def test_derive_two_tone_TY():
    assert _derive_metal("TY100") == "two-tone"


def test_derive_two_tone_TW():
    assert _derive_metal("TW100") == "two-tone"


def test_derive_two_tone_WT():
    assert _derive_metal("WT100") == "two-tone"


def test_derive_two_tone_YT():
    assert _derive_metal("YT100") == "two-tone"


def test_derive_two_tone_WY():
    assert _derive_metal("WY100") == "two-tone"


def test_derive_two_tone_YW():
    assert _derive_metal("YW100") == "two-tone"


def test_derive_skip_unrecognised():
    assert _derive_metal("GG100") is None


def test_derive_skip_empty():
    assert _derive_metal("") is None


def test_derive_case_insensitive():
    assert _derive_metal("wg") == "white"
    assert _derive_metal("yg") == "yellow"
    assert _derive_metal("rg") == "rose"
    assert _derive_metal("tw") == "two-tone"
    assert _derive_metal("yw") == "two-tone"


def test_derive_single_char_W():
    assert _derive_metal("W") == "white"


def test_derive_single_char_Y():
    assert _derive_metal("Y") == "yellow"


def test_derive_single_char_R():
    assert _derive_metal("R") == "rose"


def test_derive_single_char_T():
    assert _derive_metal("T") == "two-tone"


# ---------------------------------------------------------------------------
# Integration: derive_filename_tags
# ---------------------------------------------------------------------------


def test_derive_inserts_metal_tag(tmp_path):
    _build_index(tmp_path, ["1234WG.jpg"], ["1234"])
    counts = derive_filename_tags(tmp_path)
    assert counts["derived"] == 1
    assert counts["already_tagged"] == 0
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    rows = con.execute("SELECT field, value FROM tags WHERE row=0").fetchall()
    con.close()
    assert ("metal", "white") in rows


def test_derive_skips_already_tagged(tmp_path):
    _build_index(tmp_path, ["1234WG.jpg"], ["1234"])
    import_tags(
        tmp_path,
        [{"fn": "1234WG.jpg", "metal": "yellow"}],
        "fn",
        {"metal": "metal"},
        [],
    )
    counts = derive_filename_tags(tmp_path)
    assert counts["already_tagged"] == 1
    assert counts["derived"] == 0


def test_derive_skips_ambiguous(tmp_path):
    _build_index(tmp_path, ["1234GG.jpg"], ["1234"])
    counts = derive_filename_tags(tmp_path)
    assert counts["derived"] == 0
    assert counts["skipped_ambiguous"] == 1


def test_derive_idempotent(tmp_path):
    _build_index(tmp_path, ["1234WG.jpg"], ["1234"])
    c1 = derive_filename_tags(tmp_path)
    assert c1["derived"] == 1
    c2 = derive_filename_tags(tmp_path)
    assert c2["derived"] == 0
    assert c2["already_tagged"] == 1
    # Exactly one metal row in tags, not two
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    n = con.execute("SELECT COUNT(*) FROM tags WHERE field='metal'").fetchone()[0]
    con.close()
    assert n == 1


def test_derive_dry_run_no_writes(tmp_path):
    _build_index(tmp_path, ["1234WG.jpg"], ["1234"])
    counts = derive_filename_tags(tmp_path, dry_run=True)
    assert counts["derived"] == 1
    # No actual rows written
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


def test_derive_appends_to_existing_tag_text(tmp_path):
    _build_index(tmp_path, ["1234WG.jpg"], ["1234"])
    import_tags(
        tmp_path,
        [{"fn": "1234WG.jpg", "note": "ring"}],
        "fn",
        {},
        ["note"],
    )
    counts = derive_filename_tags(tmp_path)
    assert counts["derived"] == 1
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    text = con.execute("SELECT text FROM tag_text WHERE row=0").fetchone()[0]
    con.close()
    assert "white" in text
    assert "ring" in text


def test_derive_creates_tag_text_when_absent(tmp_path):
    _build_index(tmp_path, ["1234WG.jpg"], ["1234"])
    counts = derive_filename_tags(tmp_path)
    assert counts["derived"] == 1
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    row = con.execute("SELECT text FROM tag_text WHERE row=0").fetchone()
    con.close()
    assert row is not None
    assert "white" in row[0]


def test_derive_annotation_noise_stripped(tmp_path):
    """Dash/hash/space annotation suffixes are ignored before metal rule."""
    _build_index(tmp_path, ["1234YG-2.jpg"], ["1234"])
    counts = derive_filename_tags(tmp_path)
    assert counts["derived"] == 1
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    rows = con.execute("SELECT value FROM tags WHERE field='metal'").fetchall()
    con.close()
    assert ("yellow",) in rows


def test_derive_survives_existing_spreadsheet_tags(tmp_path):
    """Other fields in tags table must survive the derive run."""
    _build_index(
        tmp_path, ["1234WG.jpg", "5678.jpg"], ["1234", "5678"]
    )
    import_tags(
        tmp_path,
        [{"fn": "5678.jpg", "cat": "ring"}],
        "fn",
        {"category": "cat"},
        [],
    )
    derive_filename_tags(tmp_path)
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    cat = con.execute(
        "SELECT value FROM tags WHERE row=1 AND field='category'"
    ).fetchall()
    con.close()
    assert ("ring",) in cat
