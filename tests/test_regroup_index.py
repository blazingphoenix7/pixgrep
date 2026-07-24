"""Tests for scripts/regroup_index.py — recompute group_key for every row."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixgrep.store import save_index
from scripts.regroup_index import regroup_index

STRIP_PATTERN = r"(?<=\d)[a-z]+\d*[a-z]*(?:[-_ ]?\d+)?$"


def _build_index(tmp_path, filenames: list[str], group_keys: list[str]) -> None:
    n = len(filenames)
    paths = [f"folder/{fn}" for fn in filenames]
    emb = np.eye(max(n, 4), dtype=np.float32)[:n]
    save_index(tmp_path, paths, group_keys, emb)


def _group_keys(tmp_path) -> dict[int, str]:
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    rows = con.execute("SELECT row, group_key FROM images ORDER BY row").fetchall()
    con.close()
    return dict(rows)


def test_regroup_fixes_view_code_tail(tmp_path):
    # Simulate the pre-fix bug: group_key stored as the whole unstripped stem.
    _build_index(
        tmp_path,
        ["AB4321YC7_A1_cut.jpg"],
        ["ab4321yc7_a1_cut"],
    )
    counts = regroup_index(tmp_path, STRIP_PATTERN)
    assert counts["total"] == 1
    assert counts["changed"] == 1
    assert _group_keys(tmp_path)[0] == "ab4321"


def test_regroup_leaves_already_correct_keys_unchanged(tmp_path):
    _build_index(tmp_path, ["1234WG.jpg"], ["1234"])
    counts = regroup_index(tmp_path, STRIP_PATTERN)
    assert counts["changed"] == 0
    assert _group_keys(tmp_path)[0] == "1234"


def test_regroup_dry_run_no_writes(tmp_path):
    _build_index(tmp_path, ["AB4321YC7_A1_cut.jpg"], ["ab4321yc7_a1_cut"])
    counts = regroup_index(tmp_path, STRIP_PATTERN, dry_run=True)
    assert counts["changed"] == 1
    # DB untouched
    assert _group_keys(tmp_path)[0] == "ab4321yc7_a1_cut"


def test_regroup_multiple_rows_mixed(tmp_path):
    _build_index(
        tmp_path,
        ["AB4321YC7_A1_cut.jpg", "1234WG.jpg", "final_approved.jpg"],
        ["ab4321yc7_a1_cut", "1234", "final_approved"],
    )
    counts = regroup_index(tmp_path, STRIP_PATTERN)
    assert counts["total"] == 3
    assert counts["changed"] == 1
    keys = _group_keys(tmp_path)
    assert keys[0] == "ab4321"
    assert keys[1] == "1234"
    assert keys[2] == "final_approved"
