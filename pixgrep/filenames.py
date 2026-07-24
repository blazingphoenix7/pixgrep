from __future__ import annotations

import re
from pathlib import Path

from .config import DEFAULT_STRIP_PATTERN

# Trailing annotation segment: a space/underscore/hash delimiter followed by a
# short (<=4 char) alphanumeric run at the very end of the stem. Covers
# duplicate-copy markers ("#2"), loose suffix words (e.g. a metal-color code
# after a space), and underscore-delimited view codes ("_A1_cut", "_B_cut",
# "_T_cut").
_ANNOTATION_TAIL = re.compile(r"[ _#][a-z0-9]{1,4}$")
# A base SKU shape: some letters, then a run of >=3 digits, somewhere in the
# string. Used to gate annotation stripping so non-SKU names (no digits, or
# too few) are left untouched.
_SKU_SHAPE = re.compile(r"[a-z].*\d{3,}")


def _strip_annotation_tail(stem: str) -> str:
    """Peel trailing annotation segments one at a time, right to left.

    Each candidate (stem with one more trailing segment removed) must still
    look like a SKU before the strip is accepted; otherwise stop and return
    the stem as of the last accepted strip. This keeps names like
    'final_approved' or 'img_1234' untouched (no digit run remains once you
    strip past the letters), while still peeling multi-segment tails like
    '_a1_cut' down to the true base style.
    """
    working = stem
    while True:
        m = _ANNOTATION_TAIL.search(working)
        if not m:
            break
        candidate = working[: m.start()]
        if not _SKU_SHAPE.search(candidate):
            break
        working = candidate
    return working


def group_key(filename: str, strip_pattern: str = DEFAULT_STRIP_PATTERN) -> str:
    """Reduce a filename to a 'same item' key.

    Strips the trailing view/variant token so that different views of one item
    collapse to the same key. Tune ``strip_pattern`` for your naming scheme
    (set ``group_strip_pattern`` in config.local.json). Never returns an empty
    string.
    """
    original = Path(filename).stem.strip().lower()
    pre_stripped = _strip_annotation_tail(original)
    stem = re.sub(strip_pattern, "", pre_stripped, flags=re.IGNORECASE).strip(" -_")
    return stem or original
