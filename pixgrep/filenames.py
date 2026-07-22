from __future__ import annotations

import re
from pathlib import Path

from .config import DEFAULT_STRIP_PATTERN


def group_key(filename: str, strip_pattern: str = DEFAULT_STRIP_PATTERN) -> str:
    """Reduce a filename to a 'same item' key.

    Strips the trailing view/variant token so that different views of one item
    collapse to the same key. Tune ``strip_pattern`` for your naming scheme
    (set ``group_strip_pattern`` in config.local.json). Never returns an empty
    string.
    """
    original = Path(filename).stem.strip().lower()
    stem = re.sub(strip_pattern, "", original, flags=re.IGNORECASE).strip(" -_")
    return stem or original
