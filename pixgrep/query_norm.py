from __future__ import annotations

import re

# Generic jewelry-trade shorthand -> plain English. Whole-token match only.
_SHORTHAND = {
    "wg": "white gold",
    "yg": "yellow gold",
    "rg": "rose gold",
    "tt": "two-tone",
    "2t": "two-tone",
    "dia": "diamond",
    "diam": "diamond",
    "cttw": "carat total weight",
    "ctw": "carat total weight",
    "ct": "carat",
    "eng": "engagement",
    "anniv": "anniversary",
    # "sol" is ambiguous (solitaire vs solid) — deliberately left alone.
}

# A token like "1ct" or "0.5ct": number fused directly to the "ct" abbreviation.
_FUSED_CARAT_RE = re.compile(r"^(\d+(?:\.\d+)?)ct$", re.IGNORECASE)

# Tokens: an optional decimal/integer number fused to trailing letters (to keep
# "1ct"/"0.5ct" intact for the fused-carat check), or a plain alphanumeric run.
_TOKEN_RE = re.compile(r"\d+\.\d+[A-Za-z]+|\d+[A-Za-z]+|[A-Za-z0-9]+")

_MULTISPACE_RE = re.compile(r" {2,}")


def _replace_token(match: re.Match) -> str:
    token = match.group(0)
    fused = _FUSED_CARAT_RE.match(token)
    if fused:
        return f"{fused.group(1)} carat"
    return _SHORTHAND.get(token.lower(), token)


def normalize_query(q: str) -> str:
    """Expand trade shorthand to plain English via whole-token replacement.

    Case-insensitive; tokens are alphanumeric runs, so "dia" inside "diamond"
    or "radiant" never matches. Casing of unmatched tokens is preserved.
    """
    result = _TOKEN_RE.sub(_replace_token, q)
    return _MULTISPACE_RE.sub(" ", result)
