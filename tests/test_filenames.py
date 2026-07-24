from pixgrep.filenames import group_key


def test_strips_trailing_numbers():
    pat = r"[-_ ]*\d+$"
    assert group_key("ABC123-1.jpg", pat) == "abc123"
    assert group_key("ABC123-2.jpg", pat) == "abc123"


def test_variants_group_together():
    pat = r"[-_ ]*(?:main|alt|\d+)$"
    assert group_key("ring_main.jpg", pat) == "ring"
    assert group_key("ring_alt.jpg", pat) == "ring"
    assert group_key("ring_2.jpg", pat) == "ring"


def test_no_strip_when_no_match():
    assert group_key("solo.png", r"[-_ ]*\d+$") == "solo"


def test_never_returns_empty():
    # A name that is entirely strippable must still yield a stable non-empty key.
    assert group_key("12345.jpg", r"[-_ ]*\d+$") != ""


# ---------------------------------------------------------------------------
# Annotation-tail pre-strip (view codes, space/# suffixes). Shapes mirror the
# metal-audit examples (base+colorway SKU with a view-code/duplicate-marker
# tail); SKU numbers are synthetic, not real style numbers.
# ---------------------------------------------------------------------------

# Production group_strip_pattern (see config.local.json / _local/config.fullindex.json).
_PROD_PATTERN = r"(?<=\d)[a-z]+\d*[a-z]*(?:[-_ ]?\d+)?$"


def test_strips_underscore_view_code_tail_a1_cut():
    # Base+colorway SKU with a view-code tail (the bug found in the metal
    # audit): "_A1_cut" must not survive into the group key.
    assert group_key("R9001YC7_A1_cut.jpg", _PROD_PATTERN) == "r9001"


def test_strips_underscore_view_code_tail_c1_cut():
    assert group_key("R9002YC7_C1_cut.jpg", _PROD_PATTERN) == "r9002"


def test_strips_underscore_view_code_tail_t_cut():
    assert group_key("R9003TYC7_T_cut.jpg", _PROD_PATTERN) == "r9003"


def test_strips_trailing_hash_duplicate_marker():
    assert group_key("R9101#2.JPG", _PROD_PATTERN) == "r9101"


def test_strips_trailing_space_annotation():
    assert group_key("R9201 YG.jpg", _PROD_PATTERN) == "r9201"
    assert group_key("R9202 W.jpg", _PROD_PATTERN) == "r9202"
    assert group_key("R9202 WG.jpg", _PROD_PATTERN) == "r9202"


def test_non_sku_underscore_name_untouched():
    # No digit run survives stripping past the letters — must not be mangled.
    assert group_key("final_approved.jpg", _PROD_PATTERN) == "final_approved"


def test_non_sku_img_number_untouched():
    # "1234" alone (after stripping "img_") has no letters — not a SKU shape.
    assert group_key("img_1234.jpg", _PROD_PATTERN) == "img_1234"


def test_bare_item_type_letter_suffix_not_touched_by_prestrip():
    # No space/underscore/# delimiter present — annotation-tail strip is a
    # no-op here; skip-prefix handling for item-type-letter product lines
    # lives in derive_filename_tags.py, not in group_key.
    assert group_key("ZZ0368R.jpg", _PROD_PATTERN) == "zz0368"
    assert group_key("QQ0306R.jpg", _PROD_PATTERN) == "qq0306"
