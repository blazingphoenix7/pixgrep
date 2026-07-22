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
