import numpy as np
import pytest

from pixgrep.query_norm import normalize_query
from pixgrep.search import SearchEngine
from pixgrep.store import save_index


# --- individual mappings ---

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("wg", "white gold"),
        ("yg", "yellow gold"),
        ("rg", "rose gold"),
        ("tt", "two-tone"),
        ("2t", "two-tone"),
        ("dia", "diamond"),
        ("diam", "diamond"),
        ("cttw", "carat total weight"),
        ("ctw", "carat total weight"),
        ("ct", "carat"),
        ("eng", "engagement"),
        ("anniv", "anniversary"),
    ],
)
def test_mapping(raw, expected):
    assert normalize_query(raw) == expected


def test_sol_left_alone():
    assert normalize_query("sol") == "sol"


# --- token-boundary safety ---

def test_radiant_cut_unchanged():
    assert normalize_query("radiant cut") == "radiant cut"


def test_diamond_unchanged():
    assert normalize_query("diamond") == "diamond"


def test_design_unchanged():
    assert normalize_query("design") == "design"


# --- fused carat ---

def test_fused_carat_integer():
    assert normalize_query("1ct") == "1 carat"


def test_fused_carat_decimal():
    assert normalize_query("0.5ct") == "0.5 carat"


# --- case variants ---

def test_case_insensitive_upper():
    assert normalize_query("YG") == "yellow gold"


def test_case_insensitive_mixed():
    assert normalize_query("Yg") == "yellow gold"


# --- full phrases ---

def test_phrase_yg_band():
    assert normalize_query("YG band") == "yellow gold band"


def test_phrase_tt_bracelet():
    assert normalize_query("TT bracelet") == "two-tone bracelet"


def test_phrase_dia_solitare_1ct():
    assert normalize_query("dia solitare 1ct") == "diamond solitare 1 carat"


# --- idempotency ---

@pytest.mark.parametrize(
    "q",
    ["YG band", "TT bracelet", "dia solitare 1ct", "cttw eng ring", "radiant cut diamond"],
)
def test_idempotent(q):
    once = normalize_query(q)
    twice = normalize_query(once)
    assert once == twice


# --- integration: normalized query reaches the embedder identically ---

class RecordingEmbedder:
    """Records every text batch it's asked to embed; returns a fixed vector."""

    def __init__(self, dim=3):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return np.tile(np.eye(self.dim, dtype=np.float32)[0], (len(texts), 1))

    def embed_images(self, images):
        raise NotImplementedError


@pytest.fixture()
def engine_and_embedder(tmp_path):
    emb = np.eye(3, dtype=np.float32)
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    groups = ["a", "b", "c"]
    save_index(tmp_path, paths, groups, emb)
    embedder = RecordingEmbedder(dim=3)
    return SearchEngine(tmp_path, embedder), embedder


def test_shorthand_query_reaches_embedder_normalized(engine_and_embedder):
    engine, embedder = engine_and_embedder
    engine.text_search("YG band")
    assert embedder.calls[-1] == ["yellow gold band"]


def test_shorthand_and_plain_query_produce_identical_results(engine_and_embedder):
    engine, embedder = engine_and_embedder
    shorthand_results = engine.text_search("YG band")
    plain_results = engine.text_search("yellow gold band")
    assert shorthand_results == plain_results
    assert embedder.calls[-2] == embedder.calls[-1] == ["yellow gold band"]
