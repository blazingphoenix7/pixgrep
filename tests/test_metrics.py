import numpy as np

from pixgrep.metrics import group_recall_at_k


def test_perfect_clusters_score_one():
    # two tight clusters, well separated → siblings are nearest neighbors
    a = np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
    emb = np.vstack([a, b])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    groups = ["a", "a", "b", "b"]
    assert group_recall_at_k(emb, groups, k=1) == 1.0


def test_singletons_are_ignored():
    emb = np.eye(3, dtype=np.float32)
    groups = ["x", "y", "z"]  # no group has 2 members
    assert group_recall_at_k(emb, groups, k=1) == 0.0


def test_partial_recall_between_zero_and_one():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((10, 5)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5"]
    score = group_recall_at_k(emb, groups, k=3)
    assert 0.0 <= score <= 1.0
