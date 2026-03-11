import numpy as np

from popgen.population.evaluation import dot_product, dtw, rouge_l


def test_dot_product_with_normalization():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    sim = dot_product(a, norm=True)
    assert sim.shape == (2, 2)
    assert sim[0, 0] == 1.0
    assert sim[1, 1] == 1.0
    assert sim[0, 1] == 0.0


def test_dtw_similarity_identical_vs_different():
    same = dtw([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]])
    diff = dtw([[0.0], [1.0], [2.0]], [[2.0], [2.0], [2.0]])
    assert same == 1.0
    assert 0.0 < diff < same


def test_rouge_l_similarity():
    s1 = rouge_l("go left then up", "go left then up")
    s2 = rouge_l("go left then up", "wait in place")
    assert s1 == 1.0
    assert 0.0 <= s2 < s1
