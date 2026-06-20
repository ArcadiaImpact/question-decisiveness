"""Tests for the metric + glue. The metric path is torch-free; the fit test is
skipped if torch is unavailable."""

import numpy as np
import pytest

from question_decisiveness import (
    decisiveness, decisiveness_raw, predict_matrix_caseV,
    combine_orderings, pref_to_edges,
)


def test_bounds_and_extremes():
    # Equal utilities -> every pair is a coin flip -> 0.
    assert decisiveness(np.zeros(5)) == pytest.approx(0.0, abs=1e-9)
    # Huge spread -> every pair near-certain -> ~1.
    assert decisiveness(np.array([-1e3, 0.0, 1e3])) == pytest.approx(1.0, abs=1e-6)


def test_monotonic_in_spread():
    weak = decisiveness(np.array([0.5, 0.0, -0.5]))
    strong = decisiveness(np.array([5.0, 0.0, -5.0]))
    assert 0.0 < weak < strong < 1.0


def test_predict_matrix_symmetry():
    mu = np.array([2.0, 0.0, -1.0])
    P = predict_matrix_caseV(mu)
    assert np.allclose(P + P.T, 1.0)          # P_ij + P_ji = 1
    assert np.allclose(np.diag(P), 0.5)


def test_decisiveness_raw():
    rows = [{"p_util": 1.0}, {"p_util": 0.0}, {"p_util": 0.5}]
    # |2*1-1|, |2*0-1|, |2*0.5-1| = 1, 1, 0 -> mean 2/3
    assert decisiveness_raw(rows) == pytest.approx(2 / 3)
    assert np.isnan(decisiveness_raw([]))


def test_combine_orderings_cancels_slot_bias():
    # A pure additive slot bias (+0.1 to whoever is in slot A) must cancel.
    n = 2
    ordered = {(0, 1): 0.6, (1, 0): 0.6}   # symmetric -> true pref is 0.5
    pref = combine_orderings(n, ordered)
    assert pref[0, 1] == pytest.approx(0.5)
    assert pref[1, 0] == pytest.approx(0.5)


def test_pref_to_edges():
    pref = np.array([[0.5, 0.8], [0.2, 0.5]])
    edges = pref_to_edges(pref)
    assert edges == [{"i": 0, "j": 1, "p_util": 0.8}]


def test_fit_recovers_order():
    torch = pytest.importorskip("torch")
    from question_decisiveness import fit_caseV_mle
    # Build edges consistent with mu_true = [2, 0, -2] via the Case V matrix.
    mu_true = np.array([2.0, 0.0, -2.0])
    edges = pref_to_edges(predict_matrix_caseV(mu_true))
    mu = fit_caseV_mle(edges, n=3, steps=2000)["mu"]
    assert list(np.argsort(-mu)) == [0, 1, 2]
    assert decisiveness(mu) == pytest.approx(decisiveness(mu_true), abs=0.02)
