"""Unit tests for A/B testing, CUPED, and Uplift modeling."""
import numpy as np
import pytest

from src.ab_testing import srm_test, ttest, proportion_ztest
from src.cuped import estimate_theta, cuped_adjust, cuped_analysis
from src.power import required_n_per_arm, sample_size_comparison
from src.learners import SLearner, TLearner, XLearner
from src.metrics import calculate_uplift_curve, uplift_at_k
from src.data import generate_coupon_rct


def test_srm_detection():
    # Balanced 50:50
    res_bal = srm_test([5020, 4980], [0.5, 0.5])
    assert not res_bal["srm_detected"]
    assert res_bal["p_value"] > 0.05

    # Severe mismatch (e.g. 60:40 with expected 50:50)
    res_mismatch = srm_test([6000, 4000], [0.5, 0.5])
    assert res_mismatch["srm_detected"]
    assert res_mismatch["p_value"] < 0.001


def test_cuped_unbiased_and_variance_reduction():
    rng = np.random.default_rng(42)
    N = 10000
    # Covariate X
    X = rng.normal(100, 20, size=N)
    # Treatment T
    T = rng.integers(0, 2, size=N)
    # True outcome Y with correlation ~0.7 to X
    true_effect = 5.0
    noise = rng.normal(0, 15, size=N)
    Y = 0.8 * X + true_effect * T + noise

    res = cuped_analysis(Y, X, T)
    # Variance must be strictly reduced
    assert res["variance_reduction"] > 0.30
    assert res["se_reduction"] > 0.15
    # Treatment effect estimate remains unbiased within confidence interval
    assert abs(res["raw"]["diff"] - true_effect) < 1.0
    assert abs(res["adjusted"]["diff"] - true_effect) < 1.0


def test_uplift_learners_recover_treatment_effect():
    data = generate_coupon_rct(n=5000, seed=42)
    # Train / test split
    X_train, y_train, t_train = data.X[:3500], data.y[:3500], data.t[:3500]
    X_test = data.X[3500:]

    t_learner = TLearner(binary=True, seed=42)
    t_learner.fit(X_train, y_train, t_train)
    cate_pred = t_learner.predict_cate(X_test)

    assert len(cate_pred) == len(X_test)
    assert not np.all(np.isnan(cate_pred))

    curve = calculate_uplift_curve(data.y[3500:], data.t[3500:], cate_pred)
    assert curve.auuc > 0
    assert -1.0 <= curve.auuc_norm <= 1.0
    assert -1.0 <= curve.qini_norm <= 1.0
    assert len(curve.fractions) == len(curve.uplift_values)

    assert data.cate_true is not None
    from src.metrics import pehe_score
    pehe = pehe_score(data.cate_true[3500:], cate_pred)
    assert 0 < pehe < 1.0
