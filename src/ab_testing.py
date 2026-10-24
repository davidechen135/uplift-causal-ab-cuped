"""Standardised A/B test statistics: SRM, t / Welch / z tests, delta-method lift."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats

SRM_ALPHA = 0.001


def srm_test(
    observed_counts: Sequence[int],
    expected_ratios: Optional[Sequence[float]] = None,
    alpha: float = SRM_ALPHA,
) -> Dict:
    """Sample Ratio Mismatch check with Pearson chi-square goodness-of-fit.

    chi2 = sum_k (O_k - E_k)^2 / E_k,  E_k = N * r_k,  df = K - 1.
    Flag SRM when p < alpha (industry convention alpha = 0.001).
    """
    obs = np.asarray(observed_counts, dtype=float)
    if expected_ratios is None:
        expected_ratios = np.full(len(obs), 1.0 / len(obs))
    r = np.asarray(expected_ratios, dtype=float)
    r = r / r.sum()
    exp = obs.sum() * r
    chi2, p = stats.chisquare(obs, exp)
    return dict(
        observed=obs.astype(int).tolist(),
        expected=exp.round(1).tolist(),
        observed_ratio=(obs / obs.sum()).round(5).tolist(),
        expected_ratio=r.round(5).tolist(),
        chi2=float(chi2),
        p_value=float(p),
        alpha=alpha,
        srm_detected=bool(p < alpha),
    )


def ttest(y_t: np.ndarray, y_c: np.ndarray, equal_var: bool = False, alpha: float = 0.05) -> Dict:
    """Two-sample t-test (Student if equal_var else Welch) for mean difference."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    n1, n0 = len(y_t), len(y_c)
    m1, m0 = y_t.mean(), y_c.mean()
    v1, v0 = y_t.var(ddof=1), y_c.var(ddof=1)
    diff = m1 - m0
    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)
        se = np.sqrt(sp2 * (1 / n1 + 1 / n0))
        dof = n1 + n0 - 2
    else:
        se = np.sqrt(v1 / n1 + v0 / n0)
        dof = (v1 / n1 + v0 / n0) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1))
    tstat = diff / se
    p = 2 * stats.t.sf(abs(tstat), dof)
    q = stats.t.ppf(1 - alpha / 2, dof)
    return dict(
        test="student_t" if equal_var else "welch_t",
        n_t=n1,
        n_c=n0,
        mean_t=float(m1),
        mean_c=float(m0),
        var_t=float(v1),
        var_c=float(v0),
        diff=float(diff),
        se=float(se),
        t=float(tstat),
        dof=float(dof),
        p_value=float(p),
        ci_low=float(diff - q * se),
        ci_high=float(diff + q * se),
        significant=bool(p < alpha),
    )


def proportion_ztest(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> Dict:
    """Two-proportion z-test. Pooled SE for the test, unpooled SE for the CI."""
    p1, p0 = x_t / n_t, x_c / n_c
    pp = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(pp * (1 - pp) * (1 / n_t + 1 / n_c))
    z = (p1 - p0) / se_pool
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p1 * (1 - p1) / n_t + p0 * (1 - p0) / n_c)
    q = stats.norm.ppf(1 - alpha / 2)
    return dict(
        test="two_proportion_z",
        rate_t=float(p1),
        rate_c=float(p0),
        diff=float(p1 - p0),
        z=float(z),
        p_value=float(p),
        se=float(se),
        ci_low=float(p1 - p0 - q * se),
        ci_high=float(p1 - p0 + q * se),
        significant=bool(p < alpha),
    )


def relative_lift_delta(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> Dict:
    """Relative lift  L = mean_t / mean_c - 1  with delta-method variance.

    Var(m_t/m_c) ≈ Var(m_t)/m_c^2 + m_t^2 Var(m_c)/m_c^4   (independent arms)
    """
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    mt, mc = y_t.mean(), y_c.mean()
    vt, vc = y_t.var(ddof=1) / len(y_t), y_c.var(ddof=1) / len(y_c)
    lift = mt / mc - 1
    se = np.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
    q = stats.norm.ppf(1 - alpha / 2)
    z = lift / se
    return dict(
        relative_lift=float(lift),
        se=float(se),
        ci_low=float(lift - q * se),
        ci_high=float(lift + q * se),
        p_value=float(2 * stats.norm.sf(abs(z))),
    )


def analyze_metric(y: np.ndarray, t: np.ndarray, binary: Optional[bool] = None, alpha: float = 0.05) -> Dict:
    """Full readout for one metric: absolute diff test + relative lift."""
    y, t = np.asarray(y, float), np.asarray(t).astype(int)
    if binary is None:
        binary = set(np.unique(y)).issubset({0.0, 1.0})
    y_t, y_c = y[t == 1], y[t == 0]
    out = dict(binary=bool(binary))
    if binary:
        out["abs"] = proportion_ztest(int(y_t.sum()), len(y_t), int(y_c.sum()), len(y_c), alpha)
    out["welch"] = ttest(y_t, y_c, equal_var=False, alpha=alpha)
    out["relative"] = relative_lift_delta(y_t, y_c, alpha)
    return out
