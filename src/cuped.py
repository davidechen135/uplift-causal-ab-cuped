"""Variance reduction: CUPED, CUPAC and post-stratification."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from .ab_testing import relative_lift_delta, ttest


def estimate_theta(y: np.ndarray, x: np.ndarray) -> float:
    """theta = Cov(Y, X) / Var(X), estimated on the POOLED sample (both arms).

    Pooling keeps theta independent of the treatment assignment so the
    adjusted estimator stays unbiased (X is pre-period => independent of T).
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    vx = x.var(ddof=1)
    if vx == 0:
        return 0.0
    return float(np.cov(y, x, ddof=1)[0, 1] / vx)


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: Optional[float] = None) -> np.ndarray:
    """Y_cuped = Y - theta * (X - mean(X))."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    if theta is None:
        theta = estimate_theta(y, x)
    return y - theta * (x - x.mean())


def cuped_analysis(y: np.ndarray, x: np.ndarray, t: np.ndarray, alpha: float = 0.05, label: str = "cuped") -> Dict:
    """Compare raw vs CUPED-adjusted Welch tests on the same experiment."""
    y, x, t = np.asarray(y, float), np.asarray(x, float), np.asarray(t).astype(int)
    theta = estimate_theta(y, x)
    rho = float(np.corrcoef(y, x)[0, 1]) if x.std() > 0 else 0.0
    y_adj = cuped_adjust(y, x, theta)
    raw = ttest(y[t == 1], y[t == 0], alpha=alpha)
    adj = ttest(y_adj[t == 1], y_adj[t == 0], alpha=alpha)
    var_red = 1 - y_adj.var(ddof=1) / y.var(ddof=1)
    # relative lift with raw control mean as denominator (CUPED arm means are
    # shifted by theta*(mean_x_arm - mean_x) which is ~0 in expectation)
    rel_raw = relative_lift_delta(y[t == 1], y[t == 0], alpha)
    rel_adj = dict(
        relative_lift=adj["diff"] / raw["mean_c"],
        se=adj["se"] / raw["mean_c"],
    )
    q = stats.norm.ppf(1 - alpha / 2)
    rel_adj["ci_low"] = rel_adj["relative_lift"] - q * rel_adj["se"]
    rel_adj["ci_high"] = rel_adj["relative_lift"] + q * rel_adj["se"]
    return dict(
        method=label,
        theta=theta,
        rho=rho,
        rho_squared=rho**2,
        variance_reduction=float(var_red),  # 1 - Var(Y_cuped)/Var(Y)  (≈ rho^2)
        se_reduction=float(1 - adj["se"] / raw["se"]),
        raw=raw,
        adjusted=adj,
        relative_raw=rel_raw,
        relative_adjusted=rel_adj,
        _y_adj=y_adj,
    )


def cupac_covariate(
    df: pd.DataFrame,
    features: List[str],
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """CUPAC: out-of-fold ML prediction of Y from PRE-treatment features only.

    The treatment indicator is NOT a model input, and predictions are
    cross-fitted, so the covariate is (asymptotically) independent of T and
    can be plugged into CUPED exactly like a historical metric.
    """
    from ._lgbm import make_reg

    X = df[features].to_numpy(dtype=float)
    y = np.asarray(y, float)
    pred = np.zeros(len(y))
    for k, (tr, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(X)):
        m = make_reg(seed + k, n_estimators=200)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def post_stratification(y: np.ndarray, t: np.ndarray, strata: np.ndarray, alpha: float = 0.05) -> Dict:
    """Post-stratified ATE: sum_k w_k (ybar_tk - ybar_ck), w_k = N_k / N."""
    df = pd.DataFrame(dict(y=np.asarray(y, float), t=np.asarray(t).astype(int), s=strata))
    g = df.groupby(["s", "t"])["y"].agg(["mean", "var", "count"]).unstack("t")
    g = g.dropna()
    w = (g["count"][0] + g["count"][1]) / (g["count"][0] + g["count"][1]).sum()
    est = float((w * (g["mean"][1] - g["mean"][0])).sum())
    se = float(np.sqrt((w**2 * (g["var"][1] / g["count"][1] + g["var"][0] / g["count"][0])).sum()))
    q = stats.norm.ppf(1 - alpha / 2)
    z = est / se
    return dict(
        method="post_stratification",
        n_strata=int(len(g)),
        diff=est,
        se=se,
        ci_low=est - q * se,
        ci_high=est + q * se,
        p_value=float(2 * stats.norm.sf(abs(z))),
    )
