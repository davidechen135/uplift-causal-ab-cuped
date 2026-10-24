"""Power analysis, sample-size formulas and Monte-Carlo validation (A/A + power)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats


def z_values(alpha: float = 0.05, power: float = 0.8):
    return stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)


def required_n_per_arm(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Two-sided, equal allocation:  n = 2 (z_{1-a/2} + z_{1-b})^2 sigma^2 / MDE^2."""
    za, zb = z_values(alpha, power)
    return int(np.ceil(2 * (za + zb) ** 2 * sigma**2 / mde**2))


def required_n_proportion(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Binary metric: sigma^2 ≈ p(1-p) evaluated at the baseline rate."""
    return required_n_per_arm(np.sqrt(p_base * (1 - p_base)), mde_abs, alpha, power)


def mde_for_n(sigma: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    za, zb = z_values(alpha, power)
    return float((za + zb) * sigma * np.sqrt(2.0 / n_per_arm))


def sample_size_comparison(
    sigma_raw: float, sigma_adj: float, mde: float, alpha: float = 0.05, power: float = 0.8
) -> Dict:
    """n before vs after variance reduction; reduction = 1 - sigma_adj^2/sigma_raw^2."""
    n_raw = required_n_per_arm(sigma_raw, mde, alpha, power)
    n_adj = required_n_per_arm(sigma_adj, mde, alpha, power)
    return dict(
        alpha=alpha,
        power=power,
        mde_abs=float(mde),
        sigma_raw=float(sigma_raw),
        sigma_adjusted=float(sigma_adj),
        n_per_arm_raw=n_raw,
        n_per_arm_adjusted=n_adj,
        n_total_raw=2 * n_raw,
        n_total_adjusted=2 * n_adj,
        sample_size_reduction=float(1 - n_adj / n_raw),
        theoretical_reduction=float(1 - sigma_adj**2 / sigma_raw**2),
    )


# ---------------------------------------------------------------------------
# Monte-Carlo validation
# ---------------------------------------------------------------------------
def _welch_p(yt: np.ndarray, yc: np.ndarray) -> np.ndarray:
    n1, n0 = yt.shape[1], yc.shape[1]
    v1, v0 = yt.var(axis=1, ddof=1), yc.var(axis=1, ddof=1)
    se = np.sqrt(v1 / n1 + v0 / n0)
    tstat = (yt.mean(1) - yc.mean(1)) / se
    dof = (v1 / n1 + v0 / n0) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1))
    return 2 * stats.t.sf(np.abs(tstat), dof)


def _cuped_rows(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Row-wise CUPED with pooled theta (each row = one simulated experiment)."""
    xc = x - x.mean(1, keepdims=True)
    yc = y - y.mean(1, keepdims=True)
    theta = (xc * yc).sum(1) / np.maximum((xc**2).sum(1), 1e-12)
    return y - theta[:, None] * xc


def simulate_experiments(
    y: np.ndarray,
    x: Optional[np.ndarray],
    n_per_arm: int,
    effect: float = 0.0,
    n_sims: int = 1000,
    alpha: float = 0.05,
    use_cuped: bool = False,
    seed: int = 0,
    chunk: int = 50,
) -> Dict:
    """Bootstrap experiments from a (control-only) population.

    effect = 0     -> A/A test, rejection rate = false-positive rate (target alpha)
    effect = MDE   -> rejection rate = empirical power (target 1 - beta)
    Additive constant treatment effect is injected into the treated arm.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    x = None if x is None else np.asarray(x, float)
    N = len(y)
    pvals = []
    done = 0
    while done < n_sims:
        b = min(chunk, n_sims - done)
        idx = rng.integers(0, N, size=(b, 2 * n_per_arm))
        ys = y[idx]
        ys[:, :n_per_arm] += effect
        if use_cuped:
            ys = _cuped_rows(ys, x[idx])
        pvals.append(_welch_p(ys[:, :n_per_arm], ys[:, n_per_arm:]))
        done += b
    p = np.concatenate(pvals)
    rej = float((p < alpha).mean())
    se = float(np.sqrt(rej * (1 - rej) / n_sims))
    return dict(
        n_per_arm=int(n_per_arm),
        effect=float(effect),
        n_sims=int(n_sims),
        cuped=bool(use_cuped),
        rejection_rate=rej,
        rejection_rate_ci95=[rej - 1.96 * se, rej + 1.96 * se],
        _pvalues=p,
    )
