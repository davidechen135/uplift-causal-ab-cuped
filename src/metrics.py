"""Uplift evaluation metrics: Uplift / Qini curves, AUUC, Qini coefficient, uplift@k, PEHE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class UpliftCurveResult:
    fractions: np.ndarray       # population fraction from 0 to 1
    uplift_values: np.ndarray   # cumulative incremental gains
    qini_values: np.ndarray     # cumulative Qini values
    auuc: float                 # Area under uplift curve
    qini_score: float           # Normalized Qini score vs random baseline


def calculate_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    cate_pred: np.ndarray,
    n_bins: int = 100,
) -> UpliftCurveResult:
    """Compute cumulative Uplift curve, Qini curve, AUUC and Qini score.

    Sort instances by predicted CATE in descending order.
    """
    y = np.asarray(y_true, float)
    t = np.asarray(treatment, int)
    scores = np.asarray(cate_pred, float)
    N = len(y)

    order = np.argsort(-scores)
    y_sorted = y[order]
    t_sorted = t[order]

    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_n_t = np.cumsum(t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))
    cum_n_c = np.cumsum(1 - t_sorted)

    # Subsample at n_bins points
    step = max(1, N // n_bins)
    indices = np.arange(step - 1, N, step)
    if indices[-1] != N - 1:
        indices = np.append(indices, N - 1)

    fracs = (indices + 1) / N
    y_t_vals = cum_y_t[indices]
    n_t_vals = np.maximum(cum_n_t[indices], 1e-9)
    y_c_vals = cum_y_c[indices]
    n_c_vals = np.maximum(cum_n_c[indices], 1e-9)

    # Uplift Curve: (y_t/n_t - y_c/n_c) * (n_t + n_c)
    uplift = (y_t_vals / n_t_vals - y_c_vals / n_c_vals) * (indices + 1)
    # Qini Curve: y_t - y_c * (n_t / n_c)
    qini = y_t_vals - y_c_vals * (n_t_vals / n_c_vals)

    # Prepend origin (0, 0)
    fracs = np.insert(fracs, 0, 0.0)
    uplift = np.insert(uplift, 0, 0.0)
    qini = np.insert(qini, 0, 0.0)

    # Area under curve via trapezoidal rule (support numpy 1.x and 2.x)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auuc = float(trapz_fn(uplift, fracs))
    au_qini = float(trapz_fn(qini, fracs))

    # Random baseline area
    random_au_qini = 0.5 * qini[-1]
    # Qini coefficient = (Area(Model) - Area(Random)) / max(1e-9, Area(Random))
    qini_coef = float(au_qini - random_au_qini)

    return UpliftCurveResult(
        fractions=fracs,
        uplift_values=uplift,
        qini_values=qini,
        auuc=auuc,
        qini_score=qini_coef,
    )


def uplift_at_k(
    y_true: np.ndarray,
    treatment: np.ndarray,
    cate_pred: np.ndarray,
    k: float = 0.2,
) -> Dict[str, float]:
    """Incremental response rate in the top k fraction of users."""
    y = np.asarray(y_true, float)
    t = np.asarray(treatment, int)
    scores = np.asarray(cate_pred, float)

    n_top = int(np.ceil(len(y) * k))
    order = np.argsort(-scores)[:n_top]

    y_top = y[order]
    t_top = t[order]

    nt = np.sum(t_top == 1)
    nc = np.sum(t_top == 0)
    yt = np.sum(y_top[t_top == 1])
    yc = np.sum(y_top[t_top == 0])

    rate_t = yt / max(nt, 1)
    rate_c = yc / max(nc, 1)
    incremental_rate = rate_t - rate_c

    return {
        f"uplift@{int(k*100)}%": float(incremental_rate),
        "treatment_rate": float(rate_t),
        "control_rate": float(rate_c),
        "net_conversions_top_k": float(incremental_rate * n_top),
    }


def pehe_score(tau_true: np.ndarray, tau_pred: np.ndarray) -> float:
    """Precision in Estimation of Heterogeneous Effect (PEHE)."""
    tau_true = np.asarray(tau_true, float)
    tau_pred = np.asarray(tau_pred, float)
    return float(np.sqrt(np.mean((tau_pred - tau_true) ** 2)))
