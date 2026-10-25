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
    auuc: float                 # Raw Area under uplift curve
    qini_score: float           # Raw Qini score vs random baseline (Area - Area_random)
    auuc_norm: float = 0.0      # Normalized AUUC in [0, 1] relative to oracle upper bound
    qini_norm: float = 0.0      # Normalized Qini coefficient in [0, 1] relative to oracle upper bound


def _compute_curve_arrays(
    y_sorted: np.ndarray,
    t_sorted: np.ndarray,
    indices: np.ndarray,
    N: int,
    trapz_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_n_t = np.cumsum(t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))
    cum_n_c = np.cumsum(1 - t_sorted)

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

    auuc = float(trapz_fn(uplift, fracs))
    au_qini = float(trapz_fn(qini, fracs))
    return fracs, uplift, qini, auuc, au_qini


def calculate_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    cate_pred: np.ndarray,
    n_bins: int = 100,
) -> UpliftCurveResult:
    """Compute cumulative Uplift curve, Qini curve, AUUC, Qini score and normalized variants.

    Sort instances by predicted CATE in descending order.
    Normalized AUUC and normalized Qini are scaled relative to the theoretical upper bound
    (optimal oracle ordering where treated converters and control non-converters rank first).
    """
    y = np.asarray(y_true, float)
    t = np.asarray(treatment, int)
    scores = np.asarray(cate_pred, float)
    N = len(y)

    order = np.argsort(-scores)
    step = max(1, N // n_bins)
    indices = np.arange(step - 1, N, step)
    if indices[-1] != N - 1:
        indices = np.append(indices, N - 1)

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    fracs, uplift, qini, auuc, au_qini = _compute_curve_arrays(
        y[order], t[order], indices, N, trapz_fn
    )

    # Theoretical optimal ordering for upper bound normalization
    opt_score = np.where(
        (t == 1) & (y == 1), 3.0,
        np.where((t == 0) & (y == 0), 2.0,
        np.where((t == 1) & (y == 0), 1.0, 0.0))
    )
    order_opt = np.argsort(-opt_score)
    _, _, _, opt_auuc, opt_au_qini = _compute_curve_arrays(
        y[order_opt], t[order_opt], indices, N, trapz_fn
    )

    # Random baseline areas
    random_auuc = 0.5 * uplift[-1]
    random_au_qini = 0.5 * qini[-1]

    # Raw Qini score (Area(Model) - Area(Random))
    qini_coef = float(au_qini - random_au_qini)

    # Normalized Qini and AUUC relative to optimal upper bound
    qini_norm = float(np.clip((au_qini - random_au_qini) / max(1e-9, opt_au_qini - random_au_qini), -1.0, 1.0))
    auuc_norm = float(np.clip((auuc - random_auuc) / max(1e-9, opt_auuc - random_auuc), -1.0, 1.0))

    return UpliftCurveResult(
        fractions=fracs,
        uplift_values=uplift,
        qini_values=qini,
        auuc=auuc,
        qini_score=qini_coef,
        auuc_norm=auuc_norm,
        qini_norm=qini_norm,
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
