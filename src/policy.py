"""Budget-constrained subsidy / coupon allocation policies and quadrant analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class PolicyEvaluationResult:
    policy_name: str
    target_fraction: float
    n_targeted: int
    net_incremental_conversions: float
    treatment_rate: float
    control_rate: float
    cost: float
    revenue: float
    net_profit: float
    quadrant_shares: Optional[Dict[str, float]] = None


def evaluate_targeting_policy(
    policy_name: str,
    target_mask: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    coupon_cost: float = 1.0,
    conversion_value: float = 20.0,
    quadrants: Optional[np.ndarray] = None,
) -> PolicyEvaluationResult:
    """Evaluate targeting policy on an RCT sample."""
    n_targeted = int(target_mask.sum())
    frac = n_targeted / max(1, len(target_mask))

    if n_targeted == 0:
        return PolicyEvaluationResult(
            policy_name=policy_name,
            target_fraction=0.0,
            n_targeted=0,
            net_incremental_conversions=0.0,
            treatment_rate=0.0,
            control_rate=0.0,
            cost=0.0,
            revenue=0.0,
            net_profit=0.0,
        )

    y_sub = y[target_mask]
    t_sub = t[target_mask]

    n_t = np.sum(t_sub == 1)
    n_c = np.sum(t_sub == 0)
    y_t = np.sum(y_sub[t_sub == 1])
    y_c = np.sum(y_sub[t_sub == 0])

    p_t = y_t / max(1, n_t)
    p_c = y_c / max(1, n_c)
    incremental_rate = p_t - p_c
    net_conversions = incremental_rate * n_targeted

    cost = n_targeted * coupon_cost
    revenue = net_conversions * conversion_value
    net_profit = revenue - cost

    quad_dict = None
    if quadrants is not None:
        quad_sub = quadrants[target_mask]
        quad_dict = {
            "Persuadables": float(np.mean(quad_sub == "Persuadables")),
            "Sure Things": float(np.mean(quad_sub == "Sure Things")),
            "Lost Causes": float(np.mean(quad_sub == "Lost Causes")),
            "Sleeping Dogs": float(np.mean(quad_sub == "Sleeping Dogs")),
        }

    return PolicyEvaluationResult(
        policy_name=policy_name,
        target_fraction=frac,
        n_targeted=n_targeted,
        net_incremental_conversions=float(net_conversions),
        treatment_rate=float(p_t),
        control_rate=float(p_c),
        cost=float(cost),
        revenue=float(revenue),
        net_profit=float(net_profit),
        quadrant_shares=quad_dict,
    )


def compare_policies(
    cate_pred: np.ndarray,
    response_pred: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    budget_fraction: float = 0.25,
    coupon_cost: float = 1.0,
    conversion_value: float = 20.0,
    quadrants: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, PolicyEvaluationResult]:
    """Compare Random vs Response Model vs Uplift Model under identical budget."""
    N = len(y)
    k = int(N * budget_fraction)
    rng = np.random.default_rng(seed)

    # 1. Random Policy
    rand_perm = rng.permutation(N)
    random_mask = np.zeros(N, bool)
    random_mask[rand_perm[:k]] = True

    # 2. Response Model Policy (targets highest P(Y=1))
    resp_order = np.argsort(-response_pred)
    response_mask = np.zeros(N, bool)
    response_mask[resp_order[:k]] = True

    # 3. Uplift Model Policy (targets highest CATE tau(x))
    # Exclude Sleeping Dogs with tau < 0
    uplift_order = np.argsort(-cate_pred)
    valid_uplift = [idx for idx in uplift_order if cate_pred[idx] > 0]
    uplift_mask = np.zeros(N, bool)
    uplift_mask[valid_uplift[:k]] = True

    res_random = evaluate_targeting_policy(
        "Random", random_mask, y, t, coupon_cost, conversion_value, quadrants
    )
    res_response = evaluate_targeting_policy(
        "ResponseModel (SureThings)", response_mask, y, t, coupon_cost, conversion_value, quadrants
    )
    res_uplift = evaluate_targeting_policy(
        "Uplift (Persuadables)", uplift_mask, y, t, coupon_cost, conversion_value, quadrants
    )

    return {
        "Random": res_random,
        "ResponseModel": res_response,
        "Uplift": res_uplift,
    }
