"""Comprehensive benchmark runner for A/B testing, CUPED, and Uplift Modeling."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.ab_testing import srm_test, ttest, relative_lift_delta
from src.cuped import cuped_analysis
from src.data import generate_coupon_rct, load_hillstrom
from src.learners import SLearner, TLearner, XLearner, ClassTransformationLearner
from src.metrics import calculate_uplift_curve, uplift_at_k, pehe_score
from src.policy import compare_policies
from src.power import sample_size_comparison, simulate_experiments


def run_ab_cuped_benchmark(output_dir: Path) -> dict:
    print("--- [1/3] Running A/B Testing & CUPED Variance Reduction ---")
    data = generate_coupon_rct(n=50_000, seed=42)

    # 1. SRM Check
    n_treat = int(data.t.sum())
    n_ctrl = len(data.t) - n_treat
    srm_res = srm_test([n_treat, n_ctrl], [0.5, 0.5])

    # 2. CUPED on GMV / spend continuous outcome
    cont_col = data.continuous_outcome if data.continuous_outcome in data.df.columns else "conversion"
    y = data.df[cont_col].values
    x_pre = data.x_pre
    cuped_res = cuped_analysis(y, x_pre, data.t)

    # 3. Sample size reduction calculation
    sigma_raw = float(y.std(ddof=1))
    sigma_adj = float(cuped_res["_y_adj"].std(ddof=1))
    mde = 0.05 * float(y.mean())  # 5% relative MDE
    sample_size_res = sample_size_comparison(sigma_raw, sigma_adj, mde=mde, alpha=0.05, power=0.80)

    # 4. Monte Carlo A/A and Power validation (1000 runs)
    print("  Running Monte-Carlo A/A and Power simulations...")
    aa_sim = simulate_experiments(y[data.t == 0], x_pre[data.t == 0], n_per_arm=1000, effect=0.0, n_sims=500, use_cuped=True)
    power_sim = simulate_experiments(y[data.t == 0], x_pre[data.t == 0], n_per_arm=1000, effect=mde, n_sims=500, use_cuped=True)

    # Plot CUPED variance comparison
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    plt.hist(y, bins=60, alpha=0.5, label=f"Raw Metric (Var={y.var(ddof=1):.2f})", color="steelblue", density=True)
    plt.hist(cuped_res["_y_adj"], bins=60, alpha=0.5, label=f"CUPED Adjusted (Var={cuped_res['_y_adj'].var(ddof=1):.2f})", color="crimson", density=True)
    plt.title(f"CUPED Variance Reduction: {cuped_res['variance_reduction']*100:.1f}% Reduction")
    plt.xlabel("Outcome Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "cuped_variance_reduction.png", dpi=200)
    plt.close()

    return {
        "srm": {
            "observed": srm_res["observed"],
            "chi2": srm_res["chi2"],
            "p_value": srm_res["p_value"],
            "srm_detected": srm_res["srm_detected"],
        },
        "cuped": {
            "correlation_rho": cuped_res["rho"],
            "variance_reduction_pct": cuped_res["variance_reduction"] * 100,
            "se_reduction_pct": cuped_res["se_reduction"] * 100,
            "raw_diff": cuped_res["raw"]["diff"],
            "raw_p_value": cuped_res["raw"]["p_value"],
            "adjusted_diff": cuped_res["adjusted"]["diff"],
            "adjusted_p_value": cuped_res["adjusted"]["p_value"],
        },
        "sample_size": {
            "n_per_arm_raw": sample_size_res["n_per_arm_raw"],
            "n_per_arm_adjusted": sample_size_res["n_per_arm_adjusted"],
            "sample_size_reduction_pct": sample_size_res["sample_size_reduction"] * 100,
        },
        "simulations": {
            "aa_false_positive_rate": aa_sim["rejection_rate"],
            "empirical_power": power_sim["rejection_rate"],
        },
    }


def run_uplift_benchmark(output_dir: Path) -> dict:
    print("--- [2/3] Running Uplift Modeling Benchmark ---")
    data = generate_coupon_rct(n=40_000, seed=42)

    n_train = 28_000
    X_train, y_train, t_train = data.X[:n_train], data.y[:n_train], data.t[:n_train]
    X_test, y_test, t_test = data.X[n_train:], data.y[n_train:], data.t[n_train:]
    tau_test = data.cate_true[n_train:] if data.cate_true is not None else None
    quad_test = data.segments[n_train:] if data.segments is not None else None

    models = {
        "S-Learner": SLearner(binary=True, seed=42),
        "T-Learner": TLearner(binary=True, seed=42),
        "X-Learner": XLearner(binary=True, seed=42),
        "ClassTransform": ClassTransformationLearner(seed=42),
    }

    curves = {}
    preds = {}
    metrics_summary = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train, t_train)
        cate = model.predict_cate(X_test)
        preds[name] = cate
        curve = calculate_uplift_curve(y_test, t_test, cate)
        curves[name] = curve
        u10 = uplift_at_k(y_test, t_test, cate, k=0.10)
        u20 = uplift_at_k(y_test, t_test, cate, k=0.20)
        pehe = pehe_score(tau_test, cate) if tau_test is not None else None

        metrics_summary[name] = {
            "auuc": float(curve.auuc),
            "qini_score": float(curve.qini_score),
            "uplift@10%": float(u10["uplift@10%"]),
            "uplift@20%": float(u20["uplift@20%"]),
            "pehe": float(pehe) if pehe is not None else None,
        }

    # Plot Qini curves
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    # Random baseline
    c0 = list(curves.values())[0]
    plt.plot([0, 1], [0, c0.qini_values[-1]], "k--", label="Random Targeting", alpha=0.7)
    for name, c in curves.items():
        plt.plot(c.fractions, c.qini_values, label=f"{name} (Qini={c.qini_score:.1f})", linewidth=1.8)
    plt.xlabel("Population Targeted Fraction")
    plt.ylabel("Cumulative Incremental Conversions (Qini)")
    plt.title("Uplift Modeling: Qini Curves on Held-Out Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "qini_curves.png", dpi=200)
    plt.close()

    # Also test on real Hillstrom data if available
    hillstrom_metrics = None
    try:
        data_h = load_hillstrom()
        if len(data_h.df) > 0:
            print("  Evaluating on Hillstrom MineThatData benchmark...")
            n_h_train = int(len(data_h.df) * 0.7)
            xh_tr, yh_tr, th_tr = data_h.X[:n_h_train], data_h.y[:n_h_train], data_h.t[:n_h_train]
            xh_te, yh_te, th_te = data_h.X[n_h_train:], data_h.y[n_h_train:], data_h.t[n_h_train:]
            tl = TLearner(binary=True, seed=42)
            tl.fit(xh_tr, yh_tr, th_tr)
            cate_h = tl.predict_cate(xh_te)
            ch = calculate_uplift_curve(yh_te, th_te, cate_h)
            hillstrom_metrics = {
                "dataset": "Hillstrom Email Analytics",
                "n_samples": len(data_h.df),
                "auuc": float(ch.auuc),
                "qini_score": float(ch.qini_score),
            }
    except Exception as e:
        print(f"  Note: Hillstrom test skipped ({e})")

    # [3/3] Policy evaluation
    print("--- [3/3] Running Budget-Constrained Subsidy Targeting ---")
    best_cate = preds["X-Learner"]
    # Fit a response model (predicts P(Y=1|X, T=1))
    t_learner = models["T-Learner"]
    resp_prob = t_learner.predict_response(X_test) if hasattr(t_learner, "predict_response") else resp_prob

    policies = compare_policies(
        cate_pred=best_cate,
        response_pred=resp_prob,
        y=y_test,
        t=t_test,
        budget_fraction=0.25,
        coupon_cost=1.0,
        conversion_value=25.0,
        quadrants=quad_test,
    )

    policy_summary = {}
    for pname, pres in policies.items():
        policy_summary[pname] = {
            "n_targeted": pres.n_targeted,
            "net_incremental_conversions": pres.net_incremental_conversions,
            "cost": pres.cost,
            "revenue": pres.revenue,
            "net_profit": pres.net_profit,
            "quadrant_shares": pres.quadrant_shares,
        }

    return {
        "models": metrics_summary,
        "hillstrom": hillstrom_metrics,
        "policy_comparison": policy_summary,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    ab_res = run_ab_cuped_benchmark(output_dir)
    uplift_res = run_uplift_benchmark(output_dir)

    all_out = {
        "ab_cuped": ab_res,
        "uplift": uplift_res,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nAll Project 3 results saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
