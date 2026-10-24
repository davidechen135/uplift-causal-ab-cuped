"""Uplift modeling meta-learners: S-Learner, T-Learner, X-Learner, and Class Transformation.

Supports both LightGBM (default) and scikit-learn GradientBoosting fallback.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from ._lgbm import make_clf, make_reg
    _USE_LGBM = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as make_clf
    from sklearn.ensemble import GradientBoostingRegressor as make_reg
    _USE_LGBM = False


class BaseLearner:
    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> BaseLearner:
        raise NotImplementedError

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate Individual Treatment Effect (CATE) tau(x) = E[Y|X,T=1] - E[Y|X,T=0]."""
        raise NotImplementedError


class SLearner(BaseLearner):
    """S-Learner (Single Model).

    Fits a single model mu(X, T) -> Y.
    tau(x) = mu(x, 1) - mu(x, 0)
    """

    name = "S-Learner"

    def __init__(self, binary: bool = True, seed: int = 42):
        self.binary = binary
        self.seed = seed
        self.model = make_clf(seed=seed) if binary else make_reg(seed=seed)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> SLearner:
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        t = np.asarray(t, float).reshape(-1, 1)
        Xt = np.hstack([X, t])
        self.model.fit(Xt, y)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        n = len(X)
        ones = np.ones((n, 1))
        zeros = np.zeros((n, 1))
        X1 = np.hstack([X, ones])
        X0 = np.hstack([X, zeros])

        if self.binary and hasattr(self.model, "predict_proba"):
            p1 = self.model.predict_proba(X1)[:, 1]
            p0 = self.model.predict_proba(X0)[:, 1]
        else:
            p1 = self.model.predict(X1)
            p0 = self.model.predict(X0)
        return p1 - p0


class TLearner(BaseLearner):
    """T-Learner (Two Models).

    Fits mu_1(X) on treatment group (T=1), mu_0(X) on control group (T=0).
    tau(x) = mu_1(x) - mu_0(x)
    """

    name = "T-Learner"

    def __init__(self, binary: bool = True, seed: int = 42):
        self.binary = binary
        self.seed = seed
        self.m0 = make_clf(seed=seed) if binary else make_reg(seed=seed)
        self.m1 = make_clf(seed=seed + 1) if binary else make_reg(seed=seed + 1)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> TLearner:
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        t = np.asarray(t, int)
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if self.binary and hasattr(self.m0, "predict_proba"):
            p1 = self.m1.predict_proba(X)[:, 1]
            p0 = self.m0.predict_proba(X)[:, 1]
        else:
            p1 = self.m1.predict(X)
            p0 = self.m0.predict(X)
        return p1 - p0

    def predict_response(self, X: np.ndarray) -> np.ndarray:
        """Overall response baseline under treatment: P(Y=1|X, T=1)."""
        X = np.asarray(X, float)
        if self.binary and hasattr(self.m1, "predict_proba"):
            return self.m1.predict_proba(X)[:, 1]
        return self.m1.predict(X)


class XLearner(BaseLearner):
    """X-Learner (Künzel et al., PNAS 2019).

    Particularly effective in RCTs with imbalanced treatment/control ratio or sparse treatment effect.
    Step 1: Fit mu_0(X) and mu_1(X)
    Step 2: Impute imputed counterfactual effects:
            D_1 = Y_1 - mu_0(X_1)
            D_0 = mu_1(X_0) - Y_0
    Step 3: Fit tau_1(X) on (X_1, D_1) and tau_0(X) on (X_0, D_0)
    Step 4: Combine: tau(x) = e(x) * tau_0(x) + (1 - e(x)) * tau_1(x)
    """

    name = "X-Learner"

    def __init__(self, binary: bool = True, seed: int = 42, propensity: Optional[float] = None):
        self.binary = binary
        self.seed = seed
        self.propensity = propensity
        self.m0 = make_clf(seed=seed) if binary else make_reg(seed=seed)
        self.m1 = make_clf(seed=seed + 1) if binary else make_reg(seed=seed + 1)
        # Second stage regressions on continuous imputed effects D
        self.t0 = make_reg(seed=seed + 2)
        self.t1 = make_reg(seed=seed + 3)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> XLearner:
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        t = np.asarray(t, int)

        # 1. Fit first stage response models
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])

        # 2. Impute counterfactuals
        if self.binary and hasattr(self.m0, "predict_proba"):
            y0_hat_on_1 = self.m0.predict_proba(X[t == 1])[:, 1]
            y1_hat_on_0 = self.m1.predict_proba(X[t == 0])[:, 1]
        else:
            y0_hat_on_1 = self.m0.predict(X[t == 1])
            y1_hat_on_0 = self.m1.predict(X[t == 0])

        D1 = y[t == 1] - y0_hat_on_1
        D0 = y1_hat_on_0 - y[t == 0]

        # 3. Fit second stage models on imputed effects
        self.t1.fit(X[t == 1], D1)
        self.t0.fit(X[t == 0], D0)

        # 4. Propensity
        if self.propensity is None:
            self.e = float(t.mean())
        else:
            self.e = float(self.propensity)

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        tau1 = self.t1.predict(X)
        tau0 = self.t0.predict(X)
        # Weighted combination by propensity
        return self.e * tau0 + (1.0 - self.e) * tau1


class ClassTransformationLearner(BaseLearner):
    """Lai's Generalized Class Transformation (for binary 50/50 RCT).

    Z_i = Y_i * T_i + (1 - Y_i) * (1 - T_i)
    When P(T=1) = 0.5, E[2*Z - 1 | X] = tau(X).
    """

    name = "ClassTransform"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = make_clf(seed=seed)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> ClassTransformationLearner:
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        t = np.asarray(t, int)
        z = y * t + (1 - y) * (1 - t)
        self.model.fit(X, z)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if hasattr(self.model, "predict_proba"):
            pz = self.model.predict_proba(X)[:, 1]
        else:
            pz = self.model.predict(X)
        return 2.0 * pz - 1.0
