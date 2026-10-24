"""LightGBM import helper.

On macOS without Homebrew's ``libomp`` the LightGBM wheel cannot find
``@rpath/libomp.dylib``.  ``scripts/run_all.sh`` exports
``DYLD_FALLBACK_LIBRARY_PATH`` pointing to the libomp shipped inside the
scikit-learn wheel, which fixes it.  This module only gives a helpful error.
"""
from __future__ import annotations

try:
    import lightgbm as lgb  # noqa: F401
except OSError as e:  # pragma: no cover - environment specific
    raise ImportError(
        "LightGBM failed to load its OpenMP runtime (libomp). On macOS run:\n"
        "  export DYLD_FALLBACK_LIBRARY_PATH=$(python -c 'import sklearn,os;"
        "print(os.path.join(os.path.dirname(sklearn.__file__),\".dylibs\"))')\n"
        "or `brew install libomp`, or simply use scripts/run_all.sh.\n"
        f"Original error: {e}"
    ) from e

DEFAULT_CLF_PARAMS = dict(
    n_estimators=250,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=4,
)

DEFAULT_REG_PARAMS = dict(DEFAULT_CLF_PARAMS)


def make_clf(seed: int = 0, **overrides):
    p = dict(DEFAULT_CLF_PARAMS, random_state=seed)
    p.update(overrides)
    return lgb.LGBMClassifier(**p)


def make_reg(seed: int = 0, **overrides):
    p = dict(DEFAULT_REG_PARAMS, random_state=seed)
    p.update(overrides)
    return lgb.LGBMRegressor(**p)
