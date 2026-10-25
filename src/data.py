"""Data layer.

1. ``generate_coupon_rct``  - realistic synthetic coupon-marketing RCT with known
   ground truth (potential outcomes, CATE, four principal strata / segments)
   and pre-period covariates for CUPED.
2. ``load_hillstrom``       - Kevin Hillstrom MineThatData e-mail RCT (64,000 rows).
3. ``load_criteo``          - Criteo Uplift v2.1 (13.9M rows) loader with sampling.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
# Official Criteo AI Lab distribution (≈ 297 MB gz, 13,979,592 rows).
# See https://ailab.criteo.com/criteo-uplift-prediction-dataset/
CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
CRITEO_FEATURES = [f"f{i}" for i in range(12)]

SEGMENTS = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]

SYN_FEATURES = [
    "age",
    "tenure_months",
    "is_member",
    "city_tier",
    "device",
    "sessions_30d",
    "cart_adds_7d",
    "days_since_last_order",
    "pre_orders_30d",
    "pre_spend_30d",
    "hist_coupon_redeem_rate",
]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Dataset:
    """Unified container for an RCT dataset."""

    name: str
    df: pd.DataFrame
    features: List[str]
    treatment: str = "treatment"
    outcome: str = "conversion"
    continuous_outcome: Optional[str] = None
    pre_covariate: Optional[str] = None  # pre-period covariate for CUPED
    has_ground_truth: bool = False
    notes: dict = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        return self.df[self.features].values

    @property
    def y(self) -> np.ndarray:
        return self.df[self.outcome].values

    @property
    def t(self) -> np.ndarray:
        return self.df[self.treatment].values

    @property
    def x_pre(self) -> Optional[np.ndarray]:
        if self.pre_covariate and self.pre_covariate in self.df.columns:
            return self.df[self.pre_covariate].values
        return None

    @property
    def cate_true(self) -> Optional[np.ndarray]:
        for col in ["true_cate", "tau_true", "cate_true", "cate"]:
            if col in self.df.columns:
                return self.df[col].values
        return None

    @property
    def segments(self) -> Optional[np.ndarray]:
        for col in ["segment", "quadrant"]:
            if col in self.df.columns:
                return self.df[col].values
        return None


# ---------------------------------------------------------------------------
# 1. Synthetic coupon RCT
# ---------------------------------------------------------------------------
def generate_coupon_rct(
    n: int = 200_000,
    treat_prob: float = 0.5,
    coupon_value: float = 10.0,
    seed: int = 42,
    srm_drop_frac: float = 0.0,
) -> Dataset:
    """Simulate a 14-day coupon RCT on an e-commerce user base.

    Latent traits
    -------------
    a : purchase activity (drives baseline conversion AND pre-period orders ->
        this is why pre-period spend is a good CUPED covariate)
    s : price sensitivity (drives the coupon treatment effect; only observed
        through the noisy proxy ``hist_coupon_redeem_rate`` and tier/cart signals)

    Potential outcomes
    ------------------
    p0(x) = sigmoid(b(x)),  p1(x) = sigmoid(b(x) + tau_logit(x)),
    Y(0) = 1[u < p0],  Y(1) = 1[u < p1]  with a shared u ~ U(0,1)
    (comonotone coupling).  The principal strata defined by (Y(0), Y(1)) are
    exactly the four marketing segments:

        (0,1) Persuadables   (1,1) Sure Things
        (0,0) Lost Causes    (1,0) Sleeping Dogs

    True CATE on conversion = p1(x) - p0(x).

    ``srm_drop_frac`` > 0 simulates a logging bug that silently drops a share
    of *treated low-activity* users -> sample ratio mismatch (and bias).
    """
    rng = np.random.default_rng(seed)

    # --- demographics / static
    age = np.clip(rng.normal(34, 9, n), 18, 70).round()
    tenure = np.clip(rng.exponential(20, n), 0.5, 120).round(1)
    is_member = rng.binomial(1, 0.28, n)
    city_tier = rng.choice([1, 2, 3], size=n, p=[0.3, 0.4, 0.3])
    device = rng.choice([0, 1, 2], size=n, p=[0.45, 0.45, 0.10])  # ios/android/web

    # --- latent traits
    a = rng.normal(0, 1, n) + 0.25 * is_member + 0.004 * (tenure - 20)
    s = 0.35 * (city_tier - 2) - 0.2 * is_member + rng.normal(0, 0.95, n)

    # --- pre-period (30d before experiment) behaviour
    pre_orders = rng.poisson(np.exp(0.2 + 0.75 * a))
    avg_basket = np.exp(rng.normal(4.0 - 0.15 * (city_tier - 2) + 0.1 * is_member, 0.35, n))
    pre_spend = np.where(
        pre_orders > 0, pre_orders * avg_basket * np.exp(rng.normal(0, 0.2, n)), 0.0
    ).round(2)
    sessions = rng.poisson(np.exp(1.4 + 0.5 * a + 0.1 * s))
    cart_adds = rng.poisson(np.exp(0.1 + 0.4 * a + 0.25 * s))
    days_since = np.clip(np.exp(3.3 - 0.6 * a + rng.normal(0, 0.5, n)), 1, 365).round()
    redeem_rate = np.clip(_sigmoid(0.9 * s - 0.6 + rng.normal(0, 0.7, n)), 0, 1).round(3)

    # --- potential outcomes (conversion in 14-day window)
    b = -1.9 + 0.95 * a + 0.25 * is_member - 0.005 * (age - 34) + 0.1 * (device == 0)
    sleeping = (is_member == 1) & (s < -0.3) & (tenure > 18)
    tau_logit = 1.5 * _sigmoid(1.8 * s) - 0.45 - 0.2 * a - 0.9 * sleeping
    p0 = _sigmoid(b)
    p1 = _sigmoid(b + tau_logit)
    u = rng.uniform(size=n)
    y0 = (u < p0).astype(int)
    y1 = (u < p1).astype(int)

    # --- GMV in window (same orders/basket draw for both potential outcomes)
    n_orders = 1 + rng.poisson(np.exp(-1.0 + 0.5 * a))
    gmv_if_conv = n_orders * avg_basket * np.exp(rng.normal(0, 0.25, n))
    gmv0 = y0 * gmv_if_conv
    gmv1 = y1 * gmv_if_conv

    seg = np.select(
        [(y0 == 0) & (y1 == 1), (y0 == 1) & (y1 == 1), (y0 == 0) & (y1 == 0)],
        SEGMENTS[:3],
        default=SEGMENTS[3],
    )

    t = rng.binomial(1, treat_prob, n)
    y = np.where(t == 1, y1, y0)
    gmv = np.where(t == 1, gmv1, gmv0).round(2)

    df = pd.DataFrame(
        dict(
            user_id=np.arange(n),
            age=age,
            tenure_months=tenure,
            is_member=is_member,
            city_tier=city_tier,
            device=device,
            sessions_30d=sessions,
            cart_adds_7d=cart_adds,
            days_since_last_order=days_since,
            pre_orders_30d=pre_orders,
            pre_spend_30d=pre_spend,
            hist_coupon_redeem_rate=redeem_rate,
            treatment=t,
            conversion=y,
            gmv=gmv,
            # coupon cost only incurred when a treated user redeems (converts)
            coupon_cost=(t * y * coupon_value).astype(float),
            # ---- ground truth (never used for training)
            true_p0=p0,
            true_p1=p1,
            true_cate=p1 - p0,
            true_gmv_ite=gmv1 - gmv0,  # realised individual effect on GMV
            y0=y0,
            y1=y1,
            segment=seg,
        )
    )

    if srm_drop_frac > 0:
        drop = (df.treatment == 1) & (a < 0) & (rng.uniform(size=n) < srm_drop_frac)
        df = df.loc[~drop].reset_index(drop=True)

    return Dataset(
        name="synthetic_coupon_rct",
        df=df,
        features=list(SYN_FEATURES),
        treatment="treatment",
        outcome="conversion",
        continuous_outcome="gmv",
        pre_covariate="pre_spend_30d",
        has_ground_truth=True,
        notes=dict(coupon_value=coupon_value, treat_prob=treat_prob, seed=seed, n=n),
    )


# ---------------------------------------------------------------------------
# 2. Hillstrom MineThatData
# ---------------------------------------------------------------------------
def download_file(url: str, dest: str, timeout: int = 120) -> bool:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dest)
        return True
    except Exception as e:  # network errors -> graceful fallback
        print(f"[data] download failed ({type(e).__name__}: {e})")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def load_hillstrom(
    path: str = "data/raw/hillstrom.csv",
    treatment_arm: str = "womens",
    outcome: str = "visit",
    download: bool = True,
    timeout: int = 120,
) -> Optional[Dataset]:
    """Load Hillstrom e-mail RCT (64k customers, 1/3 Mens, 1/3 Womens, 1/3 none).

    treatment_arm : 'womens' -> Womens E-Mail vs No E-Mail
                    'mens'   -> Mens E-Mail   vs No E-Mail
                    'any'    -> any e-mail    vs No E-Mail
    outcome       : 'visit' | 'conversion' (binary); 'spend' kept as continuous.
    Returns None if the file is unavailable and cannot be downloaded.
    """
    # Check fallback paths in parent data directory
    if not os.path.exists(path):
        for candidate in ["../data/raw/hillstrom.csv", "../../data/raw/hillstrom.csv", "data/raw/hillstrom.csv"]:
            if os.path.exists(candidate):
                path = candidate
                break

    if not os.path.exists(path):
        if not download or not download_file(HILLSTROM_URL, path, timeout=timeout):
            return None
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    ctrl = raw["segment"] == "No E-Mail"
    if treatment_arm == "womens":
        keep = ctrl | (raw["segment"] == "Womens E-Mail")
    elif treatment_arm == "mens":
        keep = ctrl | (raw["segment"] == "Mens E-Mail")
    else:
        keep = np.ones(len(raw), dtype=bool)
    d = raw.loc[keep].reset_index(drop=True).copy()
    d["treatment"] = (d["segment"] != "No E-Mail").astype(int)
    d["history_segment_ord"] = d["history_segment"].str.slice(0, 1).astype(int)
    d = pd.get_dummies(d, columns=["zip_code", "channel"], dtype=int)
    feats = [
        "recency",
        "history",
        "history_segment_ord",
        "mens",
        "womens",
        "newbie",
    ] + [c for c in d.columns if c.startswith("zip_code_") or c.startswith("channel_")]
    return Dataset(
        name=f"hillstrom_{treatment_arm}_{outcome}",
        df=d,
        features=feats,
        treatment="treatment",
        outcome=outcome,
        continuous_outcome="spend",
        pre_covariate="history",  # past-12-month spend (pre-period)
        has_ground_truth=False,
        notes=dict(source=HILLSTROM_URL, treatment_arm=treatment_arm),
    )


# ---------------------------------------------------------------------------
# 3. Criteo Uplift v2.1
# ---------------------------------------------------------------------------
def load_criteo(
    path: str = "data/raw/criteo-research-uplift-v2.1.csv.gz",
    sample_frac: Optional[float] = 0.01,
    nrows: Optional[int] = None,
    outcome: str = "visit",
    seed: int = 0,
    chunksize: int = 1_000_000,
    download: bool = False,
) -> Optional[Dataset]:
    """Load Criteo Uplift v2.1 (schema: f0..f11, treatment, conversion, visit, exposure).

    The file is ~297 MB compressed / 13.98M rows; by default it is NOT downloaded
    (set ``download=True`` or fetch ``CRITEO_URL`` manually into ``data/raw/``).
    Rows are Bernoulli-sampled chunk-by-chunk with ``sample_frac`` to bound memory.
    Treatment ratio is ~85/15, so propensity != 0.5 (handled by X/DR-learners and
    the weighted class-transformation).
    """
    # Check fallback paths in parent data directory
    if not os.path.exists(path):
        for candidate in ["../data/raw/hillstrom.csv", "../../data/raw/hillstrom.csv", "data/raw/hillstrom.csv"]:
            if os.path.exists(candidate):
                path = candidate
                break

    if not os.path.exists(path):
        if not download or not download_file(CRITEO_URL, path, timeout=3600):
            return None
    rng = np.random.default_rng(seed)
    dtypes = {c: "float32" for c in CRITEO_FEATURES}
    dtypes.update(treatment="int8", conversion="int8", visit="int8", exposure="int8")
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, nrows=nrows, dtype=dtypes):
        if sample_frac is not None and sample_frac < 1:
            chunk = chunk.loc[rng.uniform(size=len(chunk)) < sample_frac]
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    return Dataset(
        name=f"criteo_{outcome}",
        df=df,
        features=list(CRITEO_FEATURES),
        treatment="treatment",
        outcome=outcome,
        has_ground_truth=False,
        notes=dict(source=CRITEO_URL, sample_frac=sample_frac),
    )
