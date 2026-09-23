"""Shared data loading and offline scoring for the Everesteer hackathon."""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

TARGET = "target_everest"
GAP = 20          # expeds between fit and hold-out (target horizon)
HOLDOUT = 1000    # most recent expeds held out for scoring
W = {"corr": 1.0, "aimc": 2.0, "ncorr": 1.0}  # from explain_scoring()


def load_train():
    df = pd.read_parquet("futures_train.parquet")
    import glob
    bpath = next(p for p in ["benchmark_futures_train.parquet",
                             "futures_train_benchmark_models.parquet", *glob.glob("*train*bench*.parquet")]
                 if os.path.exists(p))  # SDK versions save this under different names
    bench = pd.read_parquet(bpath)["v1_sherpa"]
    df["v1_sherpa"] = bench.reindex(df.index).values
    feats = [c for c in df.columns if c.startswith("feature_")]
    df[feats] = df[feats].astype("float32").replace(-1, np.nan)
    df["era"] = df["exped"].str[6:].astype(int)
    return df, feats


def split(df):
    eras = np.sort(df["era"].unique())
    test_start = eras[-HOLDOUT]
    fit = df[df["era"] < test_start - GAP]
    test = df[df["era"] >= test_start]
    return fit, test


# --- per-exped kernels (mirror the server: rank-gauss preds, centre target, signed ^1.5) ---
def _gauss(x):
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def _pow(x):
    return np.sign(x) * np.abs(x) ** 1.5


def _corr(p, y):
    a, b = _pow(p), _pow(y - y.mean())
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def era_scores(pred, y, bench, core):
    g = _gauss(pred)
    corr = _corr(g, y)
    bg = _gauss(bench)
    resid = g - (g @ bg) / (bg @ bg) * bg
    aimc = float(np.mean(resid * (y - y.mean())))
    X = np.column_stack([np.nan_to_num(core, nan=np.nanmean(core)), np.ones(len(g))])
    lam = 1e-3 * np.trace(X.T @ X) / X.shape[1]
    beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ g)
    ncorr = _corr(g - X @ beta, y)
    return corr, aimc, ncorr


def evaluate(test, pred, core_feats, label=""):
    rows = []
    for _, idx in test.groupby("era").indices.items():
        sub = test.iloc[idx]
        rows.append(era_scores(pred[idx], sub[TARGET].values, sub["v1_sherpa"].values,
                               sub[core_feats].values))
    r = pd.DataFrame(rows, columns=["corr", "aimc", "ncorr"])
    m = r.mean()
    score = W["corr"] * m["corr"] + W["aimc"] * m["aimc"] + W["ncorr"] * m["ncorr"]
    bc = np.corrcoef(rankdata(pred), rankdata(test["v1_sherpa"]))[0, 1]
    print(f"{label:28s} CORR {m['corr']:+.4f}  AIMC {m['aimc']:+.4f}  NCORR {m['ncorr']:+.4f}"
          f"  SCORE {score:+.4f}  sharpe {r['corr'].mean()/r['corr'].std():.2f}  bench_corr {bc:.2f}")
    return score


def proxy_core(fit, feats, k=10):
    """Stand-in for the unpublished 10-feature NCORR core set."""
    c = fit[feats].corrwith(fit[TARGET]).abs().sort_values(ascending=False)
    return c.index[:k].tolist()
