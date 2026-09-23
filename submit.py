"""Fit candidate blends on the full train split and save each as a cloudpickled predict().

usage: python build.py            -> builds every candidate into models/<name>.pkl
"""
import os
import sys
import cloudpickle
import numpy as np
import pandas as pd
import lightgbm as lgb

PARAMS = dict(n_estimators=1500, learning_rate=0.02, num_leaves=31, max_depth=6,
              colsample_bytree=0.1, subsample=0.8, subsample_freq=1,
              min_child_samples=200, verbose=-1, n_jobs=-1)

CANDIDATES = {
    "everest": {"target_everest": 1.0},
    "main_aux": {"target_everest": 0.5, "target_Tougroute": 0.1, "target_Tiskiouine": 0.1,
                 "target_Saghro": 0.1, "target_Gourza": 0.1, "target_Ayachi": 0.1},
    "all15": None,  # equal weight over every target, filled in below
    "lowbench": {"target_everest": 0.4, "target_Ayachi": 0.3, "target_Tiskiouine": 0.15,
                 "target_Saghro": 0.15},
}


def make_predict(models, weights, feats):
    def predict(live_features):
        X = live_features.reindex(columns=feats).astype("float32").replace(-1, np.nan)
        out = np.zeros(len(X))
        for t, w in weights.items():
            out += w * pd.Series(models[t].predict(X)).rank(pct=True).values
        return pd.Series(out, index=live_features.index).rank(pct=True).to_frame("prediction")
    return predict


def main():
    df = pd.read_parquet("futures_train.parquet")
    feats = [c for c in df.columns if c.startswith("feature_")]
    targets = [c for c in df.columns if c.startswith("target_")]
    CANDIDATES["all15"] = {t: 1 / len(targets) for t in targets}
    X = df[feats].astype("float32").replace(-1, np.nan)

    needed = sorted({t for w in CANDIDATES.values() for t in w})
    models = {}
    for t in needed:
        m = df[t].notna().values
        models[t] = lgb.LGBMRegressor(**PARAMS).fit(X[m], df.loc[m, t])
        print("fit", t, flush=True)

    os.makedirs("models", exist_ok=True)
    for name, w in CANDIDATES.items():
        fn = make_predict({t: models[t] for t in w}, w, feats)
        with open(f"models/{name}.pkl", "wb") as f:
            cloudpickle.dump(fn, f, protocol=5)
        print("saved", name)
    print("python", f"{sys.version_info.major}.{sys.version_info.minor}")


if __name__ == "__main__":
    main()
