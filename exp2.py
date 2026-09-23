"""Robustness variants, scored on the train hold-out AND submitted to the practice board.

Variants: feature neutralisation, recency-weighted training, shallow (near-linear) trees.
"""
import json
import os
import sys
import time
import cloudpickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from everestapi import EverestAPI
from common import load_train, split, evaluate, proxy_core, TARGET

BASE = dict(n_estimators=1500, learning_rate=0.02, num_leaves=31, max_depth=6,
            colsample_bytree=0.1, subsample=0.8, subsample_freq=1,
            min_child_samples=200, verbose=-1, n_jobs=-1)
SHALLOW = dict(BASE, n_estimators=800, num_leaves=4, max_depth=2)
STUMPS = dict(BASE, n_estimators=2000, num_leaves=2, max_depth=1)
SHALLOW3 = dict(BASE, n_estimators=1000, num_leaves=8, max_depth=3)
NEUT_K = 10  # neutralise against the K features most correlated with the target
BLEND = {"target_everest": 0.4, "target_Ayachi": 0.3, "target_Tiskiouine": 0.15,
         "target_Saghro": 0.15}


def neutralize(p, X, prop):
    """Remove `prop` of the linear projection of ranked p onto the features X."""
    Xf = np.nan_to_num(X, nan=4.5)
    Xf = np.column_stack([Xf - Xf.mean(0), np.ones(len(Xf))])
    r = pd.Series(p).rank(pct=True).values - 0.5
    beta = np.linalg.lstsq(Xf, r, rcond=None)[0]
    out = r - prop * (Xf @ beta)
    return out / out.std()


def per_era(fn, p, X, eras):
    out = np.empty(len(p))
    for _, idx in pd.Series(range(len(p))).groupby(eras).indices.items():
        out[idx] = fn(p[idx], X[idx])
    return out


def fit_models(df, feats, params, weight_halflife=None):
    X = df[feats].values
    models = {}
    for t in BLEND:
        m = df[t].notna().values
        w = None
        if weight_halflife:
            age = df["era"].max() - df["era"].values
            w = 0.5 ** (age / weight_halflife)
        models[t] = lgb.LGBMRegressor(**params).fit(X[m], df.loc[m, t].values,
                                                    sample_weight=None if w is None else w[m])
    return models


def blend(models, X):
    return sum(w * pd.Series(models[t].predict(X)).rank(pct=True).values
               for t, w in BLEND.items())


VARIANTS = {  # name -> (params, halflife, neutralise proportion)
    "shallow":     (SHALLOW, None, 0.0),
    "stumps":      (STUMPS, None, 0.0),
    "depth3":      (SHALLOW3, None, 0.0),
    "shallow_n50": (SHALLOW, None, 0.5),
    "shallow_n100": (SHALLOW, None, 1.0),
    "stumps_n50":  (STUMPS, None, 0.5),
}


def main():
    mode = sys.argv[1]  # "holdout" or "practice"
    df, feats = load_train()
    if mode == "holdout":
        fit, test = split(df)
        core = proxy_core(fit, feats)
        nidx = [feats.index(c) for c in proxy_core(fit, feats, NEUT_K)]
        Xt, eras = test[feats].values, test["era"].values
        cache = {}
        for name, (params, hl, prop) in VARIANTS.items():
            key = (id(params), hl)
            if key not in cache:
                cache[key] = blend(fit_models(fit, feats, params, hl), Xt)
            p = cache[key]
            if prop:
                p = per_era(lambda a, b: neutralize(a, b, prop), p, Xt[:, nidx], eras)
            evaluate(test, p, core, name)
        return

    client = EverestAPI(api_key=os.environ["EIQ_API_KEY"],
                        base_url="https://hackathon.everesteer.ai")
    val = pd.read_parquet("futures_validation.parquet")
    ids = val["id"].values if "id" in val.columns else val.index.values
    Xv = val[feats].astype("float32").replace(-1, np.nan).values
    names = json.load(open("models/names.json"))
    nidx = [feats.index(c) for c in proxy_core(df, feats, NEUT_K)]
    runs, cache = {}, {}
    for name, (params, hl, prop) in VARIANTS.items():
        key = (id(params), hl)
        if key not in cache:
            cache[key] = fit_models(df, feats, params, hl)
        models = cache[key]

        def predict(live_features, models=models, prop=prop, nidx=nidx):
            X = live_features.reindex(columns=feats).astype("float32").replace(-1, np.nan).values
            p = blend(models, X)
            if prop:
                p = neutralize(p, X[:, nidx], prop)
            return pd.Series(p, index=live_features.index).rank(pct=True).to_frame("prediction")

        pkl = f"models/v_{name}.pkl"
        with open(pkl, "wb") as f:
            cloudpickle.dump(predict, f, protocol=5)
        p = blend(models, Xv)
        if prop:
            p = per_era(lambda a, b: neutralize(a, b, prop), p, Xv[:, nidx], val["exped"].values)
        pred = pd.Series(p).groupby(val["exped"].values).rank(pct=True).values
        label = f"v_{name}"
        if label not in names:
            names[label] = client.create_model(name=f"ty-{label}")["name"]
            json.dump(names, open("models/names.json", "w"), indent=1)
        res = client.submit_validation_diagnostics(
            model_id=names[label], predictions=pd.DataFrame({"id": ids, "prediction": pred}),
            model_pkl=pkl, model_pkl_python_version="3.11")
        runs[name] = res["upload_id"]
        print("submitted", name, names[label], flush=True)

    for _ in range(60):
        st = {k: client.get_diagnostics_run(v) for k, v in runs.items()}
        if all(s["status"] in ("done", "failed") for s in st.values()):
            break
        time.sleep(10)
    for k, s in st.items():
        print(f"PRACTICE {k:12s} CORR {s.get('corr20')}  AIMC {s.get('aimc')}  NCORR {s.get('ncorr')}"
              f"  SCORE {s.get('round_score')}  bench_corr {s.get('example_preds_corr')} {s.get('error') or ''}")


if __name__ == "__main__":
    main()
