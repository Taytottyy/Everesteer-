"""Round-1 candidates beyond v_shallow: target sets, slower learning, wider columns, ensembles.

python exp4.py holdout | practice
"""
import json
import os
import sys
import time
import cloudpickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load_train, split, evaluate, proxy_core

SHALLOW = dict(n_estimators=800, learning_rate=0.02, num_leaves=4, max_depth=2,
               colsample_bytree=0.1, subsample=0.8, subsample_freq=1,
               min_child_samples=200, verbose=-1, n_jobs=-1)
STUMPS = dict(SHALLOW, n_estimators=2000, num_leaves=2, max_depth=1)
SLOW = dict(SHALLOW, n_estimators=2000, learning_rate=0.01)
WIDE = dict(SHALLOW, colsample_bytree=0.3)
BLEND4 = {"target_everest": 0.4, "target_Ayachi": 0.3, "target_Tiskiouine": 0.15,
          "target_Saghro": 0.15}

# name -> list of (params, target weights, member weight); members are rank-averaged
VARIANTS = {
    "sh_all15":   [(SHALLOW, "all15", 1.0)],
    "sh_slow":    [(SLOW, BLEND4, 1.0)],
    "sh_wide":    [(WIDE, BLEND4, 1.0)],
    "sh_stumps":  [(SHALLOW, BLEND4, 0.5), (STUMPS, BLEND4, 0.5)],
    "sh_main":    [(SHALLOW, {"target_everest": 1.0}, 1.0)],
}


def resolve(tw, targets):
    return {t: 1 / len(targets) for t in targets} if tw == "all15" else tw


def fit_all(df, feats, targets):
    """Fit every (params, target) pair any variant needs, once."""
    X, cache = df[feats].values, {}
    for members in VARIANTS.values():
        for params, tw, _ in members:
            for t in resolve(tw, targets):
                key = (id(params), t)
                if key not in cache:
                    m = df[t].notna().values
                    cache[key] = lgb.LGBMRegressor(**params).fit(X[m], df.loc[m, t].values)
    return cache


def make_predict(members, cache, feats, targets):
    parts = [([(cache[(id(p), t)], w) for t, w in resolve(tw, targets).items()], mw)
             for p, tw, mw in members]

    def predict(live_features):
        X = live_features.reindex(columns=feats).astype("float32").replace(-1, np.nan).values
        out = np.zeros(len(X))
        for models, mw in parts:
            s = sum(w * pd.Series(m.predict(X)).rank(pct=True).values for m, w in models)
            out += mw * pd.Series(s).rank(pct=True).values
        return pd.Series(out, index=live_features.index).rank(pct=True).to_frame("prediction")
    return predict


def main():
    mode = sys.argv[1]
    df, feats = load_train()
    targets = [c for c in df.columns if c.startswith("target_")]
    if mode == "holdout":
        fit, test = split(df)
        core = proxy_core(fit, feats)
        cache = fit_all(fit, feats, targets)
        Xt = test[feats].set_axis(test.index)
        for name, members in VARIANTS.items():
            p = make_predict(members, cache, feats, targets)(Xt).iloc[:, 0].values
            evaluate(test, p, core, name)
        return

    from everestapi import EverestAPI
    client = EverestAPI(api_key=os.environ["EIQ_API_KEY"], base_url="https://hackathon.everesteer.ai")
    val = pd.read_parquet("futures_validation.parquet")
    ids = val["id"].values if "id" in val.columns else val.index.values
    Xv = val[feats].set_axis(ids)
    names = json.load(open("models/names.json"))
    cache = fit_all(df, feats, targets)
    runs = {}
    for name, members in VARIANTS.items():
        fn = make_predict(members, cache, feats, targets)
        pkl = f"models/v_{name}.pkl"
        with open(pkl, "wb") as f:
            cloudpickle.dump(fn, f, protocol=5)
        pred = pd.Series(fn(Xv).iloc[:, 0].values).groupby(val["exped"].values).rank(pct=True).values
        label = f"v_{name}"
        if label not in names:
            names[label] = client.create_model(name=f"ty-{label}")["name"]
            json.dump(names, open("models/names.json", "w"), indent=1)
        res = client.submit_validation_diagnostics(
            model_id=names[label], predictions=pd.DataFrame({"id": ids, "prediction": pred}),
            model_pkl=pkl, model_pkl_python_version="3.11")
        runs[name] = res["upload_id"]
        print("submitted", name, flush=True)
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
