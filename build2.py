"""v_shallow2: depth-2 trees, 500 rounds, fit only on expeds with <20% missing features."""
import json, os, sys, time
import cloudpickle, numpy as np, pandas as pd, lightgbm as lgb
from everestapi import EverestAPI
from common import load_train

P = dict(n_estimators=500, learning_rate=0.02, num_leaves=4, max_depth=2, colsample_bytree=0.1,
         subsample=0.8, subsample_freq=1, min_child_samples=200, verbose=-1, n_jobs=-1)
B = {"target_everest": 0.4, "target_Ayachi": 0.3, "target_Tiskiouine": 0.15, "target_Saghro": 0.15}

df, feats = load_train()
emiss = df[feats].isna().mean(axis=1).groupby(df["era"]).transform("mean")
df = df[emiss < 0.2]
print("fit rows", len(df), "expeds", df["era"].nunique(), flush=True)
models = {t: lgb.LGBMRegressor(**P).fit(df.loc[df[t].notna(), feats], df.loc[df[t].notna(), t]) for t in B}


def predict(live_features):
    X = live_features.reindex(columns=feats).astype("float32").replace(-1, np.nan)
    s = sum(w * pd.Series(models[t].predict(X)).rank(pct=True).values for t, w in B.items())
    return pd.Series(s, index=live_features.index).rank(pct=True).to_frame("prediction")


with open("models/v_shallow2.pkl", "wb") as f:
    cloudpickle.dump(predict, f, protocol=5)
print("saved models/v_shallow2.pkl", flush=True)

client = EverestAPI(api_key=os.environ["EIQ_API_KEY"], base_url="https://hackathon.everesteer.ai")
val = pd.read_parquet("futures_validation.parquet")
ids = val.index.values
pred = pd.Series(predict(val[feats]).iloc[:, 0].values).groupby(val["exped"].values).rank(pct=True).values
names = json.load(open("models/names.json"))
names["v_shallow2"] = names.get("v_shallow2") or client.create_model(name="ty-v_shallow2")["name"]
json.dump(names, open("models/names.json", "w"), indent=1)
res = client.submit_validation_diagnostics(model_id=names["v_shallow2"],
      predictions=pd.DataFrame({"id": ids, "prediction": pred}),
      model_pkl="models/v_shallow2.pkl", model_pkl_python_version="3.11")
for _ in range(40):
    s = client.get_diagnostics_run(res["upload_id"])
    if s["status"] in ("done", "failed"):
        break
    time.sleep(10)
print("PRACTICE v_shallow2", names["v_shallow2"], s.get("status"), "CORR", s.get("corr20"), "AIMC", s.get("aimc"),
      "NCORR", s.get("ncorr"), "SCORE", s.get("round_score"), s.get("error") or "")
