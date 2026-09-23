"""Submit models to the OPEN EVENT ROUND (live split). Uses the per-event upload pool.

usage: python round.py v_base flip:v_base v_neut50 ...
  <name>       -> models/<name>.pkl as trained
  flip:<name>  -> the same model with its sign reversed (saved as models/flip_<name>.pkl)
Refuses to run unless get_started() reports an open round. Needs EIQ_API_KEY.
"""
import json
import os
import sys
import cloudpickle
import pandas as pd
from everestapi import EverestAPI

PYV = f"{sys.version_info.major}.{sys.version_info.minor}"
NAMES = "models/names.json"
client = EverestAPI(api_key=os.environ["EIQ_API_KEY"], base_url="https://hackathon.everesteer.ai")


def load(label):
    """Return (predict, pkl_path) for a label, building the flipped pickle on demand."""
    if label.startswith("flip:"):
        base = label[5:]
        pkl = f"models/flip_{base}.pkl"
        if not os.path.exists(pkl):
            inner = cloudpickle.load(open(f"models/{base}.pkl", "rb"))

            def predict(live_features, inner=inner):
                return 1.0 - inner(live_features)

            with open(pkl, "wb") as f:
                cloudpickle.dump(predict, f, protocol=5)
        return cloudpickle.load(open(pkl, "rb")), pkl, f"flip_{base}"
    pkl = f"models/{label}.pkl"
    return cloudpickle.load(open(pkl, "rb")), pkl, label


def main():
    cad = client.get_started().get("cadence") or {}
    if cad.get("intake_fenced") or not cad.get("open_window"):
        sys.exit(f"No round open (phase={cad.get('phase')}, "
                 f"{cad.get('seconds_until_next_phase')}s to next phase). Nothing submitted.")
    print("OPEN ROUND:", cad["open_window"], "phase ends", cad.get("phase_ends_at"))

    live = pd.read_parquet(client.download_dataset(split="live"))
    feats = [c for c in live.columns if c.startswith("feature_")]
    ids = live["id"] if "id" in live.columns else live.index
    X = live[feats].set_axis(ids.values)
    names = json.load(open(NAMES))

    for label in sys.argv[1:]:
        predict, pkl, key = load(label)
        raw = predict(X).iloc[:, 0].values
        pred = pd.Series(raw).groupby(live["exped"].values).rank(pct=True).values
        out = pd.DataFrame({"id": ids.values, "prediction": pred})
        assert out["prediction"].between(0, 1).all() and out["id"].is_unique
        if key not in names:
            names[key] = client.create_model(name=f"ty-{key}")["name"]
            json.dump(names, open(NAMES, "w"), indent=1)
        res = client.submit_event_predictions(model_id=names[key], predictions=out,
                                              model_pkl=pkl, model_pkl_python_version=PYV)
        print(f"{key:20s} -> {names[key]:32s} upload {res.get('upload_id')} "
              f"remaining {res.get('uploads_remaining')}")


if __name__ == "__main__":
    main()
