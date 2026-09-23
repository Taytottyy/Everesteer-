"""Predict with saved .pkl models and submit to whichever lane is open.

usage:
  python submit.py practice everest main_aux     # practice board (validation split, free)
  python submit.py round everest main_aux        # open event round (live split, uses uploads)
Needs EIQ_API_KEY in the environment.
"""
import json
import os
import sys
import cloudpickle
import pandas as pd
from everestapi import EverestAPI

PYV = f"{sys.version_info.major}.{sys.version_info.minor}"
client = EverestAPI(api_key=os.environ["EIQ_API_KEY"],
                    base_url=os.environ.get("EIQ_BASE_URL", "https://hackathon.everesteer.ai"))


NAMES = "models/names.json"


def model_name(label):
    """Return the server-assigned name for our private label, creating the model once."""
    names = json.load(open(NAMES)) if os.path.exists(NAMES) else {}
    if label not in names:
        res = client.create_model(name=f"ty-{label}")
        print("create_model", label, res)
        names[label] = res.get("name") or res.get("id")
        json.dump(names, open(NAMES, "w"), indent=1)
    return names[label]


def main():
    lane, labels = sys.argv[1], sys.argv[2:]
    cad = client.get_started().get("cadence") or {}
    if lane == "round":
        if cad.get("intake_fenced") or not cad.get("open_window"):
            sys.exit(f"No round open (phase={cad.get('phase')}). Not submitting.")
        split, submit = "live", client.submit_event_predictions
    else:
        split, submit = "validation", client.submit_validation_diagnostics
    print(f"phase={cad.get('phase')} open_window={cad.get('open_window')} -> {split}")

    df = pd.read_parquet(client.download_dataset(split=split))
    feats = [c for c in df.columns if c.startswith("feature_")]
    ids = df["id"] if "id" in df.columns else df.index
    X = df[feats].set_axis(ids.values)

    for label in labels:
        pkl = f"models/{label}.pkl"
        with open(pkl, "rb") as f:
            predict = cloudpickle.load(f)
        # rank within each exped so the blend is scored per cross-section
        raw = predict(X).iloc[:, 0].values
        pred = pd.Series(raw).groupby(df["exped"].values).rank(pct=True).values
        out = pd.DataFrame({"id": ids.values, "prediction": pred})
        assert out["prediction"].between(0, 1).all() and out["id"].is_unique
        name = model_name(label)
        res = submit(model_id=name, predictions=out, model_pkl=pkl,
                     model_pkl_python_version=PYV)
        print(label, "->", name, res)


if __name__ == "__main__":
    main()
