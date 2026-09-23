"""Offline experiments: baseline, per-target models, blends, benchmark orthogonalisation."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from common import load_train, split, evaluate, proxy_core, TARGET

PARAMS = dict(n_estimators=1500, learning_rate=0.02, num_leaves=31, max_depth=6,
              colsample_bytree=0.1, subsample=0.8, subsample_freq=1,
              min_child_samples=200, verbose=-1, n_jobs=-1)

df, feats = load_train()
fit, test = split(df)
core = proxy_core(fit, feats)
print(f"fit {fit['era'].nunique()} expeds / {len(fit)} rows, test {test['era'].nunique()} expeds")

evaluate(test, test["v1_sherpa"].values, core, "benchmark v1_sherpa")

# target correlations to pick diverse aux targets
tcols = [c for c in df.columns if c.startswith("target_")]
tc = fit[tcols].corr()[TARGET].sort_values()
print(tc.round(2).to_string())

preds = {}
for t in [TARGET] + [c for c in tc.index if c != TARGET][:0]:
    pass

targets = [TARGET, "target_Tougroute"] + [c for c in tc.index if c != TARGET][:4]
for t in dict.fromkeys(targets):
    tt = time.time()
    m = fit[t].notna()
    model = lgb.LGBMRegressor(**PARAMS).fit(fit.loc[m, feats], fit.loc[m, t])
    preds[t] = model.predict(test[feats])
    evaluate(test, preds[t], core, f"lgbm {t[7:]} ({time.time()-tt:.0f}s)")

# benchmark proxy (live benchmark is withheld from event keys)
bm = lgb.LGBMRegressor(**PARAMS).fit(fit[feats], fit["v1_sherpa"])
bench_proxy = bm.predict(test[feats])
print("bench proxy corr to true bench:",
      np.corrcoef(rankdata(bench_proxy), rankdata(test["v1_sherpa"]))[0, 1].round(3))


def rank_by_era(x):
    return pd.Series(x, index=test.index).groupby(test["era"].values).rank(pct=True).values


R = {k: rank_by_era(v) for k, v in preds.items()}
bp = rank_by_era(bench_proxy)
evaluate(test, sum(R.values()) / len(R), core, "blend all targets")
main = 0.5 * R[TARGET] + 0.5 * sum(v for k, v in R.items() if k != TARGET) / (len(R) - 1)
evaluate(test, main, core, "blend 50/50 main/aux")
for a in [0.25, 0.5, 0.75, 1.0]:
    evaluate(test, main - a * bp, core, f"blend - {a} bench_proxy")

pd.DataFrame(R, index=test.index).assign(bench_proxy=bp).to_parquet("holdout_preds.parquet")
