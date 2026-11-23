import os
import pandas as pd
import matplotlib.pyplot as plt

s = pd.read_csv("data/stints.csv")[["Race", "Driver", "stint", "slope"]]
labels = {"mean": "compound mean", "lr": "LR", "rf": "RF", "nn": "NN"}
maes = {}
for k, lab in labels.items():
    f = f"data/stint_oof_{k}.csv"
    if os.path.exists(f):
        m = s.merge(pd.read_csv(f), on=["Race", "Driver", "stint"])
        maes[lab] = (m.pred - m.slope).abs().mean()

fig, ax = plt.subplots()
bars = ax.bar(list(maes), list(maes.values()))
ax.bar_label(bars, fmt="%.4f")
ax.set_ylabel("MAE of stint degradation rate (s/lap), held-out races")
ax.set_title("Model comparison (out-of-fold)")
plt.savefig("plots/plot2_model_comparison.png", dpi=150)
print({k: round(float(v), 4) for k, v in maes.items()})
