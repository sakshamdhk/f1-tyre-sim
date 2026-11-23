import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

r = pd.read_csv("data/pit_results.csv")
cols = [c for c in ["mid", "mean", "lr", "rf", "nn"] if c in r]
names = {"mid": "mid-race", "mean": "compound mean", "lr": "LR", "rf": "RF", "nn": "NN"}
x = np.arange(len(r))
w = 0.8 / (len(cols) + 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x, r.actual, w, label="actual", color="black")
for i, c in enumerate(cols, 1):
    ax.bar(x + i * w, r[c], w, label=names[c])
ax.set_xticks(x + w * len(cols) / 2)
ax.set_xticklabels(r.Race.str.replace(" Grand Prix", ""), rotation=45, ha="right")
ax.set_ylabel("First pit lap")
ax.set_title("Actual vs model-optimal pit lap")
ax.legend()
plt.tight_layout()
plt.savefig("plots/plot4_pit_laps.png", dpi=150)
