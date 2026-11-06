import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t = pd.read_csv("data/model_table.csv")
t["compound"] = t[["soft", "medium", "hard"]].idxmax(axis=1)
rows = []
for (race, comp), g in t.groupby(["Race", "compound"]):
    if len(g) < 30 or g.tyre_age.max() - g.tyre_age.min() < 8:
        continue
    slope = np.polyfit(g.tyre_age, g.lap_delta, 1)[0]
    rows.append((race, comp, slope, g.track_temp.mean()))
s = pd.DataFrame(rows, columns=["Race", "compound", "slope", "track_temp"])
print(s.round(3).to_string())

fig, ax = plt.subplots()
for c, g in s.groupby("compound"):
    ax.scatter(g.track_temp, g.slope, label=c)
ax.set_xlabel("Mean track temp (C)")
ax.set_ylabel("Degradation slope (s per lap of tyre age)")
ax.set_title("Degradation rate vs track temperature")
ax.legend()
plt.savefig("plots/plot3_temp_effect.png", dpi=150)
plt.show()
