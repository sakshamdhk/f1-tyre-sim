import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("data/clean_laps.csv")
race = "Bahrain Grand Prix"
avg = df[df.Race == race].groupby(["Compound", "tyre_age"]).lap_delta.mean().reset_index()

fig, ax = plt.subplots()
for c, g in avg.groupby("Compound"):
    ax.plot(g.tyre_age, g.lap_delta, marker="o", label=c)
ax.set_xlabel("Tyre age (laps)")
ax.set_ylabel("Lap delta (s)")
ax.set_title(race)
ax.legend()
plt.savefig("plots/plot1_degradation.png", dpi=150)
plt.show()
