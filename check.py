import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
df = pd.read_csv("data/clean_laps.csv")
big = df[df.lap_delta > 8]
print(len(big))
print(big.groupby("Race").size().sort_values(ascending=False))
print(big.sort_values("lap_delta", ascending=False)[["Race", "Driver", "LapNumber", "tyre_age", "Compound", "lap_delta"]].head(10))
