import pandas as pd

d = pd.read_csv("data/season_2024_dry.csv")
d = d[d.PitInTime.notna()]
res = pd.read_csv("data/pit_results.csv")
for race in res.Race:
    c = d[d.Race == race].LapNumber.astype(int).value_counts()
    print(f"{race:<28} busiest pit lap {c.index[0]:>3} has {c.iloc[0] / c.sum():.0%} of all stops")
