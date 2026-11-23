import pandas as pd

df = pd.read_csv("data/season_2024_dry.csv")
total_laps = df.groupby("Race").LapNumber.max().rename("total_laps")

def driver_summary(g):
    pit_laps = sorted(g.loc[g.PitInTime.notna(), "LapNumber"].astype(int).tolist())
    compounds = g.groupby("Stint").Compound.first().tolist()
    return pd.Series({
        "laps_done": g.LapNumber.max(),
        "n_stops": len(pit_laps),
        "first_stop": pit_laps[0] if pit_laps else None,
        "compounds": tuple(compounds),
    })

drivers = df.groupby(["Race", "Driver"]).apply(driver_summary).reset_index()
drivers = drivers.merge(total_laps.reset_index(), on="Race")

# finished (>=90% of race laps) and exactly one stop
one = drivers[(drivers.laps_done >= 0.9 * drivers.total_laps) & (drivers.n_stops == 1)]

agg = one.groupby("Race").agg(
    n_one_stop=("Driver", "count"),
    median_first_stop=("first_stop", "median"),
    compound_pair=("compounds", lambda s: s.value_counts().idxmax()),
).reset_index()

truth = total_laps.reset_index().merge(agg, on="Race", how="left")
truth["n_one_stop"] = truth["n_one_stop"].fillna(0).astype(int)
truth["comparable"] = truth.n_one_stop >= 5
truth.loc[truth.Race == "Monaco Grand Prix", "comparable"] = False
truth.loc[truth.Race == "Saudi Arabian Grand Prix", "comparable"] = False
truth.to_csv("data/pit_truth.csv", index=False)

pd.set_option("display.width", 200)
print(truth.to_string())
print(drivers[(drivers.Race == "Bahrain Grand Prix") & (drivers.Driver == "VER")])