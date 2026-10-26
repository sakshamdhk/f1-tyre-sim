import pandas as pd

K = 0.035  # fuel effect, seconds per lap

df = pd.read_csv("data/season_2024_dry.csv", dtype={"TrackStatus": str})

def step(name, d):
    print(f"{name:<26}{len(d):>7}")
    return d

df = step("raw", df)
df = step("drop pit in/out laps", df[df.PitInTime.isna() & df.PitOutTime.isna()])
df = step("drop inaccurate", df[df.IsAccurate == True])
df = step("drop NaN lap time", df[df.LapTimeS.notna()])
df = step("green flag only", df[df.TrackStatus == "1"])
df = step("drop lap 1", df[df.LapNumber > 1])
df = step("slick compounds only", df[df.Compound.isin(["SOFT", "MEDIUM", "HARD"])])

df["corrected"] = df.LapTimeS + K * df.LapNumber
med = df.groupby("Race").corrected.transform("median")
df = step("drop >107% of median", df[df.corrected <= 1.07 * med])
df["lap_delta"] = df.corrected - df.groupby("Race").corrected.transform("min")
df = df.rename(columns={"TyreLife": "tyre_age"})

df.to_csv("data/clean_laps.csv", index=False)
print(df[["tyre_age", "lap_delta"]].describe())



