import pandas as pd

K = 0.035  # fuel effect, seconds per lap

df = pd.read_csv("data/season_2024_dry.csv", dtype={"TrackStatus": str})

def step(name, d):
    print(f"{name:<26}{len(d):>7}")
    return d

df = step("raw", df)
df = step("drop Monaco", df[~df.Race.isin(["Monaco Grand Prix", "Canadian Grand Prix"])])
df = step("drop pit in/out laps", df[df.PitInTime.isna() & df.PitOutTime.isna()])
df = step("drop inaccurate", df[df.IsAccurate == True])
df = step("drop NaN lap time", df[df.LapTimeS.notna()])
df = step("green flag only", df[df.TrackStatus == "1"])
df = step("drop lap 1", df[df.LapNumber > 1])
df = step("slick compounds only", df[df.Compound.isin(["SOFT", "MEDIUM", "HARD"])])

df["corrected"] = df.LapTimeS + K * df.LapNumber
med = df.groupby("Race").corrected.transform("median")
df = step("drop >107% of median", df[df.corrected <= 1.07 * med])
df["lap_delta"] = df.corrected - df.groupby(["Race", "Driver"]).corrected.transform("min")
df = df.rename(columns={"TyreLife": "tyre_age"})

w = pd.read_csv("data/season_2024_weather.csv").drop_duplicates(["Driver", "LapNumber", "Race"])
n = len(df)
df = df.merge(w, on=["Driver", "LapNumber", "Race"], how="left")
assert len(df) == n, "weather merge changed the row count"
print(df[["AirTemp", "TrackTemp", "Humidity"]].isna().sum())
df = step("drop missing weather", df.dropna(subset=["AirTemp", "TrackTemp", "Humidity"]))

df.to_csv("data/clean_laps.csv", index=False)
print(df[["tyre_age", "lap_delta", "TrackTemp"]].describe())
