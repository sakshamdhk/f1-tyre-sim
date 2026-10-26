import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache")

df = pd.read_csv("data/season_2024_dry.csv")
races = df["Race"].unique()

frames = []
for gp in races:
    session = fastf1.get_session(2024, gp, "R")
    session.load(telemetry=False, messages=False)

    weather = session.weather_data[["Time", "AirTemp", "TrackTemp", "Humidity"]].sort_values("Time")
    laps = session.laps[["Driver", "LapNumber", "Time"]].dropna(subset=["Time"]).sort_values("Time")

    m = pd.merge_asof(laps, weather, on="Time", direction="nearest")
    m["Race"] = gp
    frames.append(m[["Driver", "LapNumber", "Race", "AirTemp", "TrackTemp", "Humidity"]])
    print(f"{gp}: {len(m)} laps")

out = pd.concat(frames, ignore_index=True)
out.to_csv("data/season_2024_weather.csv", index=False)
print(out.shape)
print(out.isna().sum())


