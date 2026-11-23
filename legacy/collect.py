import fastf1
import pandas as pd
from utils import to_minsec


fastf1.Cache.enable_cache("cache")

schedule = fastf1.get_event_schedule(2024)
races = schedule[schedule["EventFormat"] != "testing"]["EventName"].tolist()

COLS = ["Driver", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife",
        "FreshTyre", "TrackStatus", "PitInTime", "PitOutTime", "IsAccurate"]

frames = []
for gp in races:
    try:
        session = fastf1.get_session(2024, gp, "R")
        session.load(telemetry=False, messages=False)
        laps = session.laps[COLS].copy()
        laps["Race"] = gp
        laps["Year"] = 2024

        weather = session.weather_data
        laps["Rainfall"] = weather["Rainfall"].mode()[0] if not weather.empty else False

        frames.append(laps)
        print(f"{gp}: {len(laps)} laps, rain={laps['Rainfall'].iloc[0]}")
    except Exception as e:
        print(f"{gp}: FAILED - {e}")

df = pd.concat(frames, ignore_index=True)
df["LapTimeS"] = df["LapTime"].dt.total_seconds()
df.to_csv("data/season_2024_laps.csv", index=False)
print(df.shape)