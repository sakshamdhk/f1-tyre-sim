import pandas as pd

df = pd.read_csv("data/clean_laps.csv")
comp = pd.get_dummies(df.Compound, dtype=int).rename(columns=str.lower)
t = pd.DataFrame({
    "Race": df.Race, "Driver": df.Driver, "LapNumber": df.LapNumber,
    "tyre_age": df.tyre_age,
    "track_temp": df.TrackTemp, "air_temp": df.AirTemp, "humidity": df.Humidity,
    "stint_number": df.Stint,
    "lap_delta": df.lap_delta,
})
t = pd.concat([t, comp], axis=1)
for c in ["soft", "medium", "hard"]:
    t[f"age_{c}"] = t.tyre_age * t[c]
t.to_csv("data/model_table.csv", index=False)
print(t.shape)
print(t.groupby("Race").track_temp.mean().sort_values().round(1))
