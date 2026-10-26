import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
df = pd.read_csv("data/season_2024_dry.csv")
d = df[(df.Race == "Bahrain Grand Prix") & (df.Driver == "VER")]
print(d[d.PitInTime.notna() | d.PitOutTime.notna()][["LapNumber", "Stint", "Compound", "PitInTime", "PitOutTime"]])
