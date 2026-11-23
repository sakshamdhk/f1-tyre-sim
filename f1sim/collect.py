"""Download every Grand Prix (no sprints) of 2023-2024 from the OpenF1 API.

Since 2026 the official F1 timing server (which FastF1 reads) needs an F1TV subscription,
so lap data comes from OpenF1 instead. Raw API responses are cached in cache/openf1/, and
each race is written to data/raw/<year>_<round>.csv with one row per driver-lap:
compound, tyre age, pit in/out flags, safety-car/yellow flags, per-lap weather and
circuit info. Already-built races are skipped, so the script can be re-run after a failure.

    python -m f1sim.collect            # all seasons
    python -m f1sim.collect 2023       # one season
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

SEASONS = [2023, 2024]
API = "https://api.openf1.org/v1"
CACHE = "cache/openf1"
RAW = "data/raw"
MIN_GAP = 2.2   # seconds between API calls (free tier allows ~30 requests a minute)
_last = [0.0]


def get(endpoint, **params):
    key = endpoint + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    path = f"{CACHE}/{key}.json"
    if os.path.exists(path):
        return json.load(open(path))
    for attempt in range(6):
        time.sleep(max(0.0, MIN_GAP - (time.time() - _last[0])))
        _last[0] = time.time()
        r = requests.get(f"{API}/{endpoint}", params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        data = r.json() if r.status_code == 200 else []
        if isinstance(data, dict):          # {"detail": "No results found."}
            data = []
        json.dump(data, open(path, "w"))
        return data
    raise RuntimeError(f"{endpoint} {params}: still failing after retries")


def ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


def neutralised_intervals(rc, lap_starts, session_end):
    """(start, end) times of safety car, VSC and red flag periods from race-control messages.
    OpenF1 sends no green flag after a red-flag restart, so a red flag ends when the next lap
    starts (no laps start while the cars wait in the pit lane)."""
    out, start = [], None
    lap_starts = np.sort(lap_starts.dropna().to_numpy())
    for _, m in rc.sort_values("date").iterrows():
        msg = str(m.message).upper()
        if m.category == "Flag" and m.flag == "RED":
            if start is None:
                start = m.date
            later = lap_starts[lap_starts > m.date.to_datetime64()]
            out.append((start, pd.Timestamp(later[0], tz="UTC") if len(later) else session_end))
            start = None
            continue
        opens = "SAFETY CAR DEPLOYED" in msg
        closes = (("ENDING" in msg or "IN THIS LAP" in msg) and m.category == "SafetyCar") or \
                 (m.category == "Flag" and m.flag == "GREEN" and m.scope == "Track")
        if opens and start is None:
            start = m.date
        elif closes and start is not None:
            out.append((start, m.date))
            start = None
    if start is not None:
        out.append((start, session_end))
    return out


def yellow_intervals(rc):
    out, open_ = [], {}
    for _, m in rc[rc.category == "Flag"].sort_values("date").iterrows():
        key = m.sector if m.scope == "Sector" else "track"
        if m.flag in ("YELLOW", "DOUBLE YELLOW"):
            open_.setdefault(key, m.date)
        elif m.flag in ("CLEAR", "GREEN") and key in open_:
            out.append((open_.pop(key), m.date))
    out += [(t, t + pd.Timedelta(seconds=120)) for t in open_.values()]
    return out


def overlaps(start, end, intervals):
    hit = np.zeros(len(start), dtype=bool)
    for a, b in intervals:
        hit |= (start <= b) & (end >= a)
    return hit


def build_race(year, rnd, sess, meeting):
    sk = sess["session_key"]
    laps = pd.DataFrame(get("laps", session_key=sk))
    stints = pd.DataFrame(get("stints", session_key=sk))
    weather = pd.DataFrame(get("weather", session_key=sk))
    rc = pd.DataFrame(get("race_control", session_key=sk))
    drivers = pd.DataFrame(get("drivers", session_key=sk)).drop_duplicates("driver_number")
    if laps.empty or stints.empty:
        raise RuntimeError("no lap or stint data")

    laps = laps.rename(columns={"lap_number": "LapNumber", "lap_duration": "LapTimeS",
                                "st_speed": "SpeedST", "is_pit_out_lap": "PitOut"})
    laps["start"] = ts(laps.date_start)
    laps["end"] = laps.start + pd.to_timedelta(laps.LapTimeS, unit="s")

    # stint -> compound and tyre age per lap
    stints = stints.dropna(subset=["lap_start", "lap_end"]).sort_values(["driver_number", "stint_number"])
    rows = []
    for _, s in stints.iterrows():
        for lap in range(int(s.lap_start), int(s.lap_end) + 1):
            rows.append((s.driver_number, lap, int(s.stint_number), s.compound,
                         (s.tyre_age_at_start or 0) + lap - int(s.lap_start) + 1))
    per_lap = pd.DataFrame(rows, columns=["driver_number", "LapNumber", "Stint", "Compound", "TyreLife"])
    per_lap = per_lap.drop_duplicates(["driver_number", "LapNumber"], keep="last")
    laps = laps.merge(per_lap, on=["driver_number", "LapNumber"], how="left")

    # a pit-in lap is the last lap of any stint that is followed by another stint
    last = stints.groupby("driver_number").stint_number.transform("max")
    pit_in = stints[stints.stint_number < last][["driver_number", "lap_end"]]
    pit_in = set(zip(pit_in.driver_number, pit_in.lap_end.astype(int)))
    laps["PitIn"] = [(d, l) in pit_in for d, l in zip(laps.driver_number, laps.LapNumber)]

    # race control: neutralised (SC/VSC/red) and yellow-flag laps
    session_end = ts(sess["date_end"])
    if not rc.empty:
        rc["date"] = ts(rc.date)
        for c in ["flag", "scope", "sector", "category", "message"]:
            if c not in rc:
                rc[c] = None
        neut = neutralised_intervals(rc, laps.start.dt.tz_convert(None), session_end)
        yel = yellow_intervals(rc)
    else:
        neut, yel = [], []
    s0, e0 = laps.start.fillna(laps.end), laps.end.fillna(laps.start)
    laps["Neutralised"] = overlaps(s0, e0, neut)
    # the lap after a neutralisation ends is a restart lap on cold tyres
    after = [(b, b + pd.Timedelta(seconds=100)) for _, b in neut]
    laps["Restart"] = overlaps(s0, e0, after) & ~laps.Neutralised
    laps["Yellow"] = overlaps(s0, e0, yel)

    # weather at the middle of each lap
    weather["date"] = ts(weather.date)
    weather = weather.sort_values("date")[["date", "air_temperature", "track_temperature",
                                           "humidity", "rainfall", "wind_speed"]]
    laps["mid"] = s0 + (e0 - s0) / 2
    laps = laps.dropna(subset=["mid"]).sort_values("mid")
    laps = pd.merge_asof(laps, weather, left_on="mid", right_on="date", direction="nearest")

    laps = laps.merge(drivers[["driver_number", "name_acronym", "team_name"]], on="driver_number", how="left")
    try:
        from fastf1.mvapi import get_circuit_info
        corners = len(get_circuit_info(year=year, circuit_key=sess["circuit_key"]).corners)
    except Exception:
        corners = np.nan
    laps["SessionTimeS"] = (laps.end - ts(sess["date_start"])).dt.total_seconds()

    out = pd.DataFrame({
        "Year": year, "Round": rnd, "Race": meeting["meeting_name"], "Location": sess["circuit_short_name"],
        "CircuitKey": sess["circuit_key"], "Corners": corners, "TotalLaps": int(laps.LapNumber.max()),
        "Driver": laps.name_acronym, "Team": laps.team_name, "LapNumber": laps.LapNumber,
        "LapTimeS": laps.LapTimeS, "SessionTimeS": laps.SessionTimeS, "Stint": laps.Stint,
        "Compound": laps.Compound, "TyreLife": laps.TyreLife, "PitIn": laps.PitIn,
        "PitOut": laps.PitOut.fillna(False).astype(bool), "Neutralised": laps.Neutralised,
        "Restart": laps.Restart, "Yellow": laps.Yellow, "SpeedST": laps.SpeedST,
        "AirTemp": laps.air_temperature, "TrackTemp": laps.track_temperature,
        "Humidity": laps.humidity, "Rainfall": laps.rainfall, "WindSpeed": laps.wind_speed,
    })
    return out.sort_values(["Driver", "LapNumber"])


def main(seasons):
    import logging
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    for year in seasons:
        meetings = {m["meeting_key"]: m for m in get("meetings", year=year)}
        sessions = sorted(get("sessions", year=year, session_name="Race"), key=lambda s: s["date_start"])
        sessions = [s for s in sessions if not s.get("is_cancelled")]
        for rnd, sess in enumerate(sessions, start=1):
            path = f"{RAW}/{year}_{rnd:02d}.csv"
            name = meetings.get(sess["meeting_key"], {}).get("meeting_name", sess["location"])
            if os.path.exists(path):
                continue
            try:
                df = build_race(year, rnd, sess, meetings[sess["meeting_key"]])
                df.to_csv(path, index=False)
                print(f"{year} R{rnd:02d} {name:<28} {len(df):>5} laps  "
                      f"rain {df.Rainfall.mean():.0%}  neutralised {df.Neutralised.mean():.0%}", flush=True)
            except Exception as e:
                print(f"{year} R{rnd:02d} {name:<28} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or SEASONS)
