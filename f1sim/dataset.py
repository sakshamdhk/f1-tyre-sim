"""Turn the raw per-race files into the model-ready tables.

    python -m f1sim.dataset

Outputs (all in data/):
  races.csv        one row per race: descriptors, dry/wet, pit loss, fold, actual strategy
  laps.csv         clean laps with features and the degradation target
  stints.csv       per-stint degradation slopes (Theil-Sen) and pace at tyre age 2
  compound_offsets.json   pace of soft/medium relative to hard, from a fixed-effects regression

Target: deg = fuel-corrected lap time minus the median of the same stint's first clean laps.
Subtracting a per-stint reference removes everything that is constant within a stint
(car, driver, track, compound pace), so what is left is the time lost to tyre wear.
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.linear_model import LinearRegression

from f1sim.features import add_wear
from f1sim.common import (COMPOUNDS, EXCLUDED, STREET, N_FOLDS, PIT_LOSS_DEFAULT,
                          REF_LAPS, TRAFFIC_GAP, fuel_k)


def load_raw():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/raw/*.csv"))], ignore_index=True)
    df["RaceId"] = df.Year.astype(str) + "_" + df.Round.astype(str).str.zfill(2)
    df["Race"] = df.Year.astype(str) + " " + df.Race
    return df


def add_gap_ahead(df):
    """Seconds to the car that crossed the line just before (any lap, so lapped cars count too)."""
    df = df.sort_values(["RaceId", "SessionTimeS"])
    df["gap_ahead"] = df.groupby("RaceId").SessionTimeS.diff()
    return df


def race_table(df):
    g = df.groupby("RaceId")
    r = g.agg(Race=("Race", "first"), Year=("Year", "first"), Round=("Round", "first"),
              Location=("Location", "first"), CircuitKey=("CircuitKey", "first"),
              corners=("Corners", "first"), total_laps=("TotalLaps", "first"),
              top_speed=("SpeedST", "median"),
              rain_frac=("Rainfall", lambda s: (s > 0).mean()))
    slick = df.Compound.isin(COMPOUNDS)
    r["wet_tyre_frac"] = (~slick & df.Compound.notna()).groupby(df.RaceId).mean()
    r["dry"] = (r.rain_frac < 0.05) & (r.wet_tyre_frac < 0.05)
    r["street"] = r.Location.isin(STREET).astype(int)
    r["excluded"] = r.Location.map(EXCLUDED)
    r["used"] = r.dry & r.excluded.isna()
    return r


def pit_truth(df, races):
    """Actual strategies: per race, the one-stop drivers' median stop lap and compound pair."""
    rows = []
    for (rid, drv), g in df.groupby(["RaceId", "Driver"]):
        g = g.sort_values("LapNumber")
        stops = g.LapNumber[g.PitIn].astype(int).tolist()
        under_sc = bool(g.Neutralised[g.PitIn].iloc[0]) if stops else False
        comps = g.groupby("Stint").Compound.first().dropna().tolist()
        rows.append(dict(RaceId=rid, Driver=drv, laps_done=g.LapNumber.max(), n_stops=len(stops),
                         stops=stops, comps=tuple(comps), under_sc=under_sc))
    d = pd.DataFrame(rows).join(races[["total_laps"]], on="RaceId")
    finished = d.laps_done >= 0.9 * d.total_laps
    one = d[finished & (d.n_stops == 1)]
    out = pd.DataFrame({
        "n_one_stop": one.groupby("RaceId").size(),
        "median_stop": one.groupby("RaceId").stops.apply(lambda s: float(np.median([x[0] for x in s]))),
        "pair": one.groupby("RaceId").comps.agg(lambda s: "-".join(s.value_counts().idxmax()).lower()),
        "median_stops": d[finished].groupby("RaceId").n_stops.median(),
        # share of one-stoppers whose stop came under a safety car / VSC (a cheap stop the
        # tyre model cannot know about)
        "sc_stop_frac": one.groupby("RaceId").under_sc.mean(),
    })
    return races.join(out).assign(n_one_stop=lambda r: r.n_one_stop.fillna(0).astype(int))


def pit_loss(df, clean_ref):
    """Time lost to a stop: in-lap + out-lap minus two normal laps, median over green-flag stops."""
    d = df.sort_values(["RaceId", "Driver", "LapNumber"])
    nxt = d.groupby(["RaceId", "Driver"]).LapTimeS.shift(-1)
    nxt_out = d.groupby(["RaceId", "Driver"]).PitOut.shift(-1).fillna(False).astype(bool)
    nxt_neut = d.groupby(["RaceId", "Driver"]).Neutralised.shift(-1).fillna(True).astype(bool)
    ok = d.PitIn & nxt_out & ~d.Neutralised & ~nxt_neut & (d.LapNumber > 1)
    loss = (d.LapTimeS + nxt)[ok] - 2 * d.RaceId[ok].map(clean_ref)
    loss = loss[(loss > 10) & (loss < 45)]
    return loss.groupby(d.RaceId[ok]).median()


def track_evolution(d):
    """Per race, the lap-time trend not explained by fuel or tyres (track rubbering in, wind,
    temperature drift), in s per lap, from a fixed-effects regression:
        corrected = driver effect + compound offset + wear rate[compound] * tyre age + evo * lap.
    A per-stint effect cannot be used: within a stint, age and lap rise together, so wear and
    evolution would be perfectly collinear. With one effect per driver, the pace jump when a
    driver fits fresh tyres separates them. Races where the fit is degenerate get 0."""
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    evo = {}
    for rid, g in d.groupby("RaceId"):
        drv = pd.factorize(g.Driver)[0]
        F = sparse.csr_matrix((np.ones(len(g)), (np.arange(len(g)), drv)))
        A = np.column_stack([(g.Compound == c).to_numpy(float) for c in ("SOFT", "MEDIUM")] +
                            [(g.Compound == c).to_numpy(float) * g.TyreLife.to_numpy() for c in COMPOUNDS] +
                            [g.LapNumber.to_numpy(float)])
        A = A[:, A.std(0) > 0]
        coef = lsqr(sparse.hstack([F, sparse.csr_matrix(A)]).tocsr(), g.corrected.to_numpy(),
                    atol=1e-10, btol=1e-10)[0][-A.shape[1]:]
        ok = np.all(np.abs(coef[:-1]) < 3) and abs(coef[-1]) < 0.15
        evo[rid] = float(coef[-1]) if ok else 0.0
    return pd.Series(evo)


def clean_laps(df, races):
    def step(name, d):
        print(f"  {name:<32}{len(d):>8}")
        return d

    print("lap filters (rows left)")
    d = step("raw", df)
    d = step("dry races only", d[d.RaceId.map(races.dry)])
    d = step("excluded races (Monaco)", d[d.RaceId.map(races.used)])
    d = step("slick compounds", d[d.Compound.isin(COMPOUNDS)])
    d = step("lap time present", d[d.LapTimeS.notna()])
    d = step("drop lap 1", d[d.LapNumber > 1])
    d = step("drop pit in/out laps", d[~d.PitIn & ~d.PitOut.astype(bool)])
    d = step("drop SC/VSC/red + restart", d[~d.Neutralised & ~d.Restart])
    d = step("drop yellow-flag laps", d[~d.Yellow])
    # backup for neutralisations race control missed: laps where the whole field was slow
    field = df.groupby(["RaceId", "LapNumber"]).LapTimeS.median()
    slow = field > 1.12 * field.groupby("RaceId").transform("median")
    d = step("drop laps the whole field was slow", d[~d.set_index(["RaceId", "LapNumber"]).index.map(slow)])

    d = d.copy()
    d["k"] = d.RaceId.map(races.total_laps).map(fuel_k)
    d["corrected"] = d.LapTimeS + d.k * d.LapNumber
    # drop slow outliers (mistakes, damage, lifting): > 3 s above the driver's stint median
    med = d.groupby(["RaceId", "Driver", "Stint"]).corrected.transform("median")
    d = step("drop >3 s off stint median", d[(d.corrected - med).abs() < 3.0])
    races["evo"] = track_evolution(d)
    d["corrected"] = d.corrected - d.RaceId.map(races.evo).fillna(0) * d.LapNumber

    d = d.sort_values(["RaceId", "Driver", "LapNumber"])
    key = ["RaceId", "Driver", "Stint"]
    d["stint_start_age"] = d.groupby(key).TyreLife.transform("min")
    d["stint_lap"] = d.TyreLife - d.stint_start_age + 1
    first = d[d.stint_lap <= REF_LAPS]
    ref = first.groupby(key).corrected.median().rename("stint_ref")
    n_ref = first.groupby(key).size().rename("n_ref")
    d = d.join(ref, on=key).join(n_ref, on=key)
    d["stint_len"] = d.groupby(key).LapNumber.transform("size")
    d = step("stints with >=2 ref laps, >=8 laps", d[(d.n_ref >= 2) & (d.stint_len >= 8)])
    d["deg"] = d.corrected - d.stint_ref
    return d


def build_stints(d):
    rows = []
    for (rid, drv, st), g in d.groupby(["RaceId", "Driver", "Stint"]):
        if g.TyreLife.max() - g.TyreLife.min() < 8:
            continue
        slope, icpt = theilslopes(g.corrected, g.TyreLife)[:2]
        rows.append(dict(RaceId=rid, Driver=drv, Stint=st, compound=g.Compound.iloc[0].lower(),
                         n=len(g), slope=slope, pace2=icpt + 2 * slope, start_lap=g.LapNumber.min(),
                         start_age=g.TyreLife.min(), track_temp=g.TrackTemp.mean(),
                         air_temp=g.AirTemp.mean(), humidity=g.Humidity.mean()))
    s = pd.DataFrame(rows)
    return s[s.slope.between(-0.15, 0.5)].reset_index(drop=True)


def compound_offsets(s):
    """Pace at tyre age 2 per compound, with a fixed effect per driver-in-race and a start-lap
    term (track evolution). Hard = 0; negative means faster."""
    fx = pd.get_dummies(s.RaceId + "|" + s.Driver, dtype=float)
    X = pd.concat([fx, pd.get_dummies(s.compound, dtype=float)[["soft", "medium"]],
                   s[["start_lap"]].astype(float)], axis=1)
    X.columns = X.columns.astype(str)
    c = pd.Series(LinearRegression(fit_intercept=False).fit(X, s.pace2).coef_, index=X.columns)
    return {"soft": float(c["soft"]), "medium": float(c["medium"]), "hard": 0.0}


def assign_folds(races):
    """Folds grouped by circuit, so a track is never in both training and test data.
    Circuits are dealt round-robin after sorting by lap count, which balances the folds."""
    circ = races[races.used].groupby("CircuitKey").total_laps.mean().sort_values()
    fold = {ck: i % N_FOLDS for i, ck in enumerate(circ.index)}
    return races.CircuitKey.map(fold).where(races.used)


def main():
    df = add_gap_ahead(load_raw())
    races = race_table(df)
    print(f"{len(races)} races, {races.dry.sum()} dry: "
          + ", ".join(races[~races.dry].Race.tolist()) + " excluded as wet")

    d = clean_laps(df, races)
    races["ref_lap"] = d.groupby("RaceId").corrected.quantile(0.05)
    races["pit_loss"] = pit_loss(df, d.groupby("RaceId").LapTimeS.median())
    races["pit_loss"] = races.pit_loss.fillna(PIT_LOSS_DEFAULT)
    races = pit_truth(df, races)
    races["fold"] = assign_folds(races)
    races["comparable"] = races.used & (races.n_one_stop >= 5) & (races.sc_stop_frac < 0.5)

    d = d.drop(columns=["CircuitKey"]).join(races[["CircuitKey", "corners", "top_speed", "ref_lap", "total_laps",
                                                 "street", "fold"]], on="RaceId")
    d["stint_start_frac"] = d.groupby(["RaceId", "Driver", "Stint"]).LapNumber.transform("min") / d.total_laps
    d["corner_density"] = d.corners / d.ref_lap * 60          # corners per minute of lapping
    d["lap_frac"] = d.LapNumber / d.total_laps
    d["in_traffic"] = (d.gap_ahead < TRAFFIC_GAP).astype(int)
    for c in COMPOUNDS:
        d[c.lower()] = (d.Compound == c).astype(int)
    d = d.rename(columns={"TrackTemp": "track_temp", "AirTemp": "air_temp", "Humidity": "humidity",
                          "WindSpeed": "wind_speed", "TyreLife": "tyre_age", "Stint": "stint_number"})
    cols = ["RaceId", "Race", "Year", "CircuitKey", "Driver", "Team", "LapNumber", "Compound", "fold",
            "tyre_age", "stint_lap", "stint_start_age", "stint_number", "soft", "medium", "hard",
            "track_temp", "air_temp", "humidity", "wind_speed", "corners", "corner_density",
            "top_speed", "ref_lap", "total_laps", "street", "stint_start_frac", "lap_frac", "gap_ahead", "in_traffic",
            "LapTimeS", "corrected", "deg"]
    d = d[cols].dropna(subset=["track_temp", "air_temp", "humidity", "deg"]).reset_index(drop=True)
    d = add_wear(d)
    d.to_csv("data/laps.csv", index=False)

    s = build_stints(d.rename(columns={"tyre_age": "TyreLife", "stint_number": "Stint",
                                       "track_temp": "TrackTemp", "air_temp": "AirTemp",
                                       "humidity": "Humidity"}))
    s = s.join(races[["fold"]], on="RaceId")
    s.to_csv("data/stints.csv", index=False)
    off = compound_offsets(s)
    json.dump(off, open("data/compound_offsets.json", "w"), indent=1)
    # longest stint the optimiser may plan per compound: the 95th percentile of tyre age reached
    # in the data, so no model is asked to extrapolate beyond what it was trained on
    age = d.groupby(["RaceId", "Driver", "stint_number"]).agg(c=("Compound", "first"), a=("tyre_age", "max"))
    limits = age.groupby("c").a.quantile(0.95).round().astype(int)
    json.dump({k.lower(): int(v) for k, v in limits.items()}, open("data/compound_limits.json", "w"), indent=1)
    print("max stint length per compound:", limits.to_dict())
    # reference pace on hard tyres at full fuel, and per-lap weather for the optimiser
    st = d.groupby(["RaceId", "Driver", "stint_number"]).agg(ref=("corrected", "first"), c=("Compound", "first"))
    st["ref"] = d.groupby(["RaceId", "Driver", "stint_number"]).corrected.apply(lambda s: s.iloc[:REF_LAPS].median())
    st["base"] = st.ref - st.c.str.lower().map(off)
    races["base_pace"] = st.groupby("RaceId").base.median()
    w = df.groupby(["RaceId", "LapNumber"])[["TrackTemp", "AirTemp", "Humidity", "WindSpeed"]].median()
    w.columns = ["track_temp", "air_temp", "humidity", "wind_speed"]
    w = w.groupby("RaceId").ffill()
    w.groupby("RaceId").bfill().to_csv("data/race_weather.csv")
    races.to_csv("data/races.csv")

    print(f"\n{len(d)} clean laps, {len(s)} stints, {d.RaceId.nunique()} races, "
          f"{races[races.used].CircuitKey.nunique()} circuits")
    print("compound offsets vs hard (s):", {k: round(v, 3) for k, v in off.items()})
    print("races per fold:", races.fold.value_counts().sort_index().to_dict())
    print(f"comparable races for pit-lap error: {races.comparable.sum()}")
    print(f"pit loss: median {races.pit_loss.median():.1f} s, range "
          f"{races.pit_loss.min():.1f}-{races.pit_loss.max():.1f} s")
    print(d.groupby("Compound").deg.describe().round(2).to_string())


if __name__ == "__main__":
    main()
