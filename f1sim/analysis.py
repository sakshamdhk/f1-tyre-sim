"""Everything that turns trained models into results: metrics with confidence intervals,
pit-stop evaluation, feature importance, the temperature effect and learning curves.
Results are written to results/ as CSV/JSON; plots.py draws them.

    python -m f1sim.analysis              # all analyses
    python -m f1sim.analysis metrics pit  # some of them
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from f1sim.common import SEED
from f1sim.features import COND, LAP
from f1sim.models import MODELS
from f1sim import strategy as S

OUT = "results"
N_BOOT = 2000


def _laps(protocol):
    d = pd.read_csv("data/laps.csv")
    p = pd.read_csv(f"data/preds_{protocol}.csv")
    names = [m for m in MODELS if m in p]
    for m in names:
        d[m] = p[m].to_numpy()
    if protocol == "temporal":
        d = d[d.Year == 2024].reset_index(drop=True)
    return d, names


def curve_table(d, names, min_n=5):
    """Average observed and predicted deg per (race, compound, tyre age): lap noise averages out
    and what is left is the degradation curve itself."""
    b = d.groupby(["RaceId", "Compound", "tyre_age"]).agg(n=("deg", "size"), deg=("deg", "mean"),
                                                         **{m: (m, "mean") for m in names})
    return b[b.n >= min_n].reset_index()


def bootstrap_by_race(err, race, n=N_BOOT):
    """95% CI of the mean absolute error, resampling whole races (laps of a race are correlated)."""
    rng = np.random.default_rng(SEED)
    per = pd.DataFrame({"e": err, "r": race}).groupby("r").e.agg(["sum", "size"])
    idx = rng.integers(0, len(per), (n, len(per)))
    s, c = per["sum"].to_numpy()[idx].sum(1), per["size"].to_numpy()[idx].sum(1)
    return np.percentile(s / c, [2.5, 97.5])


def paired_win_rate(err_a, err_b, race, n=N_BOOT):
    """Share of race-bootstrap resamples in which model a has the lower MAE than model b."""
    rng = np.random.default_rng(SEED)
    per = pd.DataFrame({"a": err_a, "b": err_b, "r": race}).groupby("r")[["a", "b"]].sum()
    cnt = pd.Series(race).value_counts().reindex(per.index).to_numpy()
    idx = rng.integers(0, len(per), (n, len(per)))
    a = per.a.to_numpy()[idx].sum(1) / cnt[idx].sum(1)
    b = per.b.to_numpy()[idx].sum(1) / cnt[idx].sum(1)
    return float((a < b).mean())


def metrics():
    rows = []
    for protocol in ["cv", "temporal"]:
        d, names = _laps(protocol)
        c = curve_table(d, names)
        for m in names:
            e = (d[m] - d.deg).abs()
            lo, hi = bootstrap_by_race(e.to_numpy(), d.RaceId.to_numpy())
            ce = (c[m] - c.deg).abs()
            rows.append(dict(protocol=protocol, model=m, description=MODELS[m][0],
                             lap_mae=e.mean(), lap_mae_lo=lo, lap_mae_hi=hi,
                             lap_rmse=float(np.sqrt(((d[m] - d.deg) ** 2).mean())), lap_r2=r2_score(d.deg, d[m]),
                             curve_mae=ce.mean(), curve_r2=r2_score(c.deg, c[m]),
                             beats_baseline=paired_win_rate(e.to_numpy(), (d.baseline - d.deg).abs().to_numpy(),
                                                            d.RaceId.to_numpy())))
        per_race = d.groupby("Race").apply(lambda g: pd.Series({m: (g[m] - g.deg).abs().mean() for m in names}),
                                          include_groups=False)
        per_race.round(4).to_csv(f"{OUT}/per_race_{protocol}.csv")
        c.to_csv(f"{OUT}/curves_{protocol}.csv", index=False)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(f"{OUT}/metrics.csv", index=False)
    print(res[["protocol", "model", "lap_mae", "lap_mae_lo", "lap_mae_hi", "lap_r2", "curve_mae", "curve_r2",
               "beats_baseline"]].round(3).to_string(index=False))


def oracle_pit(rid):
    """Tyre-optimal pit lap using the race's own measured wear rates (median stint slope per
    compound): the best any tyre model could do. Its error is the floor for the pit metric."""
    r, s = S.races().loc[rid], pd.read_csv("data/stints.csv")
    a, b = r.pair.split("-")
    n, g, off, lim = int(r.total_laps), s[s.RaceId == rid], S.offsets(), S.limits()
    ra, rb = g[g.compound == a].slope.median(), g[g.compound == b].slope.median()
    ra, rb = (ra if np.isfinite(ra) else rb), (rb if np.isfinite(rb) else ra)
    ps = np.arange(8, n - 7)
    q = n - ps
    cost = off[a] * ps + ra * ps * (ps - 1) / 2 + off[b] * q + rb * q * (q - 1) / 2 + 0 * r.pit_loss
    cost = np.where((ps <= lim[a]) & (q <= lim[b]), cost, np.inf)
    return int(ps[np.argmin(cost)])


pit_cache = {}


def pit(names=None, protocol="cv"):
    """Model-optimal pit lap vs what teams did, on races where >= 5 drivers made one stop."""
    r = S.races()
    suffix = "0" if protocol == "cv" else "2024"
    names = names or [m for m in MODELS if os.path.exists(f"models/{protocol}/{m}_{suffix}.joblib")]
    rows = []
    for rid, race in r[r.comparable].iterrows():
        if protocol == "temporal" and race.Year != 2024:
            continue
        a, b = race.pair.split("-")
        rec = dict(RaceId=rid, Race=race.Race, pair=race.pair, laps=int(race.total_laps),
                   actual=race.median_stop, actual_stops=race.median_stops, pit_loss=race.pit_loss,
                   half=int(race.total_laps) // 2)
        for m in names:
            sc = S.StintCosts(S.load_model(m, rid, protocol), rid)
            best, lo, hi = S.pit_window(sc, a, b, race.pit_loss)
            pit_cache.setdefault(m, {})[rid] = best
            top = S.best_strategies(sc, race.pit_loss)[0]
            rec.update({m: best, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_stops": len(top["stops"]),
                        f"{m}_best": "-".join(c[0].upper() for c in top["comps"]) + ":" +
                                     ",".join(map(str, top["stops"]))})
            if m == "degnet":       # each ensemble member's own optimum: the model's uncertainty
                ens = S.load_model(m, rid, protocol)
                spread = [S.pit_window(S.StintCosts(ens, rid, member=k), a, b, race.pit_loss)[0]
                          for k in range(len(ens.members))]
                rec["degnet_ens_lo"], rec["degnet_ens_hi"] = min(spread), max(spread)
        rows.append(rec)
        print(f"  {race.Race:<36} actual {race.median_stop:5.1f}  " +
              "  ".join(f"{m} {rec[m]}" for m in names), flush=True)
    res = pd.DataFrame(rows)
    res["oracle"] = [oracle_pit(rid) for rid in res.RaceId]
    # Strategy calibration: teams stop earlier than the tyre-optimal lap (undercut pressure).
    # Learn that shift, as a fraction of race distance, from the *training* races only.
    res["fold"] = res.RaceId.map(r.fold)
    # Only for cv: under the temporal protocol just 5 green-flag 2023 races exist to learn it from.
    train_sets = {k: r[r.comparable & (r.fold != k)].index for k in res.fold.unique()}
    for m in (names if protocol == "cv" else []):
        shift = {}
        for k, ids in train_sets.items():
            tr = [x for x in ids if x in pit_cache.get(m, {})]
            shift[k] = np.median([(pit_cache[m][x] - r.loc[x].median_stop) / r.loc[x].total_laps for x in tr])
        res[f"{m}_cal"] = (res[m] - res.fold.map(shift) * res.laps).round()
    res.to_csv(f"{OUT}/pit_{protocol}.csv", index=False)
    summ = []
    for m in ["half", "oracle"] + names + [f"{m}_cal" for m in names if f"{m}_cal" in res]:
        e = res[m] - res.actual
        summ.append(dict(model=m, pit_mae=e.abs().mean(), bias=e.mean(), within_3=(e.abs() <= 3).mean(),
                         in_window=((res.actual >= res[f"{m}_lo"]) & (res.actual <= res[f"{m}_hi"])).mean()
                         if f"{m}_lo" in res else np.nan,
                         stops_match=(res[f"{m}_stops"] == res.actual_stops.round()).mean()
                         if f"{m}_stops" in res else np.nan))
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{OUT}/pit_summary_{protocol}.csv", index=False)
    print(f"\n{len(res)} comparable races ({protocol})")
    print(summ.round(2).to_string(index=False))


def importance(names=("gbm", "degnet"), n_rep=3):
    """Permutation importance on the cross-validation test folds: how much the test MAE grows
    when one feature's values are shuffled across laps."""
    import joblib
    d = pd.read_csv("data/laps.csv")
    groups = {"compound": ["soft", "medium", "hard"], "tyre age": ["tyre_age", "stint_lap", "stint_start_age"],
              "track temp": ["track_temp"], "air temp": ["air_temp"], "humidity": ["humidity"],
              "wind": ["wind_speed"], "corners": ["corners", "corner_density"], "top speed": ["top_speed"],
              "lap time": ["ref_lap"], "race length": ["total_laps"], "stint number": ["stint_number"],
              "traffic": ["in_traffic"]}
    rng = np.random.default_rng(SEED)
    rows = []
    for m in names:
        for k in range(5):
            model = joblib.load(f"models/cv/{m}_{k}.joblib")
            te = d[d.fold == k].reset_index(drop=True)
            base = np.abs(model.predict(te) - te.deg).mean()
            for g, cols in groups.items():
                for _ in range(n_rep):
                    x = te.copy()
                    perm = rng.permutation(len(x))
                    x[cols] = x[cols].to_numpy()[perm]
                    rows.append(dict(model=m, fold=k, feature=g,
                                     increase=np.abs(model.predict(x) - x.deg).mean() - base))
    res = pd.DataFrame(rows).groupby(["model", "feature"]).increase.agg(["mean", "std"]).reset_index()
    res.to_csv(f"{OUT}/importance.csv", index=False)
    print(res.sort_values(["model", "mean"], ascending=[True, False]).round(4).to_string(index=False))


def temperature_effect():
    """Wear rate the DegNet ensemble predicts as track temperature varies (other conditions at
    their median), per compound, next to the observed per-stint slopes."""
    import joblib
    d = pd.read_csv("data/laps.csv")
    med = d[COND].median()
    temps = np.linspace(d.track_temp.quantile(0.02), d.track_temp.quantile(0.98), 30)
    rows = []
    for k in range(5):
        ens = joblib.load(f"models/cv/degnet_{k}.joblib")
        for c in ["soft", "medium", "hard"]:
            X = pd.DataFrame([med] * len(temps))
            X["track_temp"] = temps
            X["air_temp"] = med.air_temp + (temps - med.track_temp) * 0.45   # air follows track temp
            for cc in ["soft", "medium", "hard"]:
                X[cc] = int(cc == c)
            for j, mem in enumerate(ens.members):
                rate, _ = mem.curve_params(X)
                rows += [dict(fold=k, member=j, compound=c, track_temp=t, wear=w) for t, w in zip(temps, rate)]
    res = pd.DataFrame(rows).groupby(["compound", "track_temp"]).wear.agg(
        mean="mean", lo=lambda s: s.quantile(0.1), hi=lambda s: s.quantile(0.9)).reset_index()
    res.to_csv(f"{OUT}/temperature_effect.csv", index=False)
    s = pd.read_csv("data/stints.csv")
    for c, g in s.groupby("compound"):
        print(f"  {c:<7} observed slope vs track temp: r = {g.slope.corr(g.track_temp):+.2f} ({len(g)} stints)")
    print(res.groupby("compound").apply(lambda g: g["mean"].iloc[-1] - g["mean"].iloc[0], include_groups=False)
          .rename("predicted wear change, coolest->hottest (s/lap)").round(4).to_string())


def learning_curve(names=("ridge", "gbm", "degnet"), fracs=(0.25, 0.5, 0.75, 1.0)):
    """Test MAE as the number of training circuits grows: does more data still help?"""
    d = pd.read_csv("data/laps.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(5):
        tr_all, te = d[d.fold != k], d[d.fold == k]
        circ = rng.permutation(tr_all.CircuitKey.unique())
        for f in fracs:
            keep = circ[:max(2, round(len(circ) * f))]
            tr = tr_all[tr_all.CircuitKey.isin(keep)]
            for m in names:
                model = MODELS[m][1]()
                model.final = {**model.final, "n": 1} if m == "degnet" else model.final
                model.fit(tr, tr.deg, groups=tr.CircuitKey)
                rows.append(dict(fold=k, frac=f, n_races=tr.RaceId.nunique(), model=m,
                                 mae=np.abs(model.predict(te) - te.deg).mean()))
        print(f"  fold {k} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/learning_curve.csv", index=False)
    print(res.groupby(["model", "frac"]).agg(n_races=("n_races", "mean"), mae=("mae", "mean")).round(4).to_string())


def nn_history():
    """Training and validation loss per epoch for one fold's networks (diagnostic plot)."""
    import joblib
    rows = []
    for m in ["mlp", "degnet"]:
        model = joblib.load(f"models/cv/{m}_0.joblib")
        nets = model.members if hasattr(model, "members") else [model]
        for j, net in enumerate(nets):
            rows += [dict(model=m, member=j, epoch=i + 1, train=t, val=v) for i, (t, v) in enumerate(net.history)]
    pd.DataFrame(rows).to_csv(f"{OUT}/nn_history.csv", index=False)


TASKS = {"metrics": metrics, "pit": pit, "pit_temporal": lambda: pit(protocol="temporal"),
         "importance": importance, "temperature": temperature_effect, "learning": learning_curve,
         "history": nn_history}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for t in sys.argv[1:] or TASKS:
        print(f"\n== {t}")
        TASKS[t]()
