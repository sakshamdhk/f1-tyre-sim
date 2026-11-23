"""Lap-by-lap race simulator in the terminal.

    python cli.py --list-races
    python cli.py --race "2024 Italian" --model degnet
    python cli.py --race "2024 Italian" --model gbm --strategy M-H:20 --fast
    python cli.py --race "2024 Dutch" --strategy S-M-H:12,40 --delay 0.2
"""
import argparse
import sys
import time

from f1sim import strategy as S
from f1sim.models import MODELS

SHORT = {"soft": "S", "medium": "M", "hard": "H"}


def to_minsec(seconds):
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds >= 60:
        return f"{sign}{int(seconds // 60)}:{seconds % 60:06.3f}"
    return f"{sign}{seconds:.3f}"


def fmt_strategy(comps, stops):
    return "-".join(SHORT[c] for c in comps) + ":" + ",".join(map(str, stops))


def find_race(text):
    r = S.races()
    r = r[r.used]
    hits = r[r.Race.str.lower().str.contains(text.lower())]
    if len(hits) == 0:
        sys.exit(f"No race matches '{text}'. Try --list-races.")
    if len(hits) > 1:
        sys.exit("Several races match: " + ", ".join(hits.Race) + ". Add the year, e.g. '2024 Italian'.")
    return hits.index[0]


def list_races():
    r = S.races()
    r = r[r.used].sort_index()
    print(f"{'Race':<34}{'Laps':>5}{'Pit loss':>10}  Actual one-stop")
    for rid, x in r.iterrows():
        actual = f"{x.pair.replace('-', ' -> ')} lap {x.median_stop:g}" if x.n_one_stop >= 5 else "-"
        note = "" if x.comparable or x.n_one_stop < 5 else "  (stops under safety car)"
        print(f"{x.Race:<34}{int(x.total_laps):>5}{x.pit_loss:>9.1f}s  {actual}{note}")


def main():
    ap = argparse.ArgumentParser(description="F1 tyre degradation and pit-strategy simulator")
    ap.add_argument("--list-races", action="store_true", help="show the races you can simulate")
    ap.add_argument("--race", help="part of a race name, e.g. 'Bahrain' or '2024 Bahrain'")
    ap.add_argument("--model", default="degnet", choices=list(MODELS), help="tyre model (default degnet)")
    ap.add_argument("--strategy", help="e.g. M-H:22 or S-M-H:15,40 (default: the model-optimal strategy)")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between laps (default 0.5)")
    ap.add_argument("--fast", action="store_true", help="no delay between laps")
    a = ap.parse_args()

    if a.list_races or not a.race:
        list_races()
        return
    rid = find_race(a.race)
    race = S.races().loc[rid]
    delay = 0 if a.fast else a.delay

    print(f"\n{race.Race}  ·  {int(race.total_laps)} laps  ·  model: {MODELS[a.model][0]}")
    print(f"track {race.Location}, pit loss {race.pit_loss:.1f} s (measured from this race)")
    print("the model never saw this circuit in training\n")
    sc = S.StintCosts(S.load_model(a.model, rid), rid)
    ranked = S.best_strategies(sc, race.pit_loss)
    best = ranked[0]
    if a.strategy:
        comps, stops = S.parse_strategy(a.strategy)
        if not S.legal(comps):
            sys.exit("A dry race must use at least two different compounds.")
    else:
        comps, stops = best["comps"], best["stops"]
    window = None
    if len(comps) == 2:
        _, lo, hi = S.pit_window(sc, comps[0], comps[1], race.pit_loss)
        window = (lo, hi)
    print(f"strategy: {fmt_strategy(comps, stops)}" + (f"   pit window laps {window[0]}-{window[1]}" if window else ""))
    print(f"\n{'Lap':>4} {'Tyre':>5} {'Base':>9} {'Tyre deg':>9} {'Fuel':>8} {'Track':>7} {'Pit':>7} "
          f"{'Lap time':>10} {'Total':>12}")
    laps = S.simulate(rid, comps, stops, sc, window=window)
    for x in laps:
        mark = "  >>> PIT WINDOW <<<" if x["window"] else ""
        if x["pit"]:
            mark = "  BOX BOX"
        print(f"{x['lap']:>4} {SHORT[x['compound']]}{x['tyre_age']:>4} {to_minsec(x['base']):>9} "
              f"{x['deg']:>+9.3f} {x['fuel']:>+8.3f} {x['track']:>+7.3f} {x['pit']:>7.1f} "
              f"{to_minsec(x['lap_time']):>10} {to_minsec(x['total']):>12}{mark}", flush=True)
        if delay:
            time.sleep(delay)

    total = laps[-1]["total"]
    print("\n" + "=" * 64)
    print(f"Race time            {to_minsec(total)}")
    print(f"Tyre wear lost       {sum(x['deg'] for x in laps):.1f} s")
    print(f"Pit stops lost       {sum(x['pit'] for x in laps):.1f} s ({len(stops)} stop{'s' * (len(stops) > 1)})")
    mine = S.evaluate_strategy(sc, comps, stops, race.pit_loss)
    print(f"Model-optimal        {fmt_strategy(best['comps'], best['stops'])}   "
          f"({mine - best['time']:+.1f} s vs this strategy)" if (comps, stops) != (best["comps"], best["stops"])
          else f"Model-optimal        {fmt_strategy(comps, stops)} (this one)")
    if race.n_one_stop >= 5:
        a_comps = tuple(race.pair.split("-"))
        a_stop = (int(round(race.median_stop)),)
        t = S.evaluate_strategy(sc, a_comps, a_stop, race.pit_loss)
        sc_note = "  (most of these stops were under a safety car)" if race.sc_stop_frac >= 0.5 else ""
        print(f"Teams actually did   {fmt_strategy(a_comps, a_stop)}   ({t - best['time']:+.1f} s vs model-optimal, "
              f"median of {race.n_one_stop} one-stoppers){sc_note}")
    print("Next best strategies:")
    for s in ranked[1:4]:
        print(f"   {fmt_strategy(s['comps'], s['stops']):<14} {s['time'] - best['time']:+.1f} s")


if __name__ == "__main__":
    main()
