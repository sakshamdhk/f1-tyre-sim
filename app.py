"""Streamlit interface.   streamlit run app.py"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1sim import strategy as S
from f1sim.models import MODELS

st.set_page_config(page_title="F1 Tyre Strategy Simulator", page_icon="🏁", layout="wide")

BLUE, ORANGE, AQUA, VIOLET, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9a9893"
COMP_COLOR = {"soft": BLUE, "medium": ORANGE, "hard": AQUA}
SHORT = {"soft": "S", "medium": "M", "hard": "H"}
LABEL = {m: d for m, (d, _) in MODELS.items()}


def to_minsec(x):
    return f"{int(x // 60)}:{x % 60:06.3f}" if x >= 60 else f"{x:.3f}"


def fmt(comps, stops):
    return "-".join(SHORT[c] for c in comps) + ":" + ",".join(map(str, stops))


def layout(fig, title=None, height=380, **kw):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2), **kw)
    fig.update_xaxes(showgrid=False)
    return fig


@st.cache_data
def race_table():
    r = S.races()
    return r[r.used].sort_index()


@st.cache_resource
def model(name, rid):
    return S.load_model(name, rid)


@st.cache_resource
def stint_costs(name, rid, override=None, member=None):
    return S.StintCosts(model(name, rid), rid, member=member, override=dict(override) if override else None)


def race_picker(key, default="2024 Italian Grand Prix"):
    r = race_table()
    names = r.Race.tolist()
    name = st.selectbox("Race", names, index=names.index(default) if default in names else 0, key=key)
    return r.index[names.index(name)]


def model_picker(key, default="degnet"):
    keys = list(MODELS)
    return st.selectbox("Tyre model", keys, index=keys.index(default), format_func=lambda m: LABEL[m], key=key)


# ---------------------------------------------------------------------------------------------
def page_strategy():
    st.title("Strategy simulator")
    st.caption("Plan a race on any strategy and compare it with the model's optimum and with what teams did. "
               "The tyre model used for a race never saw that circuit during training.")
    c1, c2 = st.columns(2)
    with c1:
        rid = race_picker("s_race")
    with c2:
        name = model_picker("s_model")
    race = race_table().loc[rid]
    sc = stint_costs(name, rid)
    ranked = S.best_strategies(sc, race.pit_loss)
    best = ranked[0]
    n = int(race.total_laps)

    mode = st.radio("Strategy", ["Model-optimal", "Custom"], horizontal=True)
    if mode == "Custom":
        k = st.radio("Stops", [1, 2], horizontal=True)
        cols = st.columns(k + 1)
        comps = tuple(cols[i].selectbox(f"Stint {i + 1}", list(SHORT), index=[1, 2, 0][i % 3], key=f"c{i}")
                      for i in range(k + 1))
        if k == 1:
            stops = (st.slider("Pit lap", 8, n - 8, min(n - 8, max(8, int(race.median_stop or n // 2)))),)
        else:
            p = st.slider("Pit laps", 8, n - 8, (n // 3, 2 * n // 3))
            stops = (p[0], p[1]) if p[1] - p[0] >= 8 else (p[0], min(n - 8, p[0] + 8))
        if not S.legal(comps):
            st.error("A dry race must use at least two different compounds.")
            return
    else:
        comps, stops = best["comps"], best["stops"]

    window = S.pit_window(sc, comps[0], comps[1], race.pit_loss)[1:] if len(comps) == 2 else None
    laps = pd.DataFrame(S.simulate(rid, comps, stops, sc, window=window))
    t = S.evaluate_strategy(sc, comps, stops, race.pit_loss)

    m = st.columns(4)
    m[0].metric("Race time", to_minsec(laps.total.iloc[-1]))
    m[1].metric("Strategy", fmt(comps, stops), f"{t - best['time']:+.1f} s vs optimal", delta_color="inverse")
    m[2].metric("Tyre wear cost", f"{laps.deg.sum():.1f} s")
    m[3].metric("Pit loss (measured)", f"{race.pit_loss:.1f} s per stop")

    fig = go.Figure()
    if window:
        fig.add_vrect(x0=window[0] - 0.5, x1=window[1] + 0.5, fillcolor=BLUE, opacity=0.08, line_width=0,
                      annotation_text="pit window", annotation_position="top left")
    racing = laps.lap_time - laps.pit
    for c in comps:
        seg = laps[laps.compound == c]
        fig.add_scatter(x=seg.lap, y=racing[seg.index], mode="markers", name=c.capitalize(),
                        marker=dict(color=COMP_COLOR[c], size=7))
    for p in stops:
        fig.add_vline(x=p + 0.5, line_dash="dot", line_color=MUTED)
    if race.n_one_stop >= 5:
        a_comps, a_stop = tuple(race.pair.split("-")), (int(round(race.median_stop)),)
        actual = pd.DataFrame(S.simulate(rid, a_comps, a_stop, sc))
        fig.add_scatter(x=actual.lap, y=actual.lap_time - actual.pit, mode="lines", name="Teams' strategy",
                        line=dict(color=MUTED, width=1.5))
    layout(fig, "Lap time (excluding pit-lane time)", yaxis_title="Lap time (s)", xaxis_title="Lap")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Where the time goes")
        fig = go.Figure()
        fig.add_bar(x=laps.lap, y=laps.deg, name="Tyre wear", marker_color=ORANGE)
        fig.add_bar(x=laps.lap, y=laps.fuel, name="Fuel burn", marker_color=AQUA)
        fig.add_bar(x=laps.lap, y=laps.track, name="Track evolution", marker_color=VIOLET)
        layout(fig, None, height=300, barmode="relative", yaxis_title="s vs base pace", xaxis_title="Lap")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Best strategies")
        rows = [dict(Strategy=fmt(s["comps"], s["stops"]), Stops=len(s["stops"]),
                     **{"vs best (s)": round(s["time"] - best["time"], 1)}) for s in ranked[:8]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if race.n_one_stop >= 5:
            a_comps, a_stop = tuple(race.pair.split("-")), (int(round(race.median_stop)),)
            ta = S.evaluate_strategy(sc, a_comps, a_stop, race.pit_loss)
            note = " Most of these stops came under a safety car." if race.sc_stop_frac >= 0.5 else ""
            st.info(f"Teams did **{fmt(a_comps, a_stop)}** (median of {race.n_one_stop} one-stoppers): "
                    f"{ta - best['time']:+.1f} s vs the model's optimum.{note}")
    with st.expander("Lap-by-lap table"):
        show = laps.assign(lap_time=laps.lap_time.map(to_minsec), total=laps.total.map(to_minsec))
        st.dataframe(show.round(3), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------------------------
def page_whatif():
    st.title("What if the weather changed?")
    st.caption("Move the sliders and the physics-informed network re-predicts tyre wear for this circuit. "
               "The band is the spread across the 5 networks of the ensemble (model uncertainty).")
    c1, c2 = st.columns([1, 2])
    with c1:
        rid = race_picker("w_race")
        race = race_table().loc[rid]
        w = S.weather().loc[rid]
        tt = st.slider("Track temperature (°C)", 10, 60, int(round(w.track_temp.median())))
        at = st.slider("Air temperature (°C)", 5, 45, int(round(w.air_temp.median())))
        hu = st.slider("Humidity (%)", 10, 95, int(round(w.humidity.median())))
        pair = st.selectbox("One-stop compounds", ["medium-hard", "soft-hard", "soft-medium", "hard-medium"],
                            index=0)
    override = (("track_temp", tt), ("air_temp", at), ("humidity", hu))
    ens = model("degnet", rid)
    n = int(race.total_laps)
    ages = np.arange(1, 41)
    with c2:
        fig = go.Figure()
        for c in ["soft", "medium", "hard"]:
            curves = np.array([stint_costs("degnet", rid, override, k).deg[c, 1, 1][:40]
                               for k in range(len(ens.members))])
            lim = S.limits()[c]
            mean, lo, hi = curves.mean(0), curves.min(0), curves.max(0)
            col = COMP_COLOR[c]
            fig.add_scatter(x=np.r_[ages, ages[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor=col,
                            opacity=0.18, line_width=0, showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=ages[:lim], y=mean[:lim], name=c.capitalize(), line=dict(color=col, width=2.5))
            fig.add_scatter(x=ages[lim - 1:], y=mean[lim - 1:], line=dict(color=col, width=1.5, dash="dot"),
                            showlegend=False, hoverinfo="skip")
        layout(fig, "Predicted time lost to tyre wear", yaxis_title="s per lap vs fresh tyre",
               xaxis_title="Tyre age (laps)  ·  dotted: longer than any stint in the data")
        st.plotly_chart(fig, width="stretch")
    a, b = pair.split("-")
    base_best = S.pit_window(stint_costs("degnet", rid), a, b, race.pit_loss)
    new_best = S.pit_window(stint_costs("degnet", rid, override), a, b, race.pit_loss)
    m = st.columns(3)
    m[0].metric("Optimal pit lap, real weather", base_best[0], f"window {base_best[1]}-{base_best[2]}",
                delta_color="off")
    m[1].metric("Optimal pit lap, your weather", new_best[0], f"{new_best[0] - base_best[0]:+d} laps")
    m[2].metric("Race distance", f"{n} laps")


# ---------------------------------------------------------------------------------------------
def page_explorer():
    st.title("Race explorer")
    st.caption("Observed tyre degradation (average over all drivers at each tyre age) against every model's "
               "prediction for that race. Every prediction comes from a model that never saw the circuit.")
    c = pd.read_csv("results/curves_cv.csv")
    r = race_table()
    c1, c2 = st.columns(2)
    with c1:
        rid = race_picker("e_race")
    sub = c[c.RaceId == rid]
    with c2:
        comps = sorted(sub.Compound.unique(), key=["SOFT", "MEDIUM", "HARD"].index)
        comp = st.radio("Compound", comps, horizontal=True)
    g = sub[sub.Compound == comp]
    picks = st.multiselect("Models", list(MODELS), default=["baseline", "rf", "degnet"], format_func=lambda m: LABEL[m])
    fig = go.Figure()
    fig.add_scatter(x=g.tyre_age, y=g.deg, mode="markers", name="Observed",
                    marker=dict(color="#0b0b0b", size=np.clip(g.n, 4, 14), opacity=0.6))
    palette = [BLUE, ORANGE, AQUA, VIOLET, "#e87ba4", "#008300", "#e34948"]
    for m, col in zip(picks, palette):
        fig.add_scatter(x=g.tyre_age, y=g[m], mode="lines", name=LABEL[m], line=dict(color=col))
    layout(fig, f"{r.loc[rid].Race}: {comp.lower()} tyres", yaxis_title="Time lost vs stint start (s)",
           xaxis_title="Tyre age (laps)", height=440)
    st.plotly_chart(fig, width="stretch")
    st.caption("Marker size = number of laps averaged at that tyre age.")
    per = pd.read_csv("results/per_race_cv.csv", index_col="Race")
    if r.loc[rid].Race in per.index:
        st.write("Lap-level mean absolute error for this race (s):")
        st.dataframe(per.loc[[r.loc[rid].Race]].rename(columns=LABEL).round(3), width="stretch")


# ---------------------------------------------------------------------------------------------
def page_lab():
    st.title("Model lab")
    st.caption("How each model was tested and how it did. Two protocols: unseen circuits (5-fold CV grouped "
               "by circuit) and next season (train on 2023, test on 2024).")
    if not os.path.exists("results/metrics.csv"):
        st.warning("Run `python pipeline.py --from analysis` first.")
        return
    m = pd.read_csv("results/metrics.csv")
    m["model"] = m.model.map(LABEL)
    proto = st.radio("Protocol", ["cv", "temporal"], horizontal=True,
                     format_func={"cv": "Unseen circuits", "temporal": "Next season"}.get)
    show = m[m.protocol == proto][["model", "lap_mae", "lap_mae_lo", "lap_mae_hi", "lap_r2", "curve_mae",
                                   "curve_r2", "beats_baseline"]]
    show.columns = ["Model", "Lap MAE (s)", "95% CI low", "95% CI high", "Lap R²", "Curve MAE (s)", "Curve R²",
                    "P(beats baseline)"]
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    p = f"results/pit_summary_{proto}.csv"
    if os.path.exists(p):
        st.subheader("Pit-stop lap vs what teams did")
        ps = pd.read_csv(p)
        ps["model"] = ps.model.map(lambda x: {"half": "Half race distance", "oracle": "Oracle (true wear)"}.get(
            x, LABEL.get(x.replace("_cal", ""), x) + (" + strategy calibration" if x.endswith("_cal") else "")))
        ps = ps[["model", "pit_mae", "bias", "within_3"]]
        ps.columns = ["Prediction", "Mean abs error (laps)", "Bias (laps, + = too late)", "Within 3 laps"]
        st.dataframe(ps.round(2), hide_index=True, width="stretch")
    st.subheader("Figures")
    figs = sorted(f for f in os.listdir("plots") if f.endswith(".png"))
    cols = st.columns(2)
    for i, f in enumerate(figs):
        cols[i % 2].image(f"plots/{f}", width="stretch")


pg = st.navigation([st.Page(page_strategy, title="Strategy simulator", icon="🏁"),
                    st.Page(page_whatif, title="What-if weather", icon="🌡️"),
                    st.Page(page_explorer, title="Race explorer", icon="📈"),
                    st.Page(page_lab, title="Model lab", icon="🧪")])
pg.run()
