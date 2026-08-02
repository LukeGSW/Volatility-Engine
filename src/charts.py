"""
charts.py — Grafici Plotly, tema scuro.

Unione dei due set originali (event study del repo KQ + gauge/validazione del
forecaster) piu' i tre grafici nuovi che rendono il segnale leggibile:

    prob_curve      la curva di probabilita' sui tre orizzonti: la sua FORMA e' gia'
                    un segnale (crescente = pressione strutturale; alta a 5 e bassa
                    a 20 = spike tattico in rientro; piatta = nessun edge)
    fan_chart       distribuzione empirica del VIX atteso, dagli analoghi storici:
                    e' il modo piu' leggibile di comunicare un forecast di volatilita'
    waterfall       da dove viene la probabilita', contributo per contributo
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config

BG = "#1E1E2E"
GRID = "#2A2A3E"
TXT = "#E0E0E0"
UP = "#EF5350"      # VIX su = rischio
DOWN = "#26A69A"    # VIX giu' = calma
ACCENT = "#2196F3"
MUTED = "#9E9E9E"


def _layout(title: str = "", x: str = "", y: str = "", h: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=TXT)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT, size=12),
        xaxis=dict(title=x, gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title=y, gridcolor=GRID, zerolinecolor=GRID),
        margin=dict(l=55, r=25, t=45 if title else 20, b=40),
        height=h, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )


# ============================================================
# [1] IL SEGNALE
# ============================================================
def prob_curve(per_horizon: dict, band: float | None = None) -> go.Figure:
    """P(VIX su) sui tre orizzonti. La forma della curva e' gia' interpretazione."""
    band = config.NEUTRAL_BAND if band is None else band
    hs = sorted(per_horizon, key=int)
    x = [int(h) for h in hs]
    y = [per_horizon[h]["p_up"] * 100 for h in hs]
    raw = [per_horizon[h]["p_up_raw"] * 100 for h in hs]

    fig = go.Figure()
    fig.add_hrect(y0=(0.5 - band) * 100, y1=(0.5 + band) * 100,
                  fillcolor=MUTED, opacity=0.15, line_width=0,
                  annotation_text="banda neutra — il modello si astiene",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=MUTED))
    fig.add_hline(y=50, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=x, y=raw, mode="lines+markers", name="grezza",
                             line=dict(color=MUTED, width=1.5, dash="dot"),
                             marker=dict(size=7)))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers+text", name="P(VIX su)",
        line=dict(color=ACCENT, width=3), marker=dict(size=13),
        text=[f"{v:.0f}%" for v in y], textposition="top center",
        textfont=dict(size=13, color=TXT)))
    fig.update_layout(**_layout("Curva di probabilita' per orizzonte",
                                "giorni di borsa", "P(VIX su) %", 330))
    fig.update_yaxes(range=[max(0, min(y + raw) - 12), min(100, max(y + raw) + 14)])
    fig.update_xaxes(tickmode="array", tickvals=x, ticktext=[f"{v}gg" for v in x])
    return fig


def fan_chart(analogue: dict, horizon: int) -> go.Figure | None:
    """
    Dove puo' arrivare il VIX secondo gli ANALOGHI storici: mediana e bande
    25-75 / 10-90. Nessun modello di mezzo, solo la distribuzione empirica dei
    giorni passati in condizioni simili a oggi.
    """
    if not analogue or "levels" not in analogue:
        return None
    lv = analogue["levels"]
    now = analogue["vix_now"]
    x = [0, horizon]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=[now, lv["p90"], lv["p10"], now],
                             fill="toself", fillcolor="rgba(33,150,243,0.13)",
                             line=dict(width=0), name="10-90 percentile",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x + x[::-1], y=[now, lv["p75"], lv["p25"], now],
                             fill="toself", fillcolor="rgba(33,150,243,0.28)",
                             line=dict(width=0), name="25-75 percentile",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[now, lv["median"]], mode="lines+markers+text",
                             line=dict(color=ACCENT, width=3), marker=dict(size=11),
                             text=["", f"mediana {lv['median']}"],
                             textposition="middle right", name="mediana"))
    fig.add_hline(y=now, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"VIX oggi {now}", annotation_position="bottom left",
                  annotation_font=dict(size=11, color=MUTED))
    fig.update_layout(**_layout(
        f"VIX atteso a {horizon} giorni — distribuzione degli analoghi storici "
        f"(n={analogue['n']})", "giorni di borsa", "livello VIX", 340))
    fig.update_xaxes(tickmode="array", tickvals=[0, horizon], ticktext=["oggi", f"+{horizon}gg"])
    return fig


def waterfall(components: list[dict]) -> go.Figure:
    """
    Da dove viene la probabilita': contributo di ogni livello in log-odds.
    Rende il modello auditabile a colpo d'occhio invece che opaco.
    """
    labels = [c["label"] for c in components]
    values = [c["value"] for c in components]
    measures = [c.get("measure", "relative") for c in components]

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=labels, y=values,
        text=[f"{v:+.2f}" if m == "relative" else f"{v:.2f}"
              for v, m in zip(values, measures)],
        textposition="outside", textfont=dict(size=11, color=TXT),
        connector=dict(line=dict(color=GRID)),
        increasing=dict(marker=dict(color=UP)),
        decreasing=dict(marker=dict(color=DOWN)),
        totals=dict(marker=dict(color=ACCENT)),
    ))
    fig.update_layout(**_layout("Da dove viene la probabilita' (log-odds)",
                                "", "log-odds", 330))
    fig.update_layout(showlegend=False)
    return fig


def conviction_bars(conv: dict) -> go.Figure:
    """Le tre componenti della conviction, separate e confrontabili."""
    names = ["Edge<br><sub>distanza dal 50%</sub>",
             "Accordo<br><sub>tra orizzonti</sub>",
             "Stabilita'<br><sub>giorni recenti</sub>"]
    vals = [conv["edge"], conv["agreement"], conv["stability"]]
    colors = [ACCENT if v >= 0.5 else MUTED for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                           marker=dict(color=colors),
                           text=[f"{v:.2f}" for v in vals], textposition="outside",
                           textfont=dict(color=TXT)))
    fig.update_layout(**_layout(f"Conviction {conv['score']:.2f} — le tre componenti",
                                "", "", 260))
    fig.update_xaxes(range=[0, 1.15])
    fig.update_layout(showlegend=False)
    return fig


# ============================================================
# [2] REGIME E STORIA
# ============================================================
def regime_timeline(ts: pd.DataFrame, episodes: pd.DataFrame | None = None) -> go.Figure:
    """VIX con bande colorate per regime: la storia dei regimi a colpo d'occhio."""
    fig = go.Figure()
    if episodes is not None and not episodes.empty:
        for _, e in episodes.iterrows():
            idx = int(e["regime_idx"])
            if idx == 0:
                continue  # il regime calmo resta senza banda: meno rumore visivo
            fig.add_vrect(x0=e["start"], x1=e["end"],
                          fillcolor=config.REGIME_COLORS[idx], opacity=0.16,
                          line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=ts.index, y=ts["VIX"], mode="lines", name="VIX",
                             line=dict(color=TXT, width=1.4)))
    if "stress_score" in ts.columns:
        fig.add_trace(go.Scatter(x=ts.index, y=ts["stress_score"] * 100, mode="lines",
                                 name="stress score (x100)", yaxis="y2",
                                 line=dict(color=ACCENT, width=1, dash="dot")))
    lay = _layout("Regime di volatilita' — storia", "", "VIX", 420)
    lay["yaxis2"] = dict(overlaying="y", side="right", range=[0, 100],
                         gridcolor=GRID, title="stress")
    fig.update_layout(**lay)
    return fig


def term_structure_chart(ts: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts["ts_ratio"], mode="lines",
                             name="VIX / VIX3M", line=dict(color=ACCENT, width=1.4)))
    if "ts_short" in ts.columns and ts["ts_short"].notna().any():
        fig.add_trace(go.Scatter(x=ts.index, y=ts["ts_short"], mode="lines",
                                 name="VIX9D / VIX (short-end)",
                                 line=dict(color=MUTED, width=1, dash="dot")))
    fig.add_hline(y=1.0, line=dict(color=UP, width=1, dash="dash"),
                  annotation_text="1.0 — sopra: backwardation (stress)",
                  annotation_font=dict(size=10, color=UP))
    fig.update_layout(**_layout("Term structure", "", "ratio", 320))
    return fig


def vvix_chart(ts: pd.DataFrame) -> go.Figure:
    """VVIX col suo log-zScore: livello e anomalia insieme."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts["VVIX"], mode="lines", name="VVIX",
                             line=dict(color="#AB47BC", width=1.3)))
    if "vvix_z" in ts.columns:
        fig.add_trace(go.Scatter(x=ts.index, y=ts["vvix_z"], mode="lines",
                                 name="log-zScore 90gg", yaxis="y2",
                                 line=dict(color=MUTED, width=1)))
    lay = _layout("VVIX — vol of vol (implicita)", "", "VVIX", 320)
    lay["yaxis2"] = dict(overlaying="y", side="right", gridcolor=GRID, title="z")
    fig.update_layout(**lay)
    return fig


def vix_range_chart(ts: pd.DataFrame) -> go.Figure:
    """Range giornaliero del VIX: vol-of-vol REALIZZATA, gratis dall'OHLC."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts["vix_range"], mode="lines",
                             name="range (H-L)/C %", line=dict(color="#FFA726", width=1)))
    fig.add_trace(go.Scatter(x=ts.index,
                             y=ts["vix_range"].rolling(21, min_periods=5).mean(),
                             mode="lines", name="media 21gg",
                             line=dict(color=TXT, width=1.6)))
    fig.update_layout(**_layout("Range intragiornaliero del VIX — vol of vol realizzata",
                                "", "% del livello", 300))
    return fig


def vrp_chart(ts: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts["VIX"], mode="lines", name="VIX (implied)",
                             line=dict(color=ACCENT, width=1.3)))
    if "exp_vol_ann" in ts.columns:
        fig.add_trace(go.Scatter(x=ts.index, y=ts["exp_vol_ann"], mode="lines",
                                 name="realized attesa", line=dict(color="#FFA726", width=1.2)))
    if "vrp" in ts.columns:
        fig.add_trace(go.Bar(x=ts.index, y=ts["vrp"], name="VRP", yaxis="y2",
                             marker=dict(color=MUTED), opacity=0.35))
    lay = _layout("Volatility Risk Premium", "", "punti vol", 340)
    lay["yaxis2"] = dict(overlaying="y", side="right", gridcolor=GRID, title="VRP")
    fig.update_layout(**lay)
    return fig


def prob_history(ts: pd.DataFrame, horizon: int, lookback: int = 504) -> go.Figure:
    """Storia della probabilita': serve a vedere se il segnale e' stabile o oscilla."""
    col = f"p_up_{horizon}"
    sub = ts.tail(lookback)
    fig = go.Figure()
    fig.add_hrect(y0=(0.5 - config.NEUTRAL_BAND) * 100, y1=(0.5 + config.NEUTRAL_BAND) * 100,
                  fillcolor=MUTED, opacity=0.15, line_width=0)
    fig.add_hline(y=50, line=dict(color=MUTED, width=1, dash="dot"))
    if col in sub.columns:
        fig.add_trace(go.Scatter(x=sub.index, y=sub[col] * 100, mode="lines",
                                 name=f"P(VIX su) {horizon}gg",
                                 line=dict(color=ACCENT, width=1.6)))
    fig.add_trace(go.Scatter(x=sub.index, y=sub["VIX"], mode="lines", name="VIX",
                             yaxis="y2", line=dict(color=TXT, width=1, dash="dot")))
    lay = _layout(f"P(VIX su) a {horizon}gg — ultimi {lookback} giorni", "", "%", 330)
    lay["yaxis2"] = dict(overlaying="y", side="right", gridcolor=GRID, title="VIX")
    fig.update_layout(**lay)
    return fig


# ============================================================
# [3] EVENT STUDY
# ============================================================
def event_bars(stats: pd.DataFrame, title: str) -> go.Figure:
    """
    Rendimenti forward per orizzonte con banda interquartile — e con la barra del
    BASELINE incondizionato accanto. Il confronto e' la sostanza: senza baseline,
    una media positiva non dice nulla.
    """
    x = list(stats.index)
    mean = stats["Media %"].to_numpy(dtype=float)
    base = (stats["Baseline %"].to_numpy(dtype=float)
            if "Baseline %" in stats.columns else np.zeros(len(x)))
    p25 = stats["P25 %"].to_numpy(dtype=float)
    p75 = stats["P75 %"].to_numpy(dtype=float)
    sig = stats["Sig"].tolist() if "Sig" in stats.columns else [""] * len(x)

    colors = [UP if m > 0 else DOWN for m in mean]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=mean, name="dopo il segnale", marker=dict(color=colors),
                         error_y=dict(type="data", symmetric=False,
                                      array=p75 - mean, arrayminus=mean - p25,
                                      color=MUTED, thickness=1),
                         text=[f"{m:+.1f}%{' *' if s else ''}" for m, s in zip(mean, sig)],
                         textposition="outside", textfont=dict(size=11, color=TXT)))
    fig.add_trace(go.Bar(x=x, y=base, name="baseline incondizionato",
                         marker=dict(color=MUTED, opacity=0.55)))
    fig.update_layout(**_layout(title, "orizzonte", "%", 340))
    fig.update_layout(barmode="group")
    return fig


def event_heatmap(tbl: pd.DataFrame, value_col: str = "Excess %") -> go.Figure:
    """
    Heatmap dell'EXCESS (non della media grezza) per regime x orizzonte.
    Le celle senza campione sufficiente restano vuote invece di mostrare un numero
    che sembra informativo e non lo e'.
    """
    if tbl.empty:
        return go.Figure(layout=_layout("Nessun dato", h=280))
    piv = tbl.pivot(index="Regime", columns="Orizzonte", values=value_col)
    ann = tbl.pivot(index="Regime", columns="Orizzonte", values="N")
    fdr = tbl.pivot(index="Regime", columns="Orizzonte", values="FDR")

    text = []
    for i in range(piv.shape[0]):
        row = []
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            n = ann.iloc[i, j]
            mark = "*" if str(fdr.iloc[i, j]) == "OK" else ""
            row.append("" if pd.isna(v) else f"{v:+.1f}{mark}<br><sub>n={int(n)}</sub>")
        text.append(row)

    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(dtype=float), x=list(piv.columns), y=list(piv.index),
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale=[[0, DOWN], [0.5, BG], [1, UP]], zmid=0,
        colorbar=dict(title=value_col)))
    fig.update_layout(**_layout(f"{value_col} per regime e orizzonte "
                                "(* = sopravvive alla correzione FDR)", "", "", 330))
    return fig


def event_markers(ts: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    """VIX con i marker degli estremi VVIX passati."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts["VVIX"], mode="lines", name="VVIX",
                             line=dict(color="#AB47BC", width=1.2)))
    for kind, color, sym in (("overbought", UP, "triangle-up"),
                             ("oversold", DOWN, "triangle-down")):
        sel = events[events["signal"] == kind]
        sel = sel[sel.index.isin(ts.index)]
        if sel.empty:
            continue
        fig.add_trace(go.Scatter(x=sel.index, y=ts.loc[sel.index, "VVIX"],
                                 mode="markers", name=kind,
                                 marker=dict(color=color, size=11, symbol=sym)))
    fig.update_layout(**_layout("Estremi statistici del VVIX", "", "VVIX", 340))
    return fig


# ============================================================
# [4] VALIDAZIONE
# ============================================================
def reliability_diagram(calib: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfetta",
                             line=dict(color=MUTED, dash="dash", width=1)))
    if not calib.empty:
        fig.add_trace(go.Scatter(x=calib["pred_mean"], y=calib["obs_freq"],
                                 mode="markers+lines", name="osservato",
                                 marker=dict(size=calib["n"] / calib["n"].max() * 22 + 6,
                                             color=ACCENT),
                                 line=dict(color=ACCENT, width=1.5)))
    fig.update_layout(**_layout("Affidabilita' OOS — quando dico 70%, succede il 70%?",
                                "probabilita' prevista", "frequenza osservata", 330))
    return fig


def selective_curve(sel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sel["coverage"] * 100, y=sel["accuracy"] * 100,
                             mode="lines+markers+text",
                             text=[f"τ={t:.2f}" for t in sel["tau"]],
                             textposition="top center", textfont=dict(size=10),
                             line=dict(color=ACCENT, width=2), marker=dict(size=9),
                             name="accuratezza"))
    fig.update_layout(**_layout("Selective prediction — astenersi conviene?",
                                "copertura %", "accuratezza %", 330))
    return fig


def separation_chart(sep: dict) -> go.Figure:
    """La separazione economica: e' qui che sta il valore per le opzioni."""
    cats = ["media", "mediana", "ex-COVID"]
    up = [sep.get("fwd_ret_when_up_pct"), sep.get("median_up_pct"), None]
    dn = [sep.get("fwd_ret_when_down_pct"), sep.get("median_down_pct"), None]
    spread = [sep.get("spread_pct"), sep.get("median_spread_pct"),
              sep.get("ex_crisis_spread_pct")]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=[v if v is not None else 0 for v in up],
                         name="quando dice SU", marker=dict(color=UP)))
    fig.add_trace(go.Bar(x=cats, y=[v if v is not None else 0 for v in dn],
                         name="quando dice GIU'", marker=dict(color=DOWN)))
    fig.add_trace(go.Scatter(x=cats, y=spread, mode="markers+text", name="spread",
                             marker=dict(color=ACCENT, size=13, symbol="diamond"),
                             text=[f"{v:+.1f}" if v is not None else "" for v in spread],
                             textposition="top center"))
    fig.update_layout(**_layout("ΔVIX forward: separazione tra le due previsioni",
                                "", "%", 330))
    fig.update_layout(barmode="group")
    return fig


def rolling_hitrate(preds: pd.DataFrame, window: int, baseline: float) -> go.Figure:
    correct = ((preds["p_post"] > 0.5).astype(float) == preds["y"]).astype(float)
    roll = correct.rolling(window, min_periods=window // 2).mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds.index, y=roll, mode="lines",
                             name=f"hit rate {window}gg", line=dict(color=ACCENT, width=1.3)))
    fig.add_hline(y=baseline * 100, line=dict(color=UP, width=1, dash="dash"),
                  annotation_text="baseline maggioritaria",
                  annotation_font=dict(size=10, color=UP))
    fig.update_layout(**_layout("Stabilita' dell'edge nel tempo (OOS)", "", "%", 300))
    return fig


# ============================================================
# [5] STAGIONALITA'
# ============================================================
def seasonal_level_chart(curve: pd.DataFrame, current_doy: int | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve["ci_high"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve.index, y=curve["ci_low"], mode="lines",
                             fill="tonexty", fillcolor="rgba(33,150,243,0.15)",
                             line=dict(width=0), name="CI 90%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve.index, y=curve["mean_ratio"], mode="lines",
                             name="media", line=dict(color=MUTED, width=1)))
    fig.add_trace(go.Scatter(x=curve.index, y=curve["smooth"], mode="lines",
                             name="curva liscia", line=dict(color=ACCENT, width=2.5)))
    fig.add_hline(y=1.0, line=dict(color=MUTED, width=1, dash="dot"))
    if current_doy:
        fig.add_vline(x=current_doy, line=dict(color="#FFA726", width=2),
                      annotation_text="oggi", annotation_font=dict(size=11))
    fig.update_layout(**_layout("Stagionalita' del livello (de-regimizzata)",
                                "giorno dell'anno", "VIX / mediana annuale", 330))
    return fig


def forward_seasonality_chart(curve: pd.DataFrame, horizon: int,
                              current_doy: int | None) -> go.Figure:
    fig = go.Figure()
    colors = [UP if v > 0 else DOWN for v in curve["fwd_mean_pct"].fillna(0)]
    fig.add_trace(go.Bar(x=curve.index, y=curve["fwd_mean_pct"], name=f"ΔVIX {horizon}gg",
                         marker=dict(color=colors), opacity=0.75))
    fig.add_trace(go.Scatter(x=curve.index, y=curve["hit_rate"] * 100, mode="lines",
                             name="hit rate %", yaxis="y2",
                             line=dict(color=TXT, width=1.4)))
    if current_doy:
        fig.add_vline(x=current_doy, line=dict(color="#FFA726", width=2))
    lay = _layout(f"Stagionalita' della variazione forward ({horizon}gg)",
                  "giorno dell'anno", "ΔVIX medio %", 330)
    lay["yaxis2"] = dict(overlaying="y", side="right", range=[0, 100],
                         gridcolor=GRID, title="hit %")
    fig.update_layout(**lay)
    return fig


def scan_chart(scan: dict, windows: list[int]) -> go.Figure:
    x = [str(w) for w in windows]
    y = [scan.get(str(w), {}).get("mean_pct") or 0 for w in windows]
    hit = [(scan.get(str(w), {}).get("hit_rate") or 0) * 100 for w in windows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, name="ΔVIX medio %",
                         marker=dict(color=[UP if v > 0 else DOWN for v in y])))
    fig.add_trace(go.Scatter(x=x, y=hit, mode="lines+markers", name="hit rate %",
                             yaxis="y2", line=dict(color=TXT, width=1.5)))
    lay = _layout(f"Scan multi-finestra — consistenza del segno "
                  f"{scan.get('consistency', 0):.0%}", "finestra (giorni)", "%", 300)
    lay["yaxis2"] = dict(overlaying="y", side="right", range=[0, 100], gridcolor=GRID)
    fig.update_layout(**lay)
    return fig
