"""
app.py — KQ Volatility Engine, dashboard Streamlit.

L'app LEGGE gli snapshot prodotti dalla pipeline: non ricalcola nulla, non ha
bisogno di chiavi API, si carica in un istante e mostra esattamente gli stessi
numeri dell'alert Telegram.

Struttura in cinque righe, tutto il resto negli expander:

    1  SEMAFORO         regime, segnale, conviction, affidabilita' storica
    2  COSA ASPETTARSI  curva di probabilita' + VIX atteso dagli analoghi
    3  PERCHE'          da dove viene la probabilita', contributo per contributo
    4  STORIA           regimi passati, episodi, event study sul VVIX
    5  QUANTO FIDARSI   validazione walk-forward out-of-sample

Regola trasversale: nessun numero senza il suo termine di paragone.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src import charts, config

st.set_page_config(page_title="KQ Volatility Engine | Kriterion Quant",
                   page_icon="🌡️", layout="wide", initial_sidebar_state="expanded")

BIAS_STYLE = {
    "LONG_VOL": ("LONG VOLATILITA'", "#EF5350",
                 "contesto da convessita': comprare vol / call OTM"),
    "SHORT_VOL": ("SHORT VOLATILITA'", "#26A69A",
                  "contesto da premio: vendere vol / incassare carry"),
    "FLAT": ("FLAT — ASTENSIONE", "#9E9E9E",
             "nessun setup: e' una posizione legittima, spesso la piu' giusta"),
}


# ===================================================
# CARICAMENTO
# ===================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_all():
    snap = json.loads(config.SNAPSHOT_JSON.read_text(encoding="utf-8"))
    ts = pd.read_parquet(config.TIMESERIES_PARQUET)
    seas = pd.read_parquet(config.SEASONALITY_PARQUET)
    epi = (pd.read_parquet(config.EPISODES_PARQUET)
           if config.EPISODES_PARQUET.exists() else pd.DataFrame())
    ev = (pd.read_parquet(config.EVENTS_PARQUET)
          if config.EVENTS_PARQUET.exists() else pd.DataFrame())
    qual = (json.loads(config.DATA_QUALITY_JSON.read_text(encoding="utf-8"))
            if config.DATA_QUALITY_JSON.exists() else {})
    return snap, ts, seas, epi, ev, qual


@st.cache_data(ttl=900, show_spinner=False)
def load_validation():
    val = json.loads(config.VALIDATION_JSON.read_text(encoding="utf-8"))
    preds = (pd.read_parquet(config.OOS_PREDICTIONS_PARQUET)
             if config.OOS_PREDICTIONS_PARQUET.exists() else pd.DataFrame())
    return val, preds


try:
    snap, ts, seas_df, episodes, events, quality = load_all()
except Exception:
    st.title("KQ Volatility Engine")
    st.warning("Nessuno snapshot trovato. Esegui prima la pipeline:\n\n"
               "`python run_pipeline.py --no-alerts`\n\n"
               "oppure attendi il run notturno della GitHub Action.")
    st.stop()

f = snap["forecast"]
op = f["operational"]
conv = f["conviction"]
reg = snap["regime"]
vrp = snap["vrp"]["latest"]
H = f["horizon_primary"]
per_h = f["per_horizon"]

try:
    val, oos_preds = load_validation()
except Exception:
    val, oos_preds = None, pd.DataFrame()


def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))


# ===================================================
# SIDEBAR
# ===================================================
with st.sidebar:
    st.title("Stato")
    st.metric("Dati al", snap.get("asof", "n/d"))
    fr = snap.get("freshness", {})
    if fr.get("is_fresh", True):
        st.success("Dati freschi")
    else:
        st.error(f"Dati stantii (lag {fr.get('lag_days')}gg) — segnali sospesi")
    st.caption(f"Generato (UTC): {snap.get('generated_utc', '')}")

    st.divider()
    st.markdown("**Qualita' dei dati**")
    dq = snap.get("data_quality", {})
    st.caption(f"{dq.get('rows', 0):,} barre · {dq.get('first')} → {dq.get('last')}")
    st.caption("Serie: " + ", ".join(dq.get("series", [])))
    n_anom = dq.get("anomalies_total", 0)
    if n_anom:
        st.warning(f"{n_anom} divergenze CBOE/EODHD risolte su CBOE")
        with st.expander("Dettaglio divergenze"):
            st.dataframe(pd.DataFrame(quality.get("anomalies", [])),
                         width="stretch", height=200)
    else:
        st.caption("Nessuna divergenza tra le fonti")

    st.divider()
    st.caption(f"Orizzonti: {', '.join(str(h) + 'gg' for h in snap['params']['horizons'])} "
               f"(primario {H}gg)")
    st.caption(f"Stimatore realized vol: {snap['params'].get('rv_estimator', 'n/d')}")
    st.caption("Fonti: CBOE (primaria) + EODHD (estensione e SPX)")
    st.divider()
    st.caption("Strumento di ricerca. Nessun consiglio finanziario, "
               "nessuna esecuzione automatica.")

st.title("KQ Volatility Engine")
st.caption("Regime della volatilita' dell'S&P 500 e forecast probabilistico del VIX — "
           "Kriterion Quant")


# ===================================================
# [1] SEMAFORO
# ===================================================
label, color, subtitle = BIAS_STYLE.get(op["bias"], (op["bias"], "#9E9E9E", ""))
# La conviction descrive quanto fidarsi di una GAMBA ATTIVA. Con bias FLAT non c'e'
# nulla di cui essere convinti: mostrare un punteggio alto accanto a "astensione"
# sarebbe la stessa incoerenza della vecchia "Confluenza 100%" su una probabilita'
# del 49%. Il punteggio resta visibile sotto, dove e' etichettato per quello che e':
# coerenza del segnale direzionale, non convinzione operativa.
if op["bias"] == "FLAT":
    stars = "—"
    conv_txt = f"nessuna gamba attiva · coerenza direzionale {conv['score']:.2f}"
else:
    stars = {"ALTA": "★★★", "MEDIA": "★★☆", "BASSA": "★☆☆",
             "NULLA": "☆☆☆"}[op["conviction"]]
    conv_txt = f"{op['conviction']} ({conv['score']:.2f})"

acc_txt = "validazione non disponibile"
if val and str(H) in val.get("horizons", {}):
    vm = val["horizons"][str(H)]["metrics"]
    vs = val["horizons"][str(H)]["significance"]
    acc_txt = (f"{vm['accuracy']:.1%} OOS contro baseline {vm['majority_baseline_acc']:.1%} "
               f"· {'significativo' if vs['significant_5pct'] else 'NON significativo'}")

st.markdown(
    f"""<div style="background:{color}1A;border-left:7px solid {color};
    padding:18px 22px;border-radius:10px;margin:6px 0 4px 0;">
      <div style="display:flex;flex-wrap:wrap;gap:28px;align-items:baseline;">
        <div><div style="color:#9E9E9E;font-size:0.78rem;text-transform:uppercase;
             letter-spacing:.06em;">Regime</div>
             <div style="font-size:1.15rem;font-weight:600;">{reg['regime_label']}</div></div>
        <div><div style="color:#9E9E9E;font-size:0.78rem;text-transform:uppercase;
             letter-spacing:.06em;">Segnale · {H} giorni</div>
             <div style="font-size:1.45rem;font-weight:700;color:{color};">{label}</div></div>
        <div><div style="color:#9E9E9E;font-size:0.78rem;text-transform:uppercase;
             letter-spacing:.06em;">Conviction</div>
             <div style="font-size:1.3rem;font-weight:600;">{stars}
             <span style="font-size:0.85rem;color:#9E9E9E;">{conv_txt}</span></div></div>
        <div><div style="color:#9E9E9E;font-size:0.78rem;text-transform:uppercase;
             letter-spacing:.06em;">Affidabilita' storica</div>
             <div style="font-size:1.0rem;">{acc_txt}</div></div>
      </div>
      <div style="margin-top:12px;color:#C8C8C8;font-size:0.93rem;">
        <b>{subtitle}</b><br>{op.get('rationale', '')}</div>
    </div>""", unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("VIX", f"{reg['vix']:.2f}",
          f"percentile {reg['vix_rank']:.0%}" if reg.get("vix_rank") else None)
k2.metric("VIX3M", f"{reg.get('vix3m')}",
          f"ratio {reg.get('ts_ratio')}" if reg.get("ts_ratio") else None)
k3.metric("VVIX", f"{reg.get('vvix')}",
          f"z {reg.get('vvix_z')}" if reg.get("vvix_z") is not None else None)
k4.metric(f"P(VIX su) {H}gg", f"{f['p_up_primary']:.0%}", f["direction"])
k5.metric("VRP", f"{vrp.get('vrp') if vrp.get('vrp') is not None else 'n/d'}",
          f"{vrp.get('state')}" if vrp.get("state") else None)
k6.metric("Stress score", f"{reg['stress_score']:.2f}",
          f"range VIX {reg.get('vix_range')}%" if reg.get("vix_range") else None)

if not f.get("calibrated"):
    st.caption(f"Probabilita' **non calibrata**. {f.get('calibration_note', '')}")
else:
    st.caption(f"Probabilita' **calibrata OOS**. {f.get('calibration_note', '')} "
               f"(grezza: {f['p_up_primary_raw']:.0%})")

with st.expander("Come si legge questa dashboard — 30 secondi"):
    st.markdown(f"""
**Guarda solo la barra colorata qui sopra.** Ti dice il regime di oggi, cosa suggerisce
il modello, quanto ci crede e quanto ha funzionato storicamente. Tutto il resto serve a
*capire perche'* e a *decidere quanto fidarti*.

**I tre segnali possibili**
- **LONG VOLATILITA'** — contesto da convessita': il modello vede un edge rialzista sul
  VIX e la convessita' costa poco.
- **SHORT VOLATILITA'** — contesto da premio: il VRP e' nel percentile alto, confermato
  dai due stimatori, in regime calmo e senza rischio-spike.
- **FLAT** — nessun setup. E' una posizione legittima: la curva di selective prediction
  in fondo mostra che astenersi quando la probabilita' e' vicina al 50% alza davvero
  l'accuratezza.

**La conviction** non e' un'opinione: e' la media pesata di tre misure indipendenti —
quanto la probabilita' si allontana dal 50%, quanti orizzonti concordano, e quanto il
segnale e' stato stabile negli ultimi {config.STABILITY_WINDOW} giorni.

**Il valore economico** viene dall'asimmetria del payoff delle opzioni, non
dall'azzeccare la direzione ogni volta: un'accuratezza del 58% con una separazione di
{val['horizons'][str(H)]['separation']['spread_pct'] if val and str(H) in val.get('horizons', {}) else '~9'}%
tra le due previsioni e' molto piu' utile di quanto suoni.
""")

st.divider()

# ===================================================
# [2] COSA ASPETTARSI
# ===================================================
st.subheader("Cosa aspettarsi")
st.markdown(
    "A sinistra la **probabilita' per orizzonte**: la *forma* della curva e' gia' "
    "un'interpretazione — crescente indica pressione strutturale al rialzo della vol; "
    "alta a 5 giorni e bassa a 20 indica uno spike tattico con rientro atteso; piatta "
    "attorno al 50% significa nessun edge. A destra **dove puo' arrivare il VIX** "
    "secondo i giorni storici in condizioni simili a oggi: nessun modello di mezzo, "
    "solo la distribuzione empirica degli analoghi.")

c1, c2 = st.columns(2)
c1.plotly_chart(charts.prob_curve(per_h), width="stretch")

analogues = snap.get("analogues", {})
an = analogues.get(str(H)) or {}
fan = charts.fan_chart(an, H)
if fan is not None:
    c2.plotly_chart(fan, width="stretch")
    c2.caption(f"Analoghi: {an.get('basis')} · P(VIX su) empirica {an.get('prob_up', 0):.0%} "
               f"· mediana {an.get('median')}%")
else:
    c2.info("Analoghi storici non disponibili con i filtri correnti.")

cols = st.columns(len(per_h))
for col, h in zip(cols, sorted(per_h, key=int)):
    v = per_h[h]
    a = analogues.get(h) or {}
    col.metric(f"{h} giorni", f"{v['p_up']:.0%}", v["direction"])
    col.caption(f"bucket `{v['bucket']}` · n={v['n_bucket']}"
                + (f" · mediana analoghi {a.get('median')}%" if a else ""))

st.divider()

# ===================================================
# [3] PERCHE'
# ===================================================
st.subheader("Perche'")
_sw = snap["params"].get("seasonal_weight", config.SEASONAL_WEIGHT)
st.markdown(
    "La probabilita' non e' un numero opaco: nasce sommando contributi in log-odds. "
    "Si parte dalla **climatologia** (quanto spesso il VIX sale, storicamente) e si "
    "aggiunge quello che dice il **regime**."
    + ("  La **stagionalita' e' disattivata** (peso 0): la validazione ha mostrato che "
       "peggiorava accuratezza e calibrazione a tutti gli orizzonti. Resta calcolata e "
       "mostrata come contesto, ma non sposta piu' la probabilita'."
       if _sw == 0 else
       f"  La **stagionalita'** entra con peso ridotto ({_sw}): e' validazione, non guida."))

base_rate = f.get("cond_detail", {}).get(str(H), {}).get("base_rate", 0.5)
p_cond = per_h[str(H)]["p_cond"]
p_prior = per_h[str(H)]["p_prior"]
p_raw = per_h[str(H)]["p_up_raw"]
p_fin = per_h[str(H)]["p_up"]

comps = [
    {"label": "Climatologia", "value": _logit(base_rate), "measure": "absolute"},
    {"label": "Regime<br><sub>percentile × term structure</sub>",
     "value": _logit(p_cond) - _logit(base_rate)},
]
if _sw != 0:
    comps.append({"label": f"Stagionalita'<br><sub>peso {_sw}</sub>",
                  "value": _sw * _logit(p_prior)})
if f.get("calibrated"):
    comps.append({"label": "Calibrazione OOS", "value": _logit(p_fin) - _logit(p_raw)})
comps.append({"label": "Posterior", "value": _logit(p_fin), "measure": "total"})

w1, w2 = st.columns([3, 2])
w1.plotly_chart(charts.waterfall(comps), width="stretch")
w2.plotly_chart(charts.conviction_bars(conv), width="stretch")
w2.caption(conv["components_note"])

with st.expander("Stato dettagliato del regime e del premio"):
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Flag di regime** (0 = calma, 1 = stress)")
        st.dataframe(pd.DataFrame([{"flag": k, "valore": "n/d" if v is None else str(v)}
                                   for k, v in (reg.get("flags") or {}).items()]),
                     width="stretch", hide_index=True)
        st.caption(f"Pesi: {config.FLAG_WEIGHTS}")
    with d2:
        st.markdown("**Volatility Risk Premium**")
        # Colonna omogenea di stringhe: mescolare float, bool e str fa fallire la
        # serializzazione Arrow di Streamlit e l'intero blocco non viene renderizzato.
        st.dataframe(pd.DataFrame([
            {"voce": v, "valore": "n/d" if x is None else str(x)} for v, x in [
                ("VRP consensus", vrp.get("vrp_consensus")),
                ("VRP EGARCH", vrp.get("vrp_egarch")),
                ("VRP HAR", vrp.get("vrp_har")),
                ("percentile rolling", vrp.get("vrp_rank")),
                ("stato", vrp.get("state")),
                ("stimatori concordi", vrp.get("agree")),
                ("metodo", vrp.get("method")),
            ]]), width="stretch", hide_index=True)
        if val and (val.get("vrp_validation") or {}).get("available"):
            vv = val["vrp_validation"]
            (st.success if vv.get("any_significant") else st.warning)(vv["verdict"])

st.divider()

# ===================================================
# [4] STORIA
# ===================================================
st.subheader("Storia")
st.markdown(
    "Il regime attuale ha senso solo dentro la sua storia. Le bande colorate segnano "
    "i periodi di **transizione** (giallo) e **stress** (rosso); il regime calmo resta "
    "senza banda per non appesantire la lettura.")

st.plotly_chart(charts.regime_timeline(ts, episodes), width="stretch")

if not episodes.empty:
    st.markdown("**Episodi di regime** — quanto sono durati e come sono finiti")
    epi_view = episodes.copy()
    epi_view = epi_view[epi_view["regime_idx"] > 0].sort_values("start", ascending=False)
    epi_view["start"] = pd.to_datetime(epi_view["start"]).dt.strftime("%Y-%m-%d")
    epi_view["end"] = pd.to_datetime(epi_view["end"]).dt.strftime("%Y-%m-%d")
    epi_view = epi_view.rename(columns={
        "label": "Regime", "start": "Inizio", "end": "Fine", "days": "Giorni",
        "vix_start": "VIX ingresso", "vix_end": "VIX uscita", "vix_max": "VIX max",
        "spx_chg_pct": "SPX %"})
    st.dataframe(epi_view.drop(columns=["regime_idx"]), width="stretch",
                 hide_index=True, height=260)

g1, g2 = st.columns(2)
g1.plotly_chart(charts.term_structure_chart(ts), width="stretch")
g2.plotly_chart(charts.vvix_chart(ts), width="stretch")

g3, g4 = st.columns(2)
if "vix_range" in ts.columns and ts["vix_range"].notna().any():
    g3.plotly_chart(charts.vix_range_chart(ts), width="stretch")
if "vrp" in ts.columns and ts["vrp"].notna().any():
    g4.plotly_chart(charts.vrp_chart(ts), width="stretch")

st.plotly_chart(charts.prob_history(ts, H), width="stretch")

# --- Event study VVIX ---
st.markdown("#### Event study — estremi statistici del VVIX")
es = snap.get("event_study", {})
if es:
    st.markdown(
        f"Quando il log-zScore del VVIX supera **{config.VVIX_Z_UPPER}** o scende sotto "
        f"**{config.VVIX_Z_LOWER}** si registra un evento (solo il primo di ogni "
        f"episodio, cooldown {config.VVIX_COOLDOWN} giorni per tipo). Le barre grigie "
        "sono il **rendimento incondizionato** allo stesso orizzonte: la differenza "
        "e' l'unica cosa che conta. L'asterisco segnala le celle significative al 5%.")

    if not events.empty:
        st.plotly_chart(charts.event_markers(ts, events), width="stretch")

    tabs = st.tabs([f"Overbought VVIX (n={es.get('overbought', {}).get('n', 0)})",
                    f"Oversold VVIX (n={es.get('oversold', {}).get('n', 0)})"])
    for tab, kind in zip(tabs, ("overbought", "oversold")):
        with tab:
            info = es.get(kind, {})
            if info.get("n", 0) < config.MIN_EVENTS:
                st.warning(f"Solo {info.get('n', 0)} eventi: campione insufficiente.")
                continue
            st.caption(f"Dal {info.get('first')} al {info.get('last')}")
            for asset in ("VIX", "SPX"):
                rows = info.get(f"stats_{asset}", [])
                if not rows:
                    continue
                dfa = pd.DataFrame(rows).set_index("Orizzonte")
                st.plotly_chart(
                    charts.event_bars(dfa, f"{asset} dopo VVIX {kind}"), width="stretch")
                st.dataframe(dfa, width="stretch")
else:
    st.info("Event study non disponibile in questo snapshot.")

st.divider()

# ===================================================
# [5] QUANTO FIDARSI
# ===================================================
st.subheader("Quanto fidarsi — validazione walk-forward")

if not val:
    st.warning("Validazione non ancora eseguita. Lancia `python validate.py`.")
else:
    st.markdown(
        "Il modello ri-stimato **solo sul passato** e testato sul futuro, lungo tutta la "
        "storia, con **purging + embargo**. E' l'unico numero che conta: l'in-sample "
        "non vale nulla. Ogni orizzonte ha la sua validazione indipendente.")

    hsel = st.radio("Orizzonte", sorted(val["horizons"], key=int),
                    index=sorted(val["horizons"], key=int).index(str(H)),
                    horizontal=True, format_func=lambda x: f"{x} giorni")
    e = val["horizons"][hsel]
    m, sg, sep, cal = e["metrics"], e["significance"], e["separation"], e["calibrator"]

    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Accuratezza OOS", f"{m['accuracy']:.1%}", f"n={m['n_oos']}")
    v2.metric("Edge vs maggioritaria", f"{m['edge_vs_majority']:+.1%}",
              f"base {m['majority_baseline_acc']:.1%}")
    v3.metric("95% CI accuratezza", f"{sg['acc_ci_low']:.1%}–{sg['acc_ci_high']:.1%}",
              f"N effettiva {sg['effective_n']}")
    v4.metric("Brier Skill Score", f"{m['brier_skill_score']:+.3f}", "vs climatologia")
    v5.metric("AUC", f"{m['auc']:.3f}")

    verdict = e.get("verdict", "")
    (st.success if verdict.startswith("VERDE") else
     st.warning if verdict.startswith("GIALLO") else st.error)(verdict)

    st.caption(f"OOS {e['params']['oos_start']} → {e['params']['oos_end']} · "
               f"embargo {e['params']['embargo']}gg · refit ogni "
               f"{e['params']['refit_every']}gg · blocco bootstrap {sg['block']}gg")

    r1, r2 = st.columns(2)
    r1.plotly_chart(charts.reliability_diagram(pd.DataFrame(e["reliability"])),
                    width="stretch")
    r2.plotly_chart(charts.selective_curve(pd.DataFrame(e["selective"])), width="stretch")
    r2.caption("La curva scende a destra: astenersi quando la probabilita' e' vicina "
               "al 50% aumenta davvero l'accuratezza. E' la giustificazione empirica "
               "del segnale FLAT.")

    s1, s2 = st.columns([3, 2])
    s1.plotly_chart(charts.separation_chart(sep), width="stretch")
    with s2:
        st.markdown("**Rilevanza economica**")
        st.markdown(
            f"Quando dice **SU**, nei {hsel} giorni successivi il VIX fa in media "
            f"**{sep['fwd_ret_when_up_pct']}%**; quando dice **GIU'**, "
            f"**{sep['fwd_ret_when_down_pct']}%**.\n\n"
            f"- Spread medio **{sep['spread_pct']}%** — CI 95% "
            f"[{sep['spread_ci_low']}, {sep['spread_ci_high']}]\n"
            f"- Spread mediano **{sep['median_spread_pct']}%** (robusto alle code)\n"
            f"- Escludendo il COVID **{sep['ex_crisis_spread_pct']}%**\n"
            f"- Il modello dice SU {sep['n_up']} volte su {sep['n_up'] + sep['n_down']}")
        if (sep.get("ex_crisis_spread_pct") or 0) >= 0.7 * (sep.get("spread_pct") or 1):
            st.caption("Lo spread regge anche escludendo la crisi e in mediana: "
                       "la separazione non dipende da pochi eventi estremi.")
        else:
            st.caption("Lo spread si riduce molto escludendo la crisi: buona parte "
                       "dell'edge viene da pochi spike. Ottimo per payoff convessi, "
                       "ma va saputo.")

    if not oos_preds.empty:
        sub = oos_preds[oos_preds["horizon"] == int(hsel)] if "horizon" in oos_preds.columns \
            else oos_preds
        if not sub.empty:
            st.plotly_chart(charts.rolling_hitrate(sub, config.WF_ROLLING_WINDOW,
                                                   m["majority_baseline_acc"]),
                            width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Calibrazione isotonica — ammessa o respinta?**")
        gate = cal.get("gate")
        if gate:
            st.dataframe(pd.DataFrame([
                {"metrica": k, "valore": str(v)} for k, v in [
                    ("Brier held-out prima", gate["brier_before"]),
                    ("Brier held-out dopo", gate["brier_after"]),
                    ("Guadagno", gate["brier_improvement"]),
                    ("CI 95% del guadagno",
                     f"[{gate.get('improvement_ci_low')}, {gate.get('improvement_ci_high')}]"),
                    ("Guadagno significativo", gate.get("significant")),
                    ("Spostamento medio delle prob.", gate.get("mean_abs_shift")),
                    ("Applicata", cal.get("apply")),
                ]]), width="stretch", hide_index=True)
        st.caption(cal.get("gate_reason", ""))
    with c2:
        ab = e.get("ablation")
        if ab:
            aw, altw = ab.get("active_weight"), ab.get("alt_weight")
            st.markdown(f"**Peso stagionale: {aw} (attivo) contro {altw} (alternativo)**")
            st.dataframe(pd.DataFrame([
                {"metrica": k, "valore": str(v)} for k, v in [
                    (f"Accuratezza con peso {aw}", ab["active_acc"]),
                    (f"Accuratezza con peso {altw}", ab["alt_acc"]),
                    ("Delta accuratezza", ab["delta_acc"]),
                    (f"BSS con peso {aw}", ab["active_bss"]),
                    (f"BSS con peso {altw}", ab["alt_bss"]),
                    ("Delta BSS", ab["delta_bss"]),
                ]]), width="stretch", hide_index=True)
            # Tolleranza: sotto 0.2 punti di accuratezza la differenza e' rumore, non
            # un segnale. Senza questa soglia il pannello griderebbe al lupo per uno
            # scarto dello 0.03%.
            TOL = 0.002
            acc_worse = ab["delta_acc"] < -TOL
            bss_worse = ab["delta_bss"] < 0
            if not acc_worse and not bss_worse:
                st.caption(f"Il peso attivo ({aw}) regge su entrambe le metriche: "
                           "la stagionalita' non aggiunge nulla di dimostrabile.")
            elif acc_worse and bss_worse:
                st.caption(f"⚠️ Con peso {altw} migliorano **entrambe** le metriche: "
                           "rivedi `SEASONAL_WEIGHT` in `src/config.py`.")
            else:
                metrica = "l'accuratezza" if acc_worse else "il BSS"
                altra = "il BSS" if acc_worse else "l'accuratezza"
                st.caption(f"Con peso {altw} migliora {metrica} ma peggiora {altra}: "
                           f"segnale contrastante, il peso {aw} resta la scelta "
                           "conservativa (probabilita' meglio calibrate).")

# --- Stagionalita' ---
with st.expander("Stagionalita' — contesto, non guida" + (" (peso 0)" if _sw == 0 else "")):
    if _sw == 0:
        st.info("Il prior stagionale **non entra nel posterior**: la validazione "
                "walk-forward ha mostrato che peggiora accuratezza e calibrazione a "
                "tutti e tre gli orizzonti. Resta qui come contesto osservabile, e "
                "l'ablazione a ogni run ricontrolla se la decisione regge.")
    st.markdown(
        "Curva annuale **de-regimizzata**: ogni valore e' espresso come rapporto alla "
        "mediana mobile annuale *prima* di mediare per giorno di calendario. Senza "
        "questo passaggio la 'stagionalita'' e' solo l'impronta di dove sono caduti "
        "gli shock storici.")
    doy = snap["seasonality"].get("current_doy")
    lvl = seas_df[[c for c in seas_df.columns if c.startswith("lvl_")]].copy()
    lvl.columns = [c[4:] for c in lvl.columns]
    st.plotly_chart(charts.seasonal_level_chart(lvl, doy), width="stretch")

    fwdc = seas_df[[c for c in seas_df.columns if c.startswith("fwd_")]].copy()
    fwdc.columns = [c[4:] for c in fwdc.columns]
    st.plotly_chart(charts.forward_seasonality_chart(fwdc, H, doy), width="stretch")
    st.plotly_chart(charts.scan_chart(snap["seasonality"]["scan"],
                                      snap["params"]["scan_windows"]), width="stretch")
    st.dataframe(pd.DataFrame(snap["seasonality"]["priors"]).T.astype(str),
                 width="stretch")

st.divider()
st.caption(
    "**Disclaimer** — finalita' esclusivamente educative e di ricerca quantitativa. "
    "Le analisi storiche non garantiscono risultati futuri e non costituiscono "
    "consulenza finanziaria. | **Kriterion Quant** — kriterionquant.com · "
    f"snapshot {snap.get('asof')} · render {datetime.now().strftime('%d/%m/%Y %H:%M')}")
