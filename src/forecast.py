"""
forecast.py — Posterior multi-orizzonte, conviction e bias operativo.

Tre livelli fusi in log-odds:
    1) likelihood mean-reversion  -> P(up) empirica condizionata a percentile VIX
                                     x term structure (estimatore condiviso)
    2) prior stagionale           -> peso ridotto: e' validazione, non guida
    3) VRP / vol-of-vol           -> determina la GAMBA, non la direzione

Tre correzioni rispetto al forecaster originale:

  * CONFLUENZA SOSTITUITA DA CONVICTION. Il vecchio punteggio sommava cinque voti
    di cui almeno due ridondanti — la term structure votava gia' dentro la
    likelihood — e uno contraddiceva il proprio commento: il codice faceva votare
    la backwardation per "VIX su" mentre la documentazione diceva "propensione al
    rientro". Il sintomo era una dashboard che mostrava "Confluenza 100%" accanto a
    una probabilita' del 49.4%, cioe' un lancio di moneta. La conviction ora poggia
    su tre elementi misurabili e non ridondanti: distanza dal 50%, accordo tra
    orizzonti, stabilita' del segnale nei giorni recenti.

  * CALIBRAZIONE CONDIZIONATA. Si applica solo se ha superato il cancello held-out
    (vedi calibration.py). Prima veniva applicata sempre, anche quando peggiorava
    Brier e log-loss, spingendo il posterior dentro la banda neutra.

  * GATE VRP ONESTO. La gamba short-vol dichiara esplicitamente se il gate ha
    evidenza fuori campione. Senza evidenza puo' ancora attivarsi, ma la conviction
    e' limitata e il motivo e' scritto: e' carry, non edge dimostrato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calibration, cond_model, config


# ============================================================
# LOG-ODDS
# ============================================================
def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ============================================================
# POSTERIOR
# ============================================================
def conditional_series(df_feat: pd.DataFrame, horizon: int) -> tuple[pd.Series, dict]:
    """
    P(up | stato) per OGNI giorno, con le tabelle fittate sull'ultimo campione
    disponibile (i label incompleti in coda sono esclusi: niente look-ahead nella
    stima). E' la "ultima ri-stima" del processo validato OOS, valutata su tutta la
    storia — serve per il grafico e per misurare la stabilita' del segnale.
    """
    vix = df_feat["VIX"]
    up = (vix.shift(-horizon) > vix).to_numpy(dtype=float)
    up[pd.isna(vix.shift(-horizon).to_numpy())] = np.nan

    lvl, tsb, buck = cond_model.build_bucket_arrays(df_feat)
    valid = ~np.isnan(up)
    cond_tbl, lvl_tbl, base = cond_model.fit_conditional_tables(
        up[valid], buck[valid], lvl[valid])

    p = np.array([cond_model.predict_p_cond(cond_tbl, lvl_tbl, base, b, l)
                  for b, l in zip(buck, lvl)], dtype=float)
    meta = {
        "base_rate": round(float(base), 4),
        "table": {str(k): round(float(v), 4) for k, v in sorted(cond_tbl.items())},
        "counts": cond_model.bucket_counts(buck, valid),
        "bucket_now": str(buck[-1]),
        "level_now": str(lvl[-1]),
        "ts_now": str(tsb[-1]),
    }
    return pd.Series(p, index=df_feat.index, name=f"p_cond_{horizon}"), meta


def posterior_series(p_cond: pd.Series, p_prior: float,
                     weight: float | None = None) -> pd.Series:
    """logit(post) = logit(cond) + w * logit(prior). Il prior pesa meno: e' validazione."""
    w = config.SEASONAL_WEIGHT if weight is None else weight
    return pd.Series(sigmoid(logit(p_cond.to_numpy()) + w * float(logit(p_prior))),
                     index=p_cond.index, name=p_cond.name.replace("p_cond", "p_post"))


# ============================================================
# CONVICTION
# ============================================================
def compute_conviction(per_horizon: dict, p_series: pd.Series,
                       primary: int) -> dict:
    """
    Conviction da tre componenti indipendenti, ognuna in [0,1]:

        edge       quanto il posterior si allontana dal 50% (normalizzato sulla
                   banda neutra: 1.0 = due bande di distanza)
        agreement  frazione di orizzonti che indicano la stessa direzione del
                   primario. Tre orizzonti concordi valgono piu' di uno estremo.
        stability  frazione dei giorni recenti in cui il segnale aveva gia' lo
                   stesso segno. Un segnale che oscilla ogni giorno non e' un segnale.
    """
    p_primary = per_horizon[primary]["p_up"]
    direction_up = p_primary > 0.5
    band = config.NEUTRAL_BAND

    edge = float(min(1.0, abs(p_primary - 0.5) / (2 * band)))

    dirs = [(v["p_up"] > 0.5) == direction_up for v in per_horizon.values()]
    agreement = float(np.mean(dirs)) if dirs else 0.0

    recent = p_series.dropna().iloc[-config.STABILITY_WINDOW:]
    stability = (float(np.mean([(float(x) > 0.5) == direction_up for x in recent]))
                 if len(recent) else 0.0)

    score = float(np.clip(0.45 * edge + 0.35 * agreement + 0.20 * stability, 0.0, 1.0))
    if score >= config.CONVICTION_HIGH:
        label = "ALTA"
    elif score >= config.CONVICTION_MED:
        label = "MEDIA"
    elif abs(p_primary - 0.5) > band:
        label = "BASSA"
    else:
        label = "NULLA"

    return {"score": round(score, 3), "label": label,
            "edge": round(edge, 3), "agreement": round(agreement, 3),
            "stability": round(stability, 3),
            "components_note": "edge = distanza dal 50%; agreement = orizzonti concordi; "
                               "stability = persistenza del segno negli ultimi "
                               f"{config.STABILITY_WINDOW} giorni"}


# ============================================================
# BIAS OPERATIVO
# ============================================================
def operational_bias(p_up: float, regime_state: dict, vrp_latest: dict,
                     conviction: dict, vrp_validated: bool | None = None) -> dict:
    """
    Da segnali a bias qualitativo. Nessuno strike, nessuna scadenza, nessuna esecuzione.

    I due lati hanno edge di natura diversa, quindi gate asimmetrici:

      LONG_VOL   richiede un edge DIREZIONALE rialzista: comprare convessita' e'
                 premio bruciato se il VIX non si muove.
      SHORT_VOL  non richiede un call direzionale — il premio si incassa anche a
                 VIX piatto — ma richiede premio ricco confermato, regime calmo e
                 assenza di rischio-spike.
      FLAT       nessuno dei due: astensione. E' una posizione legittima.
    """
    band = config.NEUTRAL_BAND
    vix_rank = regime_state.get("vix_rank")
    vvix_rank = regime_state.get("vvix_rank")
    ts_state = regime_state.get("ts_state")
    vrp_state = vrp_latest.get("state")
    agree = bool(vrp_latest.get("agree", False))
    vrp_val = vrp_latest.get("vrp_consensus")

    up_edge = p_up > 0.5 + band
    down_lean = p_up < 0.5 - band
    spike_risk = up_edge or (vvix_rank is not None and vvix_rank >= config.VVIX_HIGH_PCTL)
    cheap_convexity = ((vix_rank is not None and vix_rank <= config.VIX_LOW_PCTL)
                       or vrp_state == "compresso")
    rich_confirmed = (vrp_state == "ricco") and agree
    calm = ts_state == "contango"

    long_primary = up_edge and (cheap_convexity or spike_risk)
    long_secondary = (not up_edge) and (p_up > 0.5) and cheap_convexity and spike_risk
    short_carry = rich_confirmed and calm and not spike_risk
    short_dir = down_lean and rich_confirmed and not spike_risk

    bias, reasons, caps = "FLAT", [], []

    if long_primary or long_secondary:
        bias = "LONG_VOL"
        reasons.append(
            "edge direzionale rialzista sul VIX con convessita' conveniente o rischio-spike"
            if long_primary else
            "convessita' a sconto e rischio-spike con lieve lean rialzista: "
            "posizione tattica a bassa convinzione")
        if long_secondary:
            caps.append("BASSA")
    elif short_carry or short_dir:
        bias = "SHORT_VOL"
        if short_carry:
            reasons.append("premio ricco (percentile alto, confermato dai due stimatori) "
                           "in regime contango senza rischio-spike: harvesting del carry")
        if short_dir:
            reasons.append("posterior coerente con VIX in rientro e premio ricco confermato")
        if vrp_validated is False:
            caps.append("BASSA")
            reasons.append("ATTENZIONE: il gate VRP non ha evidenza fuori campione "
                           "(spread di premio non significativo) — e' carry, non edge "
                           "dimostrato: dimensionare di conseguenza")
    else:
        if spike_risk and not up_edge:
            reasons.append("rischio-spike senza edge direzionale netto: niente short vol")
        elif vrp_state == "ricco" and not agree:
            reasons.append("premio nominalmente ricco ma i due stimatori non concordano "
                           "sul segno: astensione, premio non confermato")
        elif abs(p_up - 0.5) <= band:
            reasons.append(f"posterior dentro la banda neutra (±{band:.0%}): "
                           "nessun edge direzionale, il modello si astiene")
        else:
            reasons.append("nessun setup: ne' edge direzionale sfruttabile, "
                           "ne' carry confermato in regime calmo")

    label = conviction["label"] if bias != "FLAT" else "NULLA"
    if caps and label in ("ALTA", "MEDIA"):
        label = "BASSA"

    return {"bias": bias, "conviction": label,
            "conviction_score": conviction["score"],
            "rationale": ". ".join(reasons) + ".",
            "vrp_validated": vrp_validated}


# ============================================================
# ORCHESTRAZIONE
# ============================================================
def build_forecast(df_feat: pd.DataFrame, regime_latest: dict, vrp: dict,
                   seasonality: dict, calibrators: dict | None = None,
                   vrp_validated: bool | None = None) -> dict:
    """
    Forecast completo: posterior per orizzonte, conviction, bias, serie storica.

    `calibrators` e' una mappa {orizzonte: calibratore}: OGNI orizzonte usa il
    proprio, perche' il cancello held-out puo' ammetterlo su un orizzonte e
    respingerlo su un altro. Applicare il calibratore dell'orizzonte primario a
    tutti gli orizzonti — come faceva la prima versione — significa calibrare un
    modello con la mappa di un altro.
    """
    calibrators = calibrators or {}

    per_horizon, series = {}, {}
    cond_meta, cal_flags = {}, {}
    for h in config.FORECAST_HORIZONS:
        cal_h = calibrators.get(h) or calibrators.get(str(h))
        cal_flags[h] = bool(cal_h and not cal_h.get("identity") and cal_h.get("apply"))

        p_cond, meta = conditional_series(df_feat, h)
        prior = seasonality["priors"].get(h, {"p_up_prior": 0.5})
        p_post_raw = posterior_series(p_cond, prior.get("p_up_prior", 0.5))
        p_post = pd.Series(calibration.isotonic_apply(cal_h, p_post_raw.to_numpy()),
                           index=p_post_raw.index, name=p_post_raw.name)
        series[h] = p_post
        cond_meta[h] = meta
        per_horizon[h] = {
            "p_up": round(float(p_post.iloc[-1]), 4),
            "p_up_raw": round(float(p_post_raw.iloc[-1]), 4),
            "p_cond": round(float(p_cond.iloc[-1]), 4),
            "p_prior": prior.get("p_up_prior", 0.5),
            "bucket": meta["bucket_now"],
            "n_bucket": int(meta["counts"].get(meta["bucket_now"], 0)),
            "direction": "VIX su" if float(p_post.iloc[-1]) > 0.5 else "VIX giu'",
            "calibrated": cal_flags[h],
        }

    primary = config.PRIMARY_HORIZON
    p_up = per_horizon[primary]["p_up"]
    conviction = compute_conviction(per_horizon, series[primary], primary)

    regime_state = {
        "ts_state": cond_meta[primary]["ts_now"],
        "vix_rank": regime_latest.get("vix_rank"),
        "vvix_rank": regime_latest.get("vvix_rank"),
    }
    bias = operational_bias(p_up, regime_state, vrp["latest"], conviction, vrp_validated)

    cal_primary = calibrators.get(primary) or calibrators.get(str(primary)) or {}
    return {
        "horizon_primary": primary,
        "p_up_primary": p_up,
        "p_up_primary_raw": per_horizon[primary]["p_up_raw"],
        "calibrated": cal_flags[primary],
        "calibrated_by_horizon": {str(h): v for h, v in cal_flags.items()},
        "calibration_note": cal_primary.get("gate_reason", "nessun calibratore disponibile"),
        "direction": per_horizon[primary]["direction"],
        "per_horizon": {str(h): v for h, v in per_horizon.items()},
        "conviction": conviction,
        "operational": bias,
        "cond_detail": {str(h): {"base_rate": m["base_rate"], "bucket": m["bucket_now"],
                                 "n": int(m["counts"].get(m["bucket_now"], 0))}
                        for h, m in cond_meta.items()},
        "p_series": series,   # non serializzato: usato dalla pipeline per il parquet
    }
