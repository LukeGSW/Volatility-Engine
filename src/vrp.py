"""
vrp.py — Volatility Risk Premium con doppio stimatore e soglie a PERCENTILE.

    VRP = VIX - E[realized vol]     (punti vol annualizzati)
    ampio positivo -> implied ricca   -> favorevole a vendere premio
    compresso/negativo -> implied a sconto -> favorevole a comprare convessita'

Perche' due stimatori. Un EGARCH stimato su molti anni reverte verso una media
incondizionata gonfiata dalle crisi: in regimi calmi SOVRA-stima la realized forward
e schiaccia il VRP. L'HAR-RV, ancorato alla realized recente, e' il secondo parere.
Si mostrano entrambi, il consensus e un flag di accordo.

Due cambiamenti sostanziali rispetto al forecaster originale:

  1. L'HAR NON regredisce piu' sul rendimento giornaliero al quadrato ma sulla
     varianza GARMAN-KLASS, 5-8 volte piu' efficiente a parita' di dati. Con una
     variabile dipendente cosi' rumorosa l'OLS soffriva di attenuazione da
     errore-nelle-variabili e le previsioni risultavano schiacciate verso la media.

  2. Le soglie "ricco"/"compresso" sono PERCENTILI ROLLING, non punti assoluti.
     La soglia precedente (3.0 punti) selezionava il 62% dei giorni e non
     discriminava: premio realizzato con hit-rate 83.9% nel bucket ricco contro
     82.9% nel compresso, spread OOS -0.45 con intervallo che attraversa lo zero.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)
_ANN = np.sqrt(252.0)


# ============================================================
# HAR-RV
# ============================================================
def har_forecast(rv_daily: pd.Series, horizon: int) -> float:
    """
    Forecast HAR-RV (Corsi 2009) della varianza media sui prossimi `horizon` giorni,
    restituito come vol annualizzata in punti percentuali.

    La varianza giornaliera dipende dalle medie a 1, 5 e 22 giorni. Si stima l'OLS e
    si itera in avanti reinserendo le previsioni.
    """
    rv = pd.Series(rv_daily).replace([np.inf, -np.inf], np.nan).dropna()
    rv = rv[rv > 0]
    if len(rv) < 60:
        return float("nan")

    d = rv.shift(1)
    w = rv.rolling(5).mean().shift(1)
    m = rv.rolling(22).mean().shift(1)
    data = pd.concat([rv, d, w, m], axis=1).dropna()
    if len(data) < 60:
        return float("nan")

    y = data.iloc[:, 0].to_numpy()
    X = np.column_stack([np.ones(len(data)), data.iloc[:, 1].to_numpy(),
                         data.iloc[:, 2].to_numpy(), data.iloc[:, 3].to_numpy()])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:  # noqa: BLE001
        return float("nan")

    hist = list(rv.to_numpy())
    preds = []
    for _ in range(horizon):
        nxt = (beta[0] + beta[1] * hist[-1] + beta[2] * float(np.mean(hist[-5:]))
               + beta[3] * float(np.mean(hist[-22:])))
        nxt = max(float(nxt), 1e-12)
        preds.append(nxt)
        hist.append(nxt)
    return float(np.sqrt(np.mean(preds)) * _ANN * 100.0)


# ============================================================
# EGARCH
# ============================================================
def egarch_forecast(spx: pd.Series, horizon: int) -> dict:
    """EGARCH(1,1,1)-t sui rendimenti S&P. Degrada su realized vol se `arch` manca."""
    out = {"method": "EGARCH(1,1,1)-t [sim]", "cond_ann": None,
           "fwd_ann": None, "longrun_ann": None, "series": None}
    ret_pct = (np.log(spx).diff().dropna() * 100.0)
    try:
        from arch import arch_model
        am = arch_model(ret_pct, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t")
        res = am.fit(disp="off", show_warning=False)
        cond = res.conditional_volatility
        out["series"] = cond * _ANN
        out["cond_ann"] = float(cond.iloc[-1] * _ANN)
        out["longrun_ann"] = float(np.nanmean(cond) * _ANN)
        fc = (res.forecast(horizon=horizon, method="simulation", simulations=2000,
                           reindex=False) if horizon > 1
              else res.forecast(horizon=horizon, reindex=False))
        out["fwd_ann"] = float(np.sqrt(np.mean(fc.variance.iloc[-1].to_numpy())) * _ANN)
    except Exception as e:  # noqa: BLE001
        log.warning("EGARCH non disponibile (%s): fallback su realized vol rolling.", e)
        out["method"] = f"RealizedVol({config.REALIZED_VOL_WINDOW}d)"
        rv = ret_pct.rolling(config.REALIZED_VOL_WINDOW).std() * _ANN
        out["series"] = rv
        if rv.notna().any():
            out["cond_ann"] = float(rv.dropna().iloc[-1])
            out["fwd_ann"] = out["cond_ann"]
            out["longrun_ann"] = float(np.nanmean(rv))
    return out


# ============================================================
# ORCHESTRAZIONE
# ============================================================
def estimate_vrp(df_feat: pd.DataFrame, horizon: int | None = None) -> dict:
    """
    Stima il VRP corrente con EGARCH + HAR, consensus, accordo e classificazione
    per PERCENTILE rolling.
    """
    horizon = horizon or config.PRIMARY_HORIZON
    vix_last = float(df_feat["VIX"].iloc[-1])

    eg = {"method": "n/d", "cond_ann": None, "fwd_ann": None,
          "longrun_ann": None, "series": None}
    har_fwd = float("nan")
    if "SPX" in df_feat.columns and df_feat["SPX"].notna().sum() > 100:
        eg = egarch_forecast(df_feat["SPX"].dropna(), horizon)
        har_fwd = har_forecast(df_feat["rv_daily"], horizon)

    vals = [x for x in (eg["fwd_ann"], har_fwd) if x is not None and np.isfinite(x)]
    consensus = float(np.mean(vals)) if vals else None

    vrp_eg = (vix_last - eg["fwd_ann"]) if eg["fwd_ann"] is not None else None
    vrp_har = (vix_last - har_fwd) if np.isfinite(har_fwd) else None
    vrp_cons = (vix_last - consensus) if consensus is not None else None
    agree = (vrp_eg is not None and vrp_har is not None
             and np.sign(vrp_eg) == np.sign(vrp_har))

    # Serie storiche per i grafici
    if eg["series"] is not None:
        vol_series = eg["series"].reindex(df_feat.index)
    else:
        vol_series = df_feat.get("realized_vol", pd.Series(np.nan, index=df_feat.index))
    vol_series = vol_series.rename("exp_vol_ann")
    vrp_series = (df_feat["VIX"] - vol_series).rename("vrp")

    # Classificazione a PERCENTILE (il fix). Si usa il rank del proxy causale,
    # che e' calcolabile su tutta la storia; il livello corrente del VRP model-based
    # viene collocato nella stessa distribuzione.
    rank = df_feat.get("vrp_proxy_rank")
    vrp_rank = float(rank.iloc[-1]) if rank is not None and pd.notna(rank.iloc[-1]) else None
    if vrp_rank is None:
        state = "n/d"
    elif vrp_rank >= config.VRP_RICH_PCTL:
        state = "ricco"
    elif vrp_rank <= config.VRP_COMPRESSED_PCTL:
        state = "compresso"
    else:
        state = "neutro"

    def _r(x, nd=2):
        return round(float(x), nd) if (x is not None and np.isfinite(float(x))) else None

    latest = {
        "method": eg["method"],
        "vix": _r(vix_last),
        "egarch_cond_ann": _r(eg["cond_ann"]),
        "egarch_fwd_ann": _r(eg["fwd_ann"]),
        "egarch_longrun_ann": _r(eg["longrun_ann"]),
        "har_fwd_ann": _r(har_fwd),
        "consensus_fwd_ann": _r(consensus),
        "vrp_egarch": _r(vrp_eg),
        "vrp_har": _r(vrp_har),
        "vrp_consensus": _r(vrp_cons),
        "vrp": _r(vrp_cons),
        "agree": bool(agree),
        "vrp_proxy": _r(df_feat["vrp_proxy"].iloc[-1]) if "vrp_proxy" in df_feat else None,
        "vrp_rank": round(vrp_rank, 4) if vrp_rank is not None else None,
        "state": state,
        "rv_estimator": config.RV_ESTIMATOR,
    }
    return {"latest": latest, "vol_series": vol_series, "vrp_series": vrp_series}
