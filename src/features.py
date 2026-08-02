"""
features.py — Feature engineering, tutto rolling e causale.

Regola invariabile: ogni statistica al tempo t usa esclusivamente dati fino a t.
Percentili e z-score sono rolling (non full-sample), quindi le feature sono lecite
in un walk-forward; il leakage possibile resta solo nella label e nella stima delle
tabelle, ed e' gestito in walkforward.py con purging + embargo.

Novita' rispetto ai due repository originali:

  * REALIZED VOL RANGE-BASED (Garman-Klass / Yang-Zhang) sull'OHLC giornaliero.
    Il proxy usato prima — il rendimento giornaliero al quadrato — e' corretto ma
    rumorosissimo, e l'HAR che ci regredisce sopra soffre di attenuazione da
    errore-nelle-variabili. Garman-Klass rende 5-8 volte l'efficienza a parita' di
    dati, copre l'intero campione e non richiede intraday (che partirebbe dal 2015
    e dimezzerebbe la finestra di validazione).

  * VVIX LOG-ZSCORE, la feature del repo KQ-VVIX assente dal forecaster, che del
    VVIX usava solo il percentile a 2 anni buttando via la componente veloce.

  * RANGE E GAP DEL VIX dall'OHLC CBOE (reale dal 1992, mediana 7.0% del livello):
    vol-of-vol REALIZZATA, contro il VVIX che e' implicito e parte dal 2006.

  * SHORT-END DELLA CURVA (VIX9D/VIX), piu' reattivo di VIX/VIX3M e quindi il
    candidato naturale per l'orizzonte a 5 giorni.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_ANN = np.sqrt(252.0)


# ============================================================
# UTILITY
# ============================================================
def rolling_rank(s: pd.Series, window: int) -> pd.Series:
    """
    Percentile (0..1) del valore corrente nella finestra trailing.
    Rolling, non full-sample: usa solo il passato.
    """
    def _rank(x: np.ndarray) -> float:
        return float((x <= x[-1]).mean())
    return s.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=True)


def rolling_log_z(s: pd.Series, window: int) -> pd.Series:
    """
    Log-zScore rolling. Il logaritmo simmetrizza distribuzioni con floor e senza cap
    (VVIX, VIX), rendendo +N e -N deviazioni confrontabili.
    min_periods=window -> nessun look-ahead, NaN sulle prime (window-1) barre.
    """
    lv = np.log(s.where(s > 0))
    m = lv.rolling(window, min_periods=window).mean()
    sd = lv.rolling(window, min_periods=window).std(ddof=1).replace(0.0, np.nan)
    return (lv - m) / sd


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def linmap(x, lo: float, hi: float):
    """Mappa lineare [lo, hi] -> [0, 1] con clipping."""
    if hi == lo:
        return np.zeros_like(np.asarray(x, dtype=float))
    return _clip01((np.asarray(x, dtype=float) - lo) / (hi - lo))


# ============================================================
# REALIZED VOLATILITY
# ============================================================
def rv_close_to_close(close: pd.Series) -> pd.Series:
    """Varianza giornaliera da rendimento log al quadrato. Corretta ma rumorosa."""
    r = np.log(close).diff()
    return (r ** 2).rename("rv_cc")


def rv_garman_klass(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """
    Varianza giornaliera Garman-Klass:
        0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2
    Usa l'intera escursione della giornata invece del solo close: a parita' di
    campione l'efficienza e' ~7.4 volte quella del close-to-close.
    """
    hl = np.log(h / l) ** 2
    co = np.log(c / o) ** 2
    gk = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    return gk.clip(lower=0.0).rename("rv_gk")


def rv_yang_zhang(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
                  window: int = 21) -> pd.Series:
    """
    Varianza Yang-Zhang su finestra: gestisce gap overnight e drift, che
    Garman-Klass ignora. Ritorna la varianza GIORNALIERA media sulla finestra.
    """
    c_prev = c.shift(1)
    ro = np.log(o / c_prev)          # overnight
    rc = np.log(c / o)               # open-to-close
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)  # Rogers-Satchell

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v_o = ro.rolling(n, min_periods=n).var(ddof=1)
    v_c = rc.rolling(n, min_periods=n).var(ddof=1)
    v_rs = rs.rolling(n, min_periods=n).mean()
    return (v_o + k * v_c + (1 - k) * v_rs).clip(lower=0.0).rename("rv_yz")


def _debias_to_close(rv: pd.Series, rv_cc: pd.Series,
                     min_periods: int = 252) -> pd.Series:
    """
    Corregge il bias di LIVELLO degli stimatori range-based, mantenendone l'efficienza.

    Con un numero finito di osservazioni intragiornaliere il massimo e il minimo
    OSSERVATI sono piu' vicini fra loro di quelli veri in tempo continuo, quindi
    Garman-Klass sottostima: su un browniano con sigma nota il test misura -4.7%.
    Il close-to-close, per quanto rumoroso, e' invece non distorto. Si riscala
    percio' il range-based sul rapporto delle medie storiche.

    Causale per costruzione: il fattore al tempo t usa una media ESPANDENTE fino a t,
    mai il futuro. Sul ranking percentile del VRP un fattore moltiplicativo e'
    ininfluente, ma il VRP mostrato in dashboard e' in punti vol: li' conta.
    """
    num = rv_cc.expanding(min_periods=min_periods).mean()
    den = rv.expanding(min_periods=min_periods).mean().replace(0.0, np.nan)
    factor = (num / den).bfill().fillna(1.0).clip(lower=0.5, upper=2.0)
    return (rv * factor).rename(rv.name)


def daily_variance(df: pd.DataFrame, prefix: str = "SPX",
                   estimator: str | None = None, debias: bool = True) -> pd.Series:
    """
    Varianza giornaliera con lo stimatore configurato, con degradazione elegante
    se l'OHLC non e' disponibile per quella serie.
    """
    estimator = estimator or config.RV_ESTIMATOR
    c = df[prefix]
    o, h, l = (df.get(f"{prefix}_{k}") for k in ("open", "high", "low"))
    has_ohlc = all(x is not None and x.notna().any() for x in (o, h, l))

    rv_cc = rv_close_to_close(c)
    if estimator == "cc" or not has_ohlc:
        return rv_cc
    if estimator == "yz":
        rv = rv_yang_zhang(o, h, l, c, config.REALIZED_VOL_WINDOW).fillna(
            rv_garman_klass(o, h, l, c))
    else:
        rv = rv_garman_klass(o, h, l, c)
    rv = rv.fillna(rv_cc)
    return _debias_to_close(rv, rv_cc) if debias else rv


def annualized_vol(daily_var: pd.Series, window: int) -> pd.Series:
    """Varianza giornaliera -> vol annualizzata in punti percentuali."""
    return np.sqrt(daily_var.rolling(window, min_periods=max(5, window // 2)).mean()) * _ANN * 100.0


# ============================================================
# FEATURE SET
# ============================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce il dataset con tutte le feature causali.

    Colonne aggiunte:
        ts_ratio ts_slope ts_short          term structure (lungo e corto)
        vix_rank vvix_rank                  percentili rolling 2 anni
        vvix_z vix_z                        log-zScore rolling 90gg
        vix_ma_ratio vix_chg_*              trend e momentum
        vix_range vix_range_z vix_gap       vol-of-vol REALIZZATA da OHLC
        rv_daily realized_vol vrp_proxy     realized vol e premio implicito
        vrp_proxy_rank                      percentile rolling del VRP  <- il fix
        skew_rank move_rank vxn_spread      cross-asset (se disponibili)
    """
    out = df.copy()

    # --- Term structure ---
    if "VIX3M" in out.columns:
        out["ts_ratio"] = out["VIX"] / out["VIX3M"]
        out["ts_slope"] = out["VIX3M"] - out["VIX"]
    else:
        out["ts_ratio"] = np.nan
        out["ts_slope"] = np.nan

    # Short-end: piu' reattivo, e' il candidato per l'orizzonte a 5 giorni.
    out["ts_short"] = (out["VIX9D"] / out["VIX"]) if "VIX9D" in out.columns else np.nan

    # --- Livelli: percentili rolling ---
    out["vix_rank"] = rolling_rank(out["VIX"], config.RANK_LOOKBACK)
    out["vvix_rank"] = (rolling_rank(out["VVIX"], config.RANK_LOOKBACK)
                        if "VVIX" in out.columns else np.nan)

    # --- Log-zScore (la feature del repo KQ-VVIX) ---
    out["vvix_z"] = (rolling_log_z(out["VVIX"], config.VVIX_Z_WINDOW)
                     if "VVIX" in out.columns else np.nan)
    out["vix_z"] = rolling_log_z(out["VIX"], config.VVIX_Z_WINDOW)

    # --- Trend e momentum ---
    ma = out["VIX"].rolling(config.MEDIUM_WINDOWS[0], min_periods=10).mean()
    out["vix_ma_ratio"] = out["VIX"] / ma
    for w in config.SHORT_WINDOWS:
        out[f"vix_chg_{w}"] = out["VIX"].pct_change(w)

    # --- Range e gap del VIX: vol-of-vol realizzata, gratis dall'OHLC ---
    if {"VIX_high", "VIX_low"} <= set(out.columns):
        out["vix_range"] = (out["VIX_high"] - out["VIX_low"]) / out["VIX"] * 100.0
        out["vix_range_z"] = rolling_log_z(out["vix_range"], config.VVIX_Z_WINDOW)
    else:
        out["vix_range"] = np.nan
        out["vix_range_z"] = np.nan
    if "VIX_open" in out.columns:
        out["vix_gap"] = (out["VIX_open"] / out["VIX"].shift(1) - 1.0) * 100.0
    else:
        out["vix_gap"] = np.nan

    # --- Realized vol e VRP ---
    if "SPX" in out.columns:
        out["rv_daily"] = daily_variance(out, "SPX")
        out["realized_vol"] = annualized_vol(out["rv_daily"], config.REALIZED_VOL_WINDOW)
    else:
        out["rv_daily"] = np.nan
        out["realized_vol"] = np.nan

    out["vrp_proxy"] = out["VIX"] - out["realized_vol"]
    # IL FIX: percentile rolling invece della soglia assoluta a 3 punti, che
    # catturava il 62% dei giorni e non discriminava nulla.
    out["vrp_proxy_rank"] = rolling_rank(out["vrp_proxy"], config.VRP_RANK_LOOKBACK)

    # --- Cross-asset (raccolte ora, valutate come feature in Fase 2) ---
    if "SKEW" in out.columns:
        out["skew_rank"] = rolling_rank(out["SKEW"], config.RANK_LOOKBACK)
    if "MOVE" in out.columns:
        out["move_rank"] = rolling_rank(out["MOVE"], config.RANK_LOOKBACK)
    if "VXN" in out.columns:
        out["vxn_spread"] = out["VXN"] - out["VIX"]
    if "OVX" in out.columns:
        out["ovx_rank"] = rolling_rank(out["OVX"], config.RANK_LOOKBACK)

    return out


def feature_summary(df_feat: pd.DataFrame) -> dict:
    """Diagnostica: copertura di ogni feature. Serve a smascherare colonne morte."""
    cols = ["ts_ratio", "ts_short", "vix_rank", "vvix_rank", "vvix_z", "vix_z",
            "vix_ma_ratio", "vix_range", "vix_range_z", "vix_gap", "realized_vol",
            "vrp_proxy", "vrp_proxy_rank", "skew_rank", "move_rank", "vxn_spread"]
    out = {}
    n = len(df_feat)
    for c in cols:
        if c not in df_feat.columns:
            continue
        s = df_feat[c].dropna()
        if s.empty:
            out[c] = {"coverage": 0.0, "first": None}
            continue
        out[c] = {
            "coverage": round(len(s) / n, 4),
            "first": pd.Timestamp(s.index[0]).date().isoformat(),
            "median": round(float(s.median()), 4),
        }
    return out
