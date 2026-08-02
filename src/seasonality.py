"""
seasonality.py — Stagionalita' annuale del VIX come VALIDAZIONE, non come guida.

Trappola numero uno, gestita esplicitamente: mediare il LIVELLO grezzo del VIX per
giorno di calendario su 10 anni misura solo dove sono caduti per caso gli shock
(feb-2018, mar-2020). Si DE-REGIMIZZA prima di mediare — ogni valore espresso come
rapporto alla mediana mobile annuale — cosi' si isola la componente ricorrente da
quella di regime.

Per l'uso operativo non conta la stagionalita' del livello ma quella della VARIAZIONE
forward. Con ~10 campioni per punto si mostra sempre la dispersione e si applica
shrinkage bayesiano: niente conclusioni overconfident.

Nota: l'ablazione del sistema precedente mostrava che il prior stagionale sposta
l'accuratezza di +0.36 punti ma PEGGIORA il Brier skill score (0.0453 con, 0.0555
senza). Il peso resta configurabile e la validazione lo rimisura a ogni run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_PERIOD = 365


# ============================================================
# UTILITY
# ============================================================
def _deregime(vix: pd.Series, window: int = 252) -> pd.Series:
    med = vix.rolling(window, min_periods=60).median()
    return (vix / med).rename("deregimed")


def _circ_dist(a: np.ndarray, b: int, period: int = _PERIOD) -> np.ndarray:
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def _forward_log_change(vix: pd.Series, w: int) -> pd.Series:
    return np.log(vix.shift(-w) / vix)


def _harmonic_smooth(doy: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Regressione armonica di Fourier: evita di overfittare 365 punti rumorosi."""
    valid = ~np.isnan(y)
    if valid.sum() < (2 * k + 2):
        return y
    phase = 2 * np.pi * (doy / _PERIOD)
    cols = [np.ones_like(phase)]
    for j in range(1, k + 1):
        cols += [np.sin(j * phase), np.cos(j * phase)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
    return X @ beta


# ============================================================
# CURVE
# ============================================================
def build_level_curve(vix: pd.Series, lookback_years: int, harmonics: int,
                      bootstrap: int, bucket: int = 3) -> pd.DataFrame:
    """Curva annuale del VIX de-regimizzato con bande di confidenza bootstrap."""
    s = _deregime(vix).dropna()
    s = s[s.index >= s.index.max() - pd.Timedelta(days=365 * lookback_years)]
    doy = s.index.dayofyear.to_numpy()
    vals = s.to_numpy()
    rng = np.random.default_rng(42)

    rows = []
    for d in range(1, _PERIOD + 1):
        sample = vals[_circ_dist(doy, d) <= bucket]
        sample = sample[~np.isnan(sample)]
        n = len(sample)
        if n >= 5:
            boot = rng.choice(sample, size=(bootstrap, n), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [5, 95])
            rows.append((d, float(sample.mean()), float(lo), float(hi), n))
        else:
            rows.append((d, float(sample.mean()) if n else np.nan, np.nan, np.nan, n))

    curve = pd.DataFrame(rows, columns=["doy", "mean_ratio", "ci_low", "ci_high", "n"]
                         ).set_index("doy")
    curve["smooth"] = _harmonic_smooth(curve.index.to_numpy(dtype=float),
                                       curve["mean_ratio"].to_numpy(), harmonics)
    return curve


def build_forward_curve(vix: pd.Series, horizon: int, lookback_years: int,
                        bucket: int = 3) -> pd.DataFrame:
    """Stagionalita' della variazione forward: quello che conta operativamente."""
    fwd = _forward_log_change(vix, horizon)
    mask = vix.index >= vix.index.max() - pd.Timedelta(days=365 * lookback_years)
    doy = vix.index.dayofyear.to_numpy()[mask]
    vals = fwd.to_numpy()[mask]

    rows = []
    for d in range(1, _PERIOD + 1):
        sample = vals[_circ_dist(doy, d) <= bucket]
        sample = sample[~np.isnan(sample)]
        n = len(sample)
        rows.append((d, float(sample.mean()) * 100 if n >= 5 else np.nan,
                     float((sample > 0).mean()) if n >= 5 else np.nan, n))
    return pd.DataFrame(rows, columns=["doy", "fwd_mean_pct", "hit_rate", "n"]
                        ).set_index("doy")


def multi_window_scan(vix: pd.Series, target_doy: int, windows: list[int],
                      lookback_years: int, bucket: int = 4) -> dict:
    """
    Tilt stagionale su piu' finestre. La CONSISTENZA del segno tra finestre e' il
    filtro anti-data-snooping: se cambia segno a ogni orizzonte, e' rumore.
    """
    mask = vix.index >= vix.index.max() - pd.Timedelta(days=365 * lookback_years)
    doy_all = vix.index.dayofyear.to_numpy()[mask]
    sel = _circ_dist(doy_all, target_doy) <= bucket

    out = {}
    for w in windows:
        sample = _forward_log_change(vix, w).to_numpy()[mask][sel]
        sample = sample[~np.isnan(sample)]
        n = len(sample)
        out[w] = {"mean_pct": round(float(sample.mean()) * 100, 3) if n else None,
                  "hit_rate": round(float((sample > 0).mean()), 3) if n else None,
                  "n": int(n)}
    signs = [np.sign(out[w]["mean_pct"]) for w in windows if out[w]["mean_pct"] is not None]
    out["consistency"] = round(float(np.mean([s == signs[0] for s in signs])), 3) if signs else 0.0
    return out


# ============================================================
# PRIOR
# ============================================================
def seasonal_prior(vix: pd.Series, horizon: int, lookback_years: int,
                   asof: pd.Timestamp, bucket: int = 4, pseudo: float = 10.0) -> dict:
    """Prior P(up) dalla stagionalita' forward, con shrinkage verso 0.5."""
    target = int(pd.Timestamp(asof).dayofyear)
    mask = vix.index >= vix.index.max() - pd.Timedelta(days=365 * lookback_years)
    doy_all = vix.index.dayofyear.to_numpy()[mask]
    fwd = _forward_log_change(vix, horizon).to_numpy()[mask]
    sample = fwd[_circ_dist(doy_all, target) <= bucket]
    sample = sample[~np.isnan(sample)]
    n = len(sample)

    if n == 0:
        return {"p_up_prior": 0.5, "hit_rate_raw": None, "mean_pct": None,
                "n": 0, "tilt": "neutro"}

    hit = float((sample > 0).mean())
    mean_pct = float(sample.mean()) * 100
    p_up = (pseudo * 0.5 + n * hit) / (pseudo + n)
    tilt = "neutro" if abs(mean_pct) < 0.5 else (
        "rialzista (VIX su)" if mean_pct > 0 else "ribassista (VIX giu')")
    return {"p_up_prior": round(float(p_up), 4), "hit_rate_raw": round(hit, 4),
            "mean_pct": round(mean_pct, 3), "n": int(n), "tilt": tilt}


def build_seasonality(df: pd.DataFrame, asof: pd.Timestamp | None = None) -> dict:
    """Tutti i prodotti stagionali per la pipeline."""
    vix = df["VIX"].dropna()
    asof = pd.Timestamp(asof or vix.index.max())
    return {
        "level_curve": build_level_curve(
            vix, config.SEASONALITY_LOOKBACK_YEARS, config.SEASONALITY_HARMONICS,
            config.SEASONALITY_BOOTSTRAP),
        "fwd_curve": build_forward_curve(
            vix, config.PRIMARY_HORIZON, config.SEASONALITY_LOOKBACK_YEARS),
        "scan": multi_window_scan(
            vix, int(asof.dayofyear), config.SEASONALITY_SCAN_WINDOWS,
            config.SEASONALITY_LOOKBACK_YEARS),
        "priors": {h: seasonal_prior(vix, h, config.SEASONALITY_LOOKBACK_YEARS, asof)
                   for h in config.FORECAST_HORIZONS},
    }
