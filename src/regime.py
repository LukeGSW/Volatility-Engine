"""
regime.py — Regime di volatilita' da ensemble di flag trasparenti.

Niente HMM come oracolo: quattro flag leggibili (term structure, percentile VIX,
vol-of-vol, trend), ciascuno normalizzato in [0,1], media pesata = stress_score,
soglie -> etichetta. Robusto, debuggabile, e cattura la gran parte del valore.

Rispetto al forecaster originale:
  * vettorizzato (niente .apply riga per riga: ~100x piu' veloce sul walk-forward);
  * estrazione degli EPISODI storici, cioe' la risposta esplicita alla domanda
    "quali sono stati i regimi passati e come sono finiti".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import linmap


def _flag_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Contributo di stress (0..1) di ogni flag. Vettorizzato, NaN preservati."""
    flags = pd.DataFrame(index=df.index)

    ts = df.get("ts_ratio")
    if ts is not None:
        v = linmap(ts.to_numpy(), config.TS_DEEP_CONTANGO, config.TS_STEEP_BACKW)
        flags["term_structure"] = np.where(np.isnan(ts.to_numpy()), np.nan, v)
    else:
        flags["term_structure"] = np.nan

    flags["vix_level"] = df.get("vix_rank", pd.Series(np.nan, index=df.index))
    flags["vvix"] = df.get("vvix_rank", pd.Series(np.nan, index=df.index))

    mar = df.get("vix_ma_ratio")
    if mar is not None:
        v = linmap(mar.to_numpy(), 1.00, 1.15)
        flags["trend"] = np.where(np.isnan(mar.to_numpy()), np.nan, v)
    else:
        flags["trend"] = np.nan

    return flags


def classify_regime(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge flag_*, stress_score (0..1) e regime (0/1/2)."""
    out = df_feat.copy()
    flags = _flag_frame(out)
    for c in flags.columns:
        out[f"flag_{c}"] = flags[c]

    cols = list(config.FLAG_WEIGHTS)
    w = np.array([config.FLAG_WEIGHTS[c] for c in cols], dtype=float)
    mat = flags[cols].to_numpy(dtype=float)
    mask = ~np.isnan(mat)

    denom = (mask * w).sum(axis=1)
    denom = np.where(denom == 0, np.nan, denom)
    stress = np.nansum(np.where(mask, mat, 0.0) * w, axis=1) / denom
    out["stress_score"] = stress

    regime = np.full(len(out), 1.0)
    regime[stress < config.STRESS_CALM_MAX] = 0.0
    regime[stress >= config.STRESS_STRESS_MIN] = 2.0
    # La backwardation marcata forza lo stato di stress a prescindere dalla media.
    ts = out["ts_ratio"].to_numpy(dtype=float) if "ts_ratio" in out.columns \
        else np.full(len(out), np.nan)
    regime[np.nan_to_num(ts, nan=0.0) >= config.TS_STEEP_BACKW] = 2.0
    regime[np.isnan(stress)] = np.nan
    out["regime"] = regime

    return out


def latest_state(df_regime: pd.DataFrame) -> dict:
    """Stato di regime piu' recente, serializzabile."""
    row = df_regime.iloc[-1]

    def _f(key, nd=4):
        v = row.get(key, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(v) else round(v, nd)

    idx = row.get("regime", np.nan)
    idx = 1 if (idx is None or (isinstance(idx, float) and np.isnan(idx))) else int(idx)

    return {
        "date": pd.Timestamp(df_regime.index[-1]).date().isoformat(),
        "regime_idx": idx,
        "regime_label": config.REGIME_LABELS[idx],
        "stress_score": _f("stress_score"),
        "vix": _f("VIX", 2),
        "vix3m": _f("VIX3M", 2),
        "vix9d": _f("VIX9D", 2),
        "vvix": _f("VVIX", 2),
        "ts_ratio": _f("ts_ratio"),
        "ts_short": _f("ts_short"),
        "vix_rank": _f("vix_rank"),
        "vvix_rank": _f("vvix_rank"),
        "vvix_z": _f("vvix_z", 3),
        "vix_range": _f("vix_range", 2),
        "realized_vol": _f("realized_vol", 2),
        "vrp_proxy": _f("vrp_proxy", 2),
        "vrp_proxy_rank": _f("vrp_proxy_rank"),
        "flags": {c.replace("flag_", ""): _f(c)
                  for c in df_regime.columns if c.startswith("flag_")},
    }


def current_context(df_regime: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """
    Da quante sessioni dura il regime IN CORSO, e qual era quello prima.

    La durata si conta sulla serie grezza, non sull'ultimo episodio: gli episodi
    assorbono i tratti piu' corti di `min_days`, quindi l'ultimo episodio puo' avere
    un'etichetta diversa dal regime di oggi. Prendendola da li' l'alert direbbe
    "Calmo da 7 sessioni" mentre l'ultimo episodio e' etichettato Transizione.

    Il regime precedente e' l'ultimo episodio con etichetta DIVERSA da quella
    corrente: dire "prima: Calmo" quando siamo in Calmo non informa nessuno.
    """
    r = df_regime["regime"].dropna()
    if r.empty:
        return {}

    current = int(r.iloc[-1])
    streak = int((r.values[::-1] != current).argmax()) if (r.values != current).any() else len(r)

    out = {"days_in_regime": streak}
    if episodes is not None and not episodes.empty:
        prev = episodes[episodes["regime_idx"] != current]
        if not prev.empty:
            last = prev.iloc[-1]
            out["prev_regime_label"] = str(last["label"])
            out["prev_regime_days"] = int(last["days"])
    return out


def regime_episodes(df_regime: pd.DataFrame, min_days: int = 3) -> pd.DataFrame:
    """
    Storico dei regimi come EPISODI, non come punti.

    Per ogni tratto continuo dello stesso regime: date, durata, VIX di ingresso e
    uscita, VIX massimo raggiunto, variazione dell'S&P nel periodo. E' la risposta
    diretta alla domanda "quali sono stati i regimi passati e cosa e' successo".
    Gli episodi piu' corti di `min_days` vengono assorbiti nel precedente: senza
    questo filtro l'oscillazione attorno alle soglie genera decine di micro-episodi.
    """
    r = df_regime["regime"].dropna()
    if r.empty:
        return pd.DataFrame(columns=["regime_idx", "label", "start", "end", "days",
                                     "vix_start", "vix_end", "vix_max", "spx_chg_pct"])

    grp = (r != r.shift()).cumsum()
    rows = []
    for _, seg in r.groupby(grp):
        start, end = seg.index[0], seg.index[-1]
        n = len(seg)
        idx = int(seg.iloc[0])
        # Un tratto piu' corto di min_days viene assorbito nel precedente: senza questo
        # filtro l'oscillazione attorno alle soglie genera decine di micro-episodi.
        if n < min_days and rows:
            rows[-1]["end"] = end
            rows[-1]["days"] += n
            continue
        # Dopo un assorbimento due tratti dello STESSO regime possono ritrovarsi
        # adiacenti: vanno fusi, altrimenti la tabella mostra lo stesso episodio
        # spezzato in due righe.
        if rows and rows[-1]["regime_idx"] == idx:
            rows[-1]["end"] = end
            rows[-1]["days"] += n
            continue
        rows.append({"regime_idx": idx, "start": start, "end": end, "days": n})

    vix = df_regime["VIX"]
    spx = df_regime.get("SPX")
    out = []
    for e in rows:
        window = df_regime.loc[e["start"]:e["end"]]
        rec = {
            "regime_idx": e["regime_idx"],
            "label": config.REGIME_LABELS[e["regime_idx"]],
            "start": e["start"],
            "end": e["end"],
            "days": int(len(window)),
            "vix_start": round(float(vix.loc[e["start"]]), 2),
            "vix_end": round(float(vix.loc[e["end"]]), 2),
            "vix_max": round(float(window["VIX"].max()), 2),
        }
        if spx is not None and window["SPX"].notna().any():
            s = window["SPX"].dropna()
            rec["spx_chg_pct"] = round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 2)
        else:
            rec["spx_chg_pct"] = np.nan
        out.append(rec)

    return pd.DataFrame(out)
