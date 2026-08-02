"""
eventstudy.py — Motore di event study generalizzato (dal repo KQ-VVIX-Dashboard).

Nel repository originale l'event study era un'analisi a se': rilevava gli estremi del
VVIX e ne misurava il seguito. Qui diventa un motore riutilizzabile che accetta
QUALSIASI insieme di date-evento — estremi VVIX, ingressi in un regime, giorni in cui
il modello supera una soglia di probabilita' — e produce sempre la stessa cosa: la
distribuzione dei rendimenti forward, confrontata col drift incondizionato, con
significativita' overlap-aware e correzione per test multipli.

Cosi' l'event study smette di essere uno studio separato e diventa il modo in cui il
sistema RISPONDE alla domanda "cosa e' successo storicamente in condizioni come
oggi" — cioe' il forecast di lettura piu' immediata che si possa dare.

Due correzioni rispetto all'originale:
  * il cooldown e' PER TIPO di segnale (prima un contatore unico sopprimeva un
    oversold che seguiva un overbought, contraddicendo il testo dell'interfaccia);
  * il VIX si misura in log-rendimento (la percentuale semplice ha media dominata
    da pochi spike).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .stats import (baseline_forward_returns, benjamini_hochberg, block_bootstrap_mean,
                    block_len_for, subset_stats)

# Bucket di regime per la stratificazione. Basati sul PERCENTILE rolling, non sul
# livello assoluto: "VIX 15-20" nel 2008 e nel 2017 non sono lo stesso regime, il
# percentile si'.
RANK_BINS = [0.0, 0.20, 0.50, 0.80, 1.01]
RANK_LABELS = ["VIX molto basso (p<20)", "VIX basso (p20-50)",
               "VIX alto (p50-80)", "VIX estremo (p>80)"]


# ============================================================
# RILEVAMENTO EVENTI
# ============================================================
def detect_zscore_events(z: pd.Series, lower: float | None = None,
                         upper: float | None = None,
                         cooldown: int | None = None) -> pd.DataFrame:
    """
    Estremi del log-zScore con anti-clustering PER TIPO di segnale.

    Viene registrato solo il primo crossing di ogni episodio: senza cooldown le
    statistiche si gonfiano contando piu' volte lo stesso evento di volatilita'.
    """
    lower = config.VVIX_Z_LOWER if lower is None else lower
    upper = config.VVIX_Z_UPPER if upper is None else upper
    cooldown = config.VVIX_COOLDOWN if cooldown is None else cooldown

    valid = pd.Series(z).dropna()
    last_pos = {"overbought": -(cooldown + 1), "oversold": -(cooldown + 1)}
    records = []

    for i, (date, val) in enumerate(valid.items()):
        kind = "overbought" if val >= upper else ("oversold" if val <= lower else None)
        if kind is None:
            continue
        if (i - last_pos[kind]) < cooldown:
            continue
        records.append({"date": date, "signal": kind, "zscore": float(val)})
        last_pos[kind] = i

    if not records:
        return pd.DataFrame(columns=["signal", "zscore"],
                            index=pd.DatetimeIndex([], name="date"))
    return pd.DataFrame(records).set_index("date")


def detect_threshold_events(s: pd.Series, threshold: float, direction: str = "above",
                            cooldown: int = 20, label: str = "signal") -> pd.DataFrame:
    """Eventi generici: primo attraversamento di una soglia, con cooldown."""
    valid = pd.Series(s).dropna()
    last = -(cooldown + 1)
    records = []
    for i, (date, val) in enumerate(valid.items()):
        hit = (val >= threshold) if direction == "above" else (val <= threshold)
        if not hit or (i - last) < cooldown:
            continue
        records.append({"date": date, "signal": label, "zscore": float(val)})
        last = i
    if not records:
        return pd.DataFrame(columns=["signal", "zscore"],
                            index=pd.DatetimeIndex([], name="date"))
    return pd.DataFrame(records).set_index("date")


# ============================================================
# RENDIMENTI FORWARD
# ============================================================
def compute_forward_returns(prices: pd.DataFrame, events: pd.DataFrame,
                            horizons: list[int] | None = None,
                            context_cols: tuple[str, ...] = ("vix_rank", "ts_ratio",
                                                             "vvix_z", "stress_score",
                                                             "regime")) -> pd.DataFrame:
    """
    Rendimenti forward di SPX e VIX per ogni evento.

    SPX in percentuale semplice, VIX in LOG-rendimento. Nessun look-ahead: si
    misurano solo date successive al segnale. Gli eventi troppo recenti hanno NaN
    sugli orizzonti non ancora conclusi ed escono dai calcoli via dropna.
    """
    horizons = horizons or config.EVENT_HORIZONS
    cols = ["signal", "zscore", "vix_at_signal", "vix_rank_at_signal"]
    for h in horizons:
        cols += [f"SPX_ret_{h}d", f"VIX_ret_{h}d"]

    if events.empty:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))

    idx = prices.index
    spx = prices["SPX"].to_numpy(dtype=float) if "SPX" in prices.columns else None
    vix = prices["VIX"].to_numpy(dtype=float)
    n = len(idx)

    records, dates = [], []
    for event_date, row in events.iterrows():
        if event_date not in idx:
            continue
        loc = idx.get_loc(event_date)
        if isinstance(loc, slice):
            loc = loc.start
        rec = {
            "signal": row["signal"],
            "zscore": row.get("zscore", np.nan),
            "vix_at_signal": float(vix[loc]),
        }
        for c in context_cols:
            if c in prices.columns:
                v = prices[c].iloc[loc]
                rec[f"{c}_at_signal" if c != "vix_rank" else "vix_rank_at_signal"] = \
                    (float(v) if pd.notna(v) else np.nan)

        for h in horizons:
            fut = loc + h
            if fut < n:
                rec[f"VIX_ret_{h}d"] = float(np.log(vix[fut] / vix[loc]) * 100.0)
                rec[f"SPX_ret_{h}d"] = (float((spx[fut] - spx[loc]) / spx[loc] * 100.0)
                                        if spx is not None and np.isfinite(spx[loc])
                                        else np.nan)
            else:
                rec[f"VIX_ret_{h}d"] = np.nan
                rec[f"SPX_ret_{h}d"] = np.nan
        records.append(rec)
        dates.append(event_date)

    if not records:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
    out = pd.DataFrame(records, index=pd.DatetimeIndex(dates, name="date"))
    return out


# ============================================================
# STRATIFICAZIONE PER REGIME
# ============================================================
def assign_rank_bucket(rank_series: pd.Series) -> pd.Series:
    return pd.cut(rank_series, bins=RANK_BINS, labels=RANK_LABELS,
                  right=False, ordered=True)


def conditional_table(fwd: pd.DataFrame, signal: str, asset: str,
                      horizons: list[int], baseline: dict,
                      cooldown: int = config.VVIX_COOLDOWN,
                      min_events: int | None = None,
                      fdr_alpha: float = 0.05) -> pd.DataFrame:
    """
    Verdetto per ogni cella (bucket di regime x orizzonte), con la statistica
    ONESTA: media, baseline incondizionato, excess, CI overlap-aware, p-value e
    — cruciale — la colonna FDR, che dice quali celle sopravvivono alla correzione
    per il numero di test effettuati.

    Senza FDR, testando ~40 celle al 5% ci si aspettano 2 falsi positivi: le
    "regole" che sembrano funzionare sono il risultato atteso del rumore.
    """
    min_events = config.MIN_EVENTS if min_events is None else min_events
    sub = fwd[fwd["signal"] == signal].copy()
    base_a = (baseline or {}).get(asset, {})
    rows = []
    if sub.empty or "vix_rank_at_signal" not in sub.columns:
        return pd.DataFrame(rows)

    sub["bucket"] = assign_rank_bucket(sub["vix_rank_at_signal"])

    for bucket in RANK_LABELS:
        seg = sub[sub["bucket"] == bucket]
        for h in horizons:
            col = f"{asset}_ret_{h}d"
            s = seg[col].dropna() if col in seg.columns else pd.Series(dtype=float)
            n = len(s)
            base_h = base_a.get(h, np.nan)
            row = {"Regime": bucket, "Orizzonte": f"{h}d", "N": n,
                   "Baseline %": round(float(base_h), 2) if np.isfinite(base_h) else np.nan}
            if n < min_events:
                row |= {"Media %": round(float(s.mean()), 2) if n else np.nan,
                        "Excess %": np.nan, "Hit %": np.nan, "CI low %": np.nan,
                        "CI high %": np.nan, "p-value": np.nan, "Sig": "N/D", "FDR": ""}
            else:
                mean, lo, hi, p = block_bootstrap_mean(
                    s.to_numpy(), block_len=block_len_for(h, cooldown),
                    ref=float(base_h) if np.isfinite(base_h) else 0.0)
                row |= {
                    "Media %": round(mean, 2),
                    "Excess %": round(mean - float(base_h), 2) if np.isfinite(base_h) else np.nan,
                    "Hit %": round(float((s > 0).mean()) * 100, 1),
                    "CI low %": round(lo, 2) if not np.isnan(lo) else np.nan,
                    "CI high %": round(hi, 2) if not np.isnan(hi) else np.nan,
                    "p-value": round(p, 3) if not np.isnan(p) else np.nan,
                    "Sig": "OK" if (not np.isnan(p) and p < 0.05) else "",
                    "FDR": "",
                }
            rows.append(row)

    tbl = pd.DataFrame(rows)
    if not tbl.empty and "p-value" in tbl.columns:
        keep = benjamini_hochberg(tbl["p-value"].to_numpy(), alpha=fdr_alpha)
        tbl["FDR"] = np.where(keep, "OK", "")
    return tbl


# ============================================================
# ORCHESTRAZIONE
# ============================================================
def build_event_study(df_feat: pd.DataFrame, horizons: list[int] | None = None) -> dict:
    """
    Event study completo sugli estremi del VVIX, con la statistica cablata.

    Nel repository originale il motore di significativita' esisteva ma non era
    collegato alla dashboard: le medie venivano mostrate senza baseline e senza
    p-value. Qui il baseline e' obbligatorio per costruzione.
    """
    horizons = horizons or config.EVENT_HORIZONS
    if "vvix_z" not in df_feat.columns or df_feat["vvix_z"].notna().sum() == 0:
        return {"events": pd.DataFrame(), "forward": pd.DataFrame(),
                "baseline": {}, "tables": {}, "summary": {}}

    events = detect_zscore_events(df_feat["vvix_z"])
    fwd = compute_forward_returns(df_feat, events, horizons)
    baseline = baseline_forward_returns(df_feat, horizons)

    tables = {}
    for signal in ("overbought", "oversold"):
        for asset in ("VIX", "SPX"):
            if asset not in df_feat.columns:
                continue
            tables[f"{signal}|{asset}"] = conditional_table(
                fwd, signal, asset, horizons, baseline)

    summary = {}
    for signal in ("overbought", "oversold"):
        sub = fwd[fwd["signal"] == signal]
        summary[signal] = {
            "n": int(len(sub)),
            "first": (pd.Timestamp(sub.index.min()).date().isoformat()
                      if len(sub) else None),
            "last": (pd.Timestamp(sub.index.max()).date().isoformat()
                     if len(sub) else None),
            "stats_VIX": subset_stats(sub, "VIX", horizons, baseline).reset_index()
                         .to_dict(orient="records") if len(sub) else [],
            "stats_SPX": subset_stats(sub, "SPX", horizons, baseline).reset_index()
                         .to_dict(orient="records") if len(sub) else [],
        }

    return {"events": events, "forward": fwd, "baseline": baseline,
            "tables": tables, "summary": summary}


def current_analogues(df_feat: pd.DataFrame, horizon: int, k_rank: float = 0.10,
                      min_n: int = 30) -> dict:
    """
    "Cosa e' successo le altre volte in condizioni come oggi."

    Si selezionano i giorni storici con percentile VIX simile a quello odierno e
    stesso stato di term structure, e si restituisce la distribuzione empirica del
    ΔVIX forward: mediana e quantili. E' il forecast piu' leggibile che esista —
    nessun modello di mezzo, solo la storia degli analoghi.
    """
    need = {"vix_rank", "ts_ratio", "VIX"}
    if not need <= set(df_feat.columns):
        return {}

    cur_rank = df_feat["vix_rank"].iloc[-1]
    cur_backw = float(df_feat["ts_ratio"].iloc[-1] or 0) >= config.TS_BACKWARDATION
    if pd.isna(cur_rank):
        return {}

    vix = df_feat["VIX"]
    fwd = np.log(vix.shift(-horizon) / vix) * 100.0
    same_ts = (df_feat["ts_ratio"].fillna(0) >= config.TS_BACKWARDATION) == cur_backw
    close_rank = (df_feat["vix_rank"] - cur_rank).abs() <= k_rank
    sel = same_ts & close_rank & fwd.notna()

    # Se il filtro stretto lascia troppo pochi campioni si allarga sul solo rank.
    if int(sel.sum()) < min_n:
        sel = close_rank & fwd.notna()
    s = fwd[sel].dropna()
    if len(s) < min_n:
        return {}

    q = lambda p: round(float(s.quantile(p)), 2)  # noqa: E731
    return {
        "horizon": int(horizon),
        "n": int(len(s)),
        "basis": f"percentile VIX {float(cur_rank):.2f} ±{k_rank:.2f}, "
                 f"{'backwardation' if cur_backw else 'contango'}",
        "p10": q(0.10), "p25": q(0.25), "median": q(0.50),
        "p75": q(0.75), "p90": q(0.90),
        "prob_up": round(float((s > 0).mean()), 4),
        "vix_now": round(float(vix.iloc[-1]), 2),
        # Livelli VIX attesi: il quantile e' in log-rendimento percentuale.
        "levels": {k: round(float(vix.iloc[-1]) * float(np.exp(q(p) / 100.0)), 2)
                   for k, p in (("p10", 0.10), ("p25", 0.25), ("median", 0.50),
                                ("p75", 0.75), ("p90", 0.90))},
    }
