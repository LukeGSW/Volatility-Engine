"""
stats.py — Statistica onesta: UNA sola implementazione, usata da tutti.

I due repository originali avevano due block-bootstrap diversi (uno circolare, uno
no) in due file diversi. Qui ce n'e' uno.

Il principio che governa il modulo: nessuna media senza il suo termine di paragone.
Un rendimento forward medio del +3% dopo un segnale non significa niente finche' non
lo si confronta con il drift INCONDIZIONATO allo stesso orizzonte. La colonna che
conta e' 'Excess', non 'Media'.

Perche' il block-bootstrap e non il t-test: i rendimenti forward a H giorni misurati
su eventi ravvicinati si SOVRAPPONGONO, quindi non sono indipendenti. Con blocchi di
lunghezza pari alla sovrapposizione la dipendenza viene preservata e gli errori
standard restano onesti; la N *effettiva* e' ~N/blocco, non N.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config


# ============================================================
# BASELINE INCONDIZIONATO
# ============================================================
def baseline_forward_returns(prices: pd.DataFrame, horizons: list[int],
                             assets: tuple[str, ...] = ("SPX", "VIX"),
                             log_returns: tuple[str, ...] = ("VIX",)) -> dict:
    """
    Rendimento forward medio su TUTTI i giorni: il drift da battere.

    Il VIX viene misurato in log-rendimento: la sua distribuzione in percentuale
    semplice e' cosi' asimmetrica (+300% contro -60%) che la media e' dominata da
    pochi spike e non descrive il caso tipico.
    """
    out: dict[str, dict] = {}
    for asset in assets:
        if asset not in prices.columns:
            continue
        p = prices[asset].to_numpy(dtype=float)
        out[asset] = {}
        for h in horizons:
            if len(p) > h:
                if asset in log_returns:
                    fr = np.log(p[h:] / p[:-h]) * 100.0
                else:
                    fr = (p[h:] - p[:-h]) / p[:-h] * 100.0
                out[asset][h] = float(np.nanmean(fr))
            else:
                out[asset][h] = np.nan
    return out


# ============================================================
# BLOCK BOOTSTRAP (circolare) — implementazione unica
# ============================================================
def block_bootstrap_mean(values, block_len: int = 1, ref: float = 0.0,
                         n_boot: int | None = None, seed: int | None = None):
    """
    Distribuzione bootstrap a blocchi circolari della media.

    Returns (media, ci_low_95, ci_high_95, p_two_sided_vs_ref).
    Con ref = baseline il p-value testa l'EXCESS, non la media grezza.
    """
    n_boot = n_boot or config.N_BOOT
    seed = config.BOOT_SEED if seed is None else seed

    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    obs = float(v.mean())
    if n == 1:
        return (obs, np.nan, np.nan, np.nan)

    bl = int(max(1, min(block_len, n)))
    n_blocks = int(np.ceil(n / bl))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(bl)[None, None, :]).reshape(n_boot, -1) % n
    means = v[idx][:, :n].mean(axis=1)

    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2.0 * min(float((means <= ref).mean()), float((means >= ref).mean())))
    return (obs, float(lo), float(hi), float(p))


def block_bootstrap_series(x, block: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """Bootstrap a blocchi (non circolare) della media: per serie temporali OOS."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    max_start = max(n - block, 0)
    ar = np.arange(block)
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=nb)
        idx = (starts[:, None] + ar).ravel()
        idx = idx[idx < n]
        out[i] = x[idx].mean()
    return out


def block_len_for(horizon_days: int, cooldown: int) -> int:
    """Blocchi che coprono la sovrapposizione degli eventi: ceil(H / cooldown)."""
    return max(1, math.ceil(horizon_days / max(1, cooldown)))


# ============================================================
# RIGA STATISTICA
# ============================================================
_EMPTY = {
    "N": 0, "Media %": np.nan, "Mediana %": np.nan, "P25 %": np.nan, "P75 %": np.nan,
    "Std %": np.nan, "Hit %": np.nan, "PF": np.nan,
}
_EMPTY_SIG = {
    "Baseline %": np.nan, "Excess %": np.nan, "CI low %": np.nan,
    "CI high %": np.nan, "p-value": np.nan, "Sig": "",
}


def stats_row(series: pd.Series, horizon: int, baseline: float | None,
              cooldown: int) -> dict:
    """Riga statistica per un orizzonte, con significativita' se c'e' il baseline."""
    s = pd.Series(series).dropna()
    n = len(s)
    label = f"{horizon}d"
    if n == 0:
        row = {"Orizzonte": label, **_EMPTY}
        if baseline is not None:
            row |= _EMPTY_SIG
        return row

    pos = s[s > 0].sum()
    neg = abs(s[s < 0].sum())
    pf = (pos / neg) if neg > 0 else np.inf

    row = {
        "Orizzonte": label,
        "N": n,
        "Media %": round(float(s.mean()), 2),
        "Mediana %": round(float(s.median()), 2),
        "P25 %": round(float(s.quantile(0.25)), 2),
        "P75 %": round(float(s.quantile(0.75)), 2),
        "Std %": round(float(s.std(ddof=1)), 2) if n > 1 else np.nan,
        "Hit %": round(float((s > 0).mean()) * 100, 1),
        "PF": round(float(pf), 2) if np.isfinite(pf) else np.inf,
    }

    if baseline is not None and not (isinstance(baseline, float) and np.isnan(baseline)):
        mean, lo, hi, p = block_bootstrap_mean(
            s.to_numpy(), block_len=block_len_for(horizon, cooldown), ref=baseline,
        )
        row |= {
            "Baseline %": round(float(baseline), 2),
            "Excess %": round(float(mean - baseline), 2),
            "CI low %": round(lo, 2) if not np.isnan(lo) else np.nan,
            "CI high %": round(hi, 2) if not np.isnan(hi) else np.nan,
            "p-value": round(p, 3) if not np.isnan(p) else np.nan,
            "Sig": "OK" if (not np.isnan(p) and p < 0.05) else "",
        }
    elif baseline is not None:
        row |= _EMPTY_SIG

    return row


def subset_stats(subset: pd.DataFrame, asset: str, horizons: list[int],
                 baseline: dict | None = None,
                 cooldown: int = config.VVIX_COOLDOWN) -> pd.DataFrame:
    """Statistiche per orizzonte su un sotto-campione gia' filtrato."""
    base_a = (baseline or {}).get(asset, {})
    rows = []
    for h in horizons:
        col = f"{asset}_ret_{h}d"
        s = subset[col] if (col in subset.columns) else pd.Series(dtype=float)
        rows.append(stats_row(s, h, base_a.get(h) if baseline else None, cooldown))
    return pd.DataFrame(rows).set_index("Orizzonte")


# ============================================================
# CORREZIONE PER TEST MULTIPLI
# ============================================================
def benjamini_hochberg(pvals, alpha: float = 0.05) -> np.ndarray:
    """
    Controllo del False Discovery Rate.

    Serve perche' l'event study valuta molte celle contemporaneamente (tipo di
    segnale x bucket di regime x orizzonte x asset). Con 80 test indipendenti al 5%
    ci si aspettano 4 "scoperte" per puro caso: senza correzione, le regole che
    sembrano funzionare sono il risultato atteso del rumore.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        out[order[:kmax + 1]] = True
    return out
