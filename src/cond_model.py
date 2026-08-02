"""
cond_model.py — Estimatore condizionale condiviso (parita' live <-> validazione).

Il forecast live e il walk-forward chiamano ENTRAMBI questo modulo per costruire i
bucket, fittare le tabelle di frequenza con shrinkage verso la climatologia e predire.
Centralizzarlo garantisce che il processo *spedito* sia esattamente quello *validato*:
senza questo vincolo la dashboard mostra i numeri di un modello e la validazione
certifica quelli di un altro.

Bucket = livello VIX (percentile rolling: basso/medio/alto) x term structure
(backwardation/contango). Lo shrinkage verso il base rate impone al modello di
BATTERE la climatologia per aggiungere valore.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PSEUDO = 10.0  # pseudo-conteggi dello shrinkage bayesiano


def level_bucket(rank) -> str:
    """Bucket di livello dal percentile VIX rolling."""
    if rank is None:
        return "mid"
    try:
        r = float(rank)
    except (TypeError, ValueError):
        return "mid"
    if np.isnan(r):
        return "mid"
    if r >= config.VIX_HIGH_PCTL:
        return "high"
    if r <= config.VIX_LOW_PCTL:
        return "low"
    return "mid"


def build_bucket_arrays(df: pd.DataFrame):
    """
    Da un DataFrame con 'vix_rank' e 'ts_ratio' costruisce gli array allineati:
        lvl  low/mid/high   tsb  backw/contango   buck  "lvl|tsb"
    """
    rank = df["vix_rank"].to_numpy()
    ts = df["ts_ratio"].to_numpy(dtype=float)
    lvl = np.array([level_bucket(r) for r in rank], dtype=object)
    # Term structure mancante -> stato 'na', MAI 'contango' per default: un NaN
    # trattato come zero finisce sotto la soglia di backwardation e classifica come
    # calmo un periodo che puo' essere di stress. Il bucket 'lvl|na' non compare in
    # addestramento, quindi predict_p_cond ripiega automaticamente sul solo livello.
    tsb = np.where(np.isnan(ts), "na",
                   np.where(ts >= config.TS_BACKWARDATION, "backw", "contango"))
    buck = np.array([f"{a}|{b}" for a, b in zip(lvl, tsb)], dtype=object)
    return lvl, tsb, buck


def fit_conditional_tables(y: np.ndarray, buck: np.ndarray, lvl: np.ndarray,
                           pseudo: float = PSEUDO):
    """
    Stima P(up | bucket) e P(up | livello) con shrinkage verso il base rate.
    `y` deve contenere solo campioni con label conclusa (nessun forward in coda).
    """
    y = np.asarray(y, dtype=float)
    base = float(np.mean(y)) if len(y) else 0.5

    def _table(keys: np.ndarray) -> dict:
        tbl = {}
        for k in np.unique(keys):
            m = keys == k
            n = int(m.sum())
            hit = float(y[m].mean()) if n else base
            tbl[k] = (pseudo * base + n * hit) / (pseudo + n)
        return tbl

    return _table(buck), _table(lvl), base


def predict_p_cond(cond_tbl: dict, lvl_tbl: dict, base: float, buck_t, lvl_t) -> float:
    """P(up) per lo stato corrente: bucket completo -> fallback livello -> base rate."""
    return float(cond_tbl.get(buck_t, lvl_tbl.get(lvl_t, base)))


def bucket_counts(buck: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Numerosita' per bucket: serve a dichiarare su quanti campioni poggia la stima."""
    b = buck if valid is None else buck[valid]
    keys, counts = np.unique(b, return_counts=True)
    return {str(k): int(c) for k, c in zip(keys, counts)}
