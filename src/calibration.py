"""
calibration.py — Calibrazione isotonica CON CANCELLO DI AMMISSIONE.

Il difetto bloccante del sistema precedente: la mappa isotonica veniva fittata
sull'intera storia OOS e applicata live *sempre*, mentre il suo stesso test held-out
mostrava Brier 0.2360 -> 0.2398 e log-loss 0.6656 -> 0.6747, cioe' PEGGIORAVA
entrambe le metriche. L'effetto pratico era grave: la mappa schiacciava il posterior
verso 0.5 (nel caso reale 0.4415 -> 0.4937), spingendolo dentro la banda neutra e
zittendo un modello che fuori campione ha un edge misurato e significativo.

Due difese, entrambe attive:

  1. CANCELLO. Si fitta sul primo 60% delle predizioni OOS (in ordine temporale) e
     si valuta sul restante 40%. La calibrazione viene marcata `apply=True` SOLO se
     il Brier held-out migliora davvero. Altrimenti il calibratore resta nello
     snapshot per ispezione ma il forecast usa la probabilita' grezza.

  2. NODI LIMITATI. La mappa precedente aveva 1800 nodi su 2519 osservazioni: stava
     memorizzando, non calibrando. Qui la PAVA viene applicata su bin di quantili
     (max CALIB_MAX_NODES), che e' il modo standard di regolarizzare l'isotonica.
"""

from __future__ import annotations

import numpy as np

from . import config


# ============================================================
# METRICHE
# ============================================================
def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def logloss(p, y, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ============================================================
# PAVA
# ============================================================
def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Regressione isotonica non decrescente pesata (pool adjacent violators).
    Ritorna i valori fittati ESPANSI, uno per input: i blocchi accorpati vengono
    ri-distribuiti sui rispettivi elementi.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    vals: list[float] = []
    wts: list[float] = []
    cnts: list[int] = []
    for v, wi in zip(y, w):
        cur_v, cur_w, cur_n = float(v), float(wi), 1
        while vals and vals[-1] >= cur_v:
            pv, pw, pn = vals.pop(), wts.pop(), cnts.pop()
            cur_v = (pv * pw + cur_v * cur_w) / (pw + cur_w)
            cur_w += pw
            cur_n += pn
        vals.append(cur_v)
        wts.append(cur_w)
        cnts.append(cur_n)

    out = np.empty(len(y), dtype=float)
    i = 0
    for v, n in zip(vals, cnts):
        out[i:i + n] = v
        i += n
    return out


def isotonic_fit(p, y, min_points: int | None = None,
                 max_nodes: int | None = None) -> dict:
    """
    Mappa isotonica p_grezza -> p_calibrata, regolarizzata su bin di quantili.
    Sotto `min_points` campioni ritorna l'identita'.
    """
    min_points = min_points or config.CALIB_MIN_POINTS
    max_nodes = max_nodes or config.CALIB_MAX_NODES

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if len(p) < min_points:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0], "identity": True,
                "n": int(len(p)), "nodes": 0}

    order = np.argsort(p, kind="mergesort")
    xs, ys = p[order], y[order]

    # Bin di quantili: riduce i nodi e regolarizza.
    n_bins = int(min(max_nodes, max(3, len(xs) // 25)))
    edges = np.unique(np.quantile(xs, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, len(edges) - 2)

    bx, by, bw = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        bx.append(float(xs[sel].mean()))
        by.append(float(ys[sel].mean()))
        bw.append(float(sel.sum()))
    if len(bx) < 2:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0], "identity": True,
                "n": int(len(p)), "nodes": 0}

    fitted = np.maximum.accumulate(_pava(np.array(by), np.array(bw)))
    return {"x": [float(v) for v in bx], "y": [float(v) for v in fitted],
            "identity": False, "n": int(len(p)), "nodes": int(len(bx))}


def isotonic_apply(cal: dict | None, p):
    """Applica la mappa. `None`, identita' o apply=False -> p invariato."""
    if cal is None or cal.get("identity") or cal.get("apply") is False:
        return p
    x = np.asarray(cal["x"], dtype=float)
    y = np.asarray(cal["y"], dtype=float)
    if len(x) < 2:
        return p
    scalar = np.isscalar(p)
    arr = np.atleast_1d(np.asarray(p, dtype=float))
    res = np.interp(np.clip(arr, x[0], x[-1]), x, y)
    return float(res[0]) if scalar else res


# ============================================================
# CANCELLO DI AMMISSIONE
# ============================================================
def _paired_brier_ci(p_before, p_after, y, block: int, n_boot: int = 1000,
                     seed: int = 7) -> tuple[float, float]:
    """
    Intervallo di confidenza del MIGLIORAMENTO di Brier, accoppiato osservazione per
    osservazione e a blocchi.

    Serve perche' un miglioramento puntuale puo' essere rumore: sui dati reali la
    mappa e' arrivata a guadagnare 0.0004 di Brier spostando pero' la probabilita' di
    8 punti, dritta dentro la banda neutra. Il guadagno va dimostrato, non osservato.
    Blocchi di lunghezza pari all'orizzonte perche' i label si sovrappongono.
    """
    d = ((np.asarray(p_before, dtype=float) - np.asarray(y, dtype=float)) ** 2
         - (np.asarray(p_after, dtype=float) - np.asarray(y, dtype=float)) ** 2)
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"))
    block = int(max(1, min(block, n)))
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    max_start = max(n - block, 0)
    ar = np.arange(block)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=nb)
        idx = (starts[:, None] + ar).ravel()
        idx = idx[idx < n]
        boot[i] = d[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (float(lo), float(hi))


def fit_with_gate(p, y, split: float | None = None, block: int = 1) -> dict:
    """
    Fitta il calibratore e decide SE usarlo, in base a un held-out temporale.

    Il cancello richiede DUE condizioni, non una:
        1. il Brier held-out migliora;
        2. il miglioramento e' distinguibile dal rumore — il 95% CI block-bootstrap
           della differenza accoppiata deve escludere lo zero.

    Ritorna la mappa fittata su tutta la storia OOS (per il live) arricchita con:
        apply          bool  -> il forecast la applica solo se True
        gate           dict  -> Brier/log-loss prima e dopo, piu' il CI del guadagno
        gate_reason    str   -> perche' e' stata ammessa o respinta
    """
    split = split or config.CALIB_HELDOUT_SPLIT
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    n = len(p)

    full = isotonic_fit(p, y)

    if n < config.CALIB_MIN_POINTS:
        full |= {"apply": False, "gate": None,
                 "gate_reason": f"campione insufficiente (n={n})"}
        return full

    k = int(n * split)
    cal_train = isotonic_fit(p[:k], y[:k])
    p_test, y_test = p[k:], y[k:]
    if len(p_test) < 30 or cal_train.get("identity"):
        full |= {"apply": False, "gate": None,
                 "gate_reason": "held-out troppo piccolo per decidere"}
        return full

    p_cal = isotonic_apply({**cal_train, "apply": True}, p_test)
    b_before, b_after = brier(p_test, y_test), brier(p_cal, y_test)
    l_before, l_after = logloss(p_test, y_test), logloss(p_cal, y_test)
    improvement = b_before - b_after
    ci_low, ci_high = _paired_brier_ci(p_test, p_cal, y_test, block=block)
    # Di quanto la mappa sposta le probabilita': un guadagno statistico minimo che
    # muove il posterior di parecchi punti percentuali e' un cattivo affare, perche'
    # e' esattamente il movimento che decide se il segnale entra nella banda neutra.
    shift = float(np.mean(np.abs(np.asarray(p_cal) - np.asarray(p_test))))

    significant = bool(np.isfinite(ci_low) and ci_low > 0.0)
    ok = bool(improvement > config.CALIB_MIN_IMPROVEMENT and significant)

    full |= {
        "apply": ok,
        "gate": {
            "split_index": int(k),
            "n_test": int(len(p_test)),
            "block": int(block),
            "brier_before": round(b_before, 5),
            "brier_after": round(b_after, 5),
            "brier_improvement": round(improvement, 5),
            "improvement_ci_low": round(ci_low, 5) if np.isfinite(ci_low) else None,
            "improvement_ci_high": round(ci_high, 5) if np.isfinite(ci_high) else None,
            "significant": significant,
            "logloss_before": round(l_before, 5),
            "logloss_after": round(l_after, 5),
            "mean_abs_shift": round(shift, 4),
        },
        "gate_reason": (
            f"AMMESSA: Brier held-out {b_before:.5f} -> {b_after:.5f}, guadagno "
            f"{improvement:+.5f} con CI 95% [{ci_low:+.5f}, {ci_high:+.5f}] che esclude "
            f"lo zero. Sposta le probabilita' in media di {shift:.1%}."
            if ok else
            f"RESPINTA: Brier held-out {b_before:.5f} -> {b_after:.5f} "
            f"(guadagno {improvement:+.5f}"
            + (f", CI 95% [{ci_low:+.5f}, {ci_high:+.5f}] include lo zero"
               if np.isfinite(ci_low) else "")
            + f"). Sposterebbe le probabilita' di {shift:.1%} senza guadagno "
              "dimostrabile: si usa la probabilita' grezza."
        ),
    }
    return full
