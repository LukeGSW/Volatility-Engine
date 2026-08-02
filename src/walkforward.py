"""
walkforward.py — Validazione out-of-sample. L'unico numero che conta.

Le tabelle di frequenza e il prior stagionale sono "fit" sui dati: misurarli
in-sample non dice nulla. Qui vengono ri-stimati SOLO sul passato, si prevede il
futuro e si confronta col realizzato, ripetendo lungo tutta la storia.

Accortezze obbligatorie:
  * PURGING — la label e' la direzione del VIX a H giorni, quindi il campione al
    tempo i "conosce" il futuro fino a i+H. In training si usano solo campioni con
    i+H <= inizio_test - embargo.
  * EMBARGO — gap aggiuntivo (default = H) per spezzare l'autocorrelazione residua.
  * BASELINE ESPLICITE — classe maggioritaria e persistenza. Chi non le batte e' inutile.
  * SIGNIFICATIVITA' OVERLAP-AWARE — i label a H giorni si sovrappongono, la N
    effettiva e' ~N/H. Il block-bootstrap a blocchi H restituisce CI onesti: l'edge
    conta solo se il limite inferiore del 95% CI ESCLUDE la baseline.

Novita' di questa versione:
  * multi-orizzonte (5/10/20), ognuno con embargo e tabelle propri;
  * il calibratore passa da un CANCELLO held-out: viene ammesso solo se migliora
    davvero (prima veniva applicato anche peggiorando Brier e log-loss);
  * la gamba VRP viene testata su una GRIGLIA di percentili invece che sulla
    singola soglia assoluta a 3 punti, che catturava il 62% dei giorni.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import calibration, cond_model, config, features, regime
from .stats import block_bootstrap_series

log = logging.getLogger(__name__)
_PERIOD = 365


# ============================================================
# METRICHE
# ============================================================
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _brier(y, p):
    return float(np.mean((p - y) ** 2))


def _logloss(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y, p):
    """AUC ROC via rank di Mann-Whitney, con gestione dei ties."""
    y = np.asarray(y)
    p = np.asarray(p)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _circ(a, b, period=_PERIOD):
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def _seasonal_table(vix_slice: pd.Series, horizon: int, lookback_years: int,
                    bucket: int = 4, pseudo: float = 10.0) -> dict:
    """Tabella causale doy -> P(up), costruita SOLO sullo slice passato."""
    fwd = np.log(vix_slice.shift(-horizon) / vix_slice)
    mask = vix_slice.index >= vix_slice.index.max() - pd.Timedelta(days=365 * lookback_years)
    doy = vix_slice.index.dayofyear.to_numpy()[mask]
    vals = fwd.to_numpy()[mask]
    ok = ~np.isnan(vals)
    doy, vals = doy[ok], vals[ok]
    if len(vals) == 0:
        return {}
    base = float((vals > 0).mean())
    tbl = {}
    for d in range(1, _PERIOD + 1):
        s = vals[_circ(doy, d) <= bucket]
        n = len(s)
        hit = float((s > 0).mean()) if n else base
        tbl[d] = (pseudo * 0.5 + n * hit) / (pseudo + n)
    return tbl


# ============================================================
# ENGINE
# ============================================================
def run_walkforward(df_feat: pd.DataFrame, horizon: int | None = None,
                    min_train_years: int | None = None,
                    embargo: int | None = None,
                    refit_every: int | None = None,
                    seasonal_weight: float | None = None,
                    use_seasonal: bool = True,
                    lookback_years: int | None = None) -> dict:
    """Walk-forward espandente su un orizzonte. Ritorna predizioni + tutte le metriche."""
    H = horizon or config.PRIMARY_HORIZON
    min_train_years = min_train_years or config.WF_MIN_TRAIN_YEARS
    embargo = H if (embargo is None and config.WF_EMBARGO is None) else \
        (embargo if embargo is not None else config.WF_EMBARGO)
    refit_every = refit_every or config.WF_REFIT_EVERY
    seasonal_weight = config.SEASONAL_WEIGHT if seasonal_weight is None else seasonal_weight
    lookback_years = lookback_years or config.SEASONALITY_LOOKBACK_YEARS
    # Con peso nullo il prior e' matematicamente ininfluente: si evita di ricostruire
    # la tabella stagionale a ogni refit (365 bucket circolari, la parte piu' lenta).
    use_seasonal = bool(use_seasonal and seasonal_weight != 0.0)

    vix = df_feat["VIX"]
    n = len(df_feat)
    y_all = (vix.shift(-H) > vix).to_numpy(dtype=float)
    fwd_ret = (np.log(vix.shift(-H) / vix) * 100.0).to_numpy()
    doy = df_feat.index.dayofyear.to_numpy()
    lvl, tsb, buck = cond_model.build_bucket_arrays(df_feat)

    min_train = int(min_train_years * 252)
    test_start = max(min_train, embargo + H + 50)
    test_end = n - H - 1
    if test_end <= test_start:
        raise RuntimeError(f"Storia insufficiente per il walk-forward a H={H}.")

    cond_tbl = lvl_tbl = None
    base = 0.5
    seasonal_tbl: dict = {}
    last_refit = -10 ** 9
    rows = []

    for t in range(test_start, test_end + 1):
        if (cond_tbl is None) or (t - last_refit >= refit_every):
            last_train = (t - embargo) - H          # ultimo label completo pre-test
            if last_train < 60:
                continue
            tr = np.arange(0, last_train + 1)
            cond_tbl, lvl_tbl, base = cond_model.fit_conditional_tables(
                y_all[tr], buck[tr], lvl[tr])
            if use_seasonal:
                seasonal_tbl = _seasonal_table(vix.iloc[:t + 1], H, lookback_years)
            last_refit = t

        p_cond = cond_model.predict_p_cond(cond_tbl, lvl_tbl, base, buck[t], lvl[t])
        p_prior = seasonal_tbl.get(int(doy[t]), 0.5) if use_seasonal else 0.5
        p_post = float(_sigmoid(_logit(p_cond) + seasonal_weight * _logit(p_prior)))
        rows.append({"date": df_feat.index[t], "p_cond": round(float(p_cond), 4),
                     "p_prior": round(float(p_prior), 4), "p_post": round(p_post, 4),
                     "y": float(y_all[t]), "fwd_ret_pct": round(float(fwd_ret[t]), 3),
                     "bucket": buck[t], "level": lvl[t]})

    preds = pd.DataFrame(rows).set_index("date")
    if preds.empty:
        raise RuntimeError(f"Walk-forward H={H}: nessuna predizione prodotta.")

    block = int(config.WF_BOOTSTRAP_BLOCK) if config.WF_BOOTSTRAP_BLOCK else H
    metrics = _metrics(preds, vix, H)
    calibrator = calibration.fit_with_gate(preds["p_post"].to_numpy(),
                                           preds["y"].to_numpy(), block=block)
    calibrator["horizon"] = H

    return {
        "predictions": preds,
        "metrics": metrics,
        "significance": _significance(preds, metrics["majority_baseline_acc"], block,
                                      config.WF_BOOTSTRAP_N),
        "reliability": _reliability(preds),
        "calibrator": calibrator,
        "selective": _selective(preds),
        "separation": _separation(preds, block, config.WF_BOOTSTRAP_N),
        "params": {"horizon": H, "embargo": int(embargo),
                   "min_train_years": min_train_years, "refit_every": refit_every,
                   "seasonal_weight": seasonal_weight, "use_seasonal": use_seasonal,
                   "lookback_years": lookback_years, "n_oos": int(len(preds)),
                   "oos_start": preds.index.min().date().isoformat(),
                   "oos_end": preds.index.max().date().isoformat()},
    }


# ============================================================
# METRICHE E BASELINE
# ============================================================
def _metrics(preds: pd.DataFrame, vix: pd.Series, H: int) -> dict:
    p = preds["p_post"].to_numpy()
    y = preds["y"].to_numpy()
    pred_dir = (p > 0.5).astype(float)

    base_rate = float(y.mean())
    majority = max(base_rate, 1 - base_rate)
    acc = float((pred_dir == y).mean())

    pers = (vix - vix.shift(H)).reindex(preds.index).to_numpy()
    acc_pers = float(((pers > 0).astype(float) == y).mean())

    brier = _brier(y, p)
    brier_clim = _brier(y, np.full_like(p, base_rate))
    bss = 1 - brier / brier_clim if brier_clim > 0 else float("nan")

    return {"n_oos": int(len(preds)), "base_rate_up": round(base_rate, 4),
            "accuracy": round(acc, 4), "majority_baseline_acc": round(majority, 4),
            "persistence_baseline_acc": round(acc_pers, 4),
            "edge_vs_majority": round(acc - majority, 4),
            "edge_vs_persistence": round(acc - acc_pers, 4),
            "brier": round(brier, 4), "brier_skill_score": round(float(bss), 4),
            "logloss": round(_logloss(y, p), 4), "auc": round(_auc(y, p), 4)}


def _reliability(preds: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    p = preds["p_post"].to_numpy()
    y = preds["y"].to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin_mid": round(float((edges[b] + edges[b + 1]) / 2), 3),
                     "pred_mean": round(float(p[m].mean()), 4),
                     "obs_freq": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return pd.DataFrame(rows)


def _selective(preds: pd.DataFrame, taus=None) -> pd.DataFrame:
    """Astenersi quando |P-0.5| e' piccolo aiuta davvero? Valida la regola FLAT."""
    taus = taus or config.WF_SELECTIVE_TAUS
    p = preds["p_post"].to_numpy()
    y = preds["y"].to_numpy()
    pred_dir = (p > 0.5).astype(float)
    conf = np.abs(p - 0.5)
    rows = []
    for tau in taus:
        m = conf >= tau
        rows.append({"tau": tau, "coverage": round(float(m.mean()), 4),
                     "accuracy": round(float((pred_dir[m] == y[m]).mean()), 4)
                     if m.any() else float("nan"), "n": int(m.sum())})
    return pd.DataFrame(rows)


def _significance(preds: pd.DataFrame, majority: float, block: int, n_boot: int) -> dict:
    correct = ((preds["p_post"] > 0.5).astype(float) == preds["y"]).to_numpy(dtype=float)
    boot = block_bootstrap_series(correct, block, n_boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"block": int(block), "effective_n": int(np.ceil(len(correct) / block)),
            "acc_mean": round(float(boot.mean()), 4),
            "acc_ci_low": round(float(lo), 4), "acc_ci_high": round(float(hi), 4),
            "majority_acc": round(float(majority), 4),
            "p_value_vs_majority": round(float((boot <= majority).mean()), 4),
            "significant_5pct": bool(lo > majority)}


def _spread_bootstrap(fwd, up_mask, block, n_boot, seed=1) -> np.ndarray:
    n = len(fwd)
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
        u = up_mask[idx]
        f = fwd[idx]
        fu, fd = f[u], f[~u]
        out[i] = (fu.mean() if len(fu) else 0.0) - (fd.mean() if len(fd) else 0.0)
    return out


def _separation(preds: pd.DataFrame, block: int, n_boot: int) -> dict:
    """
    Rilevanza economica: ΔVIX forward quando il modello dice su contro giu'.
    Per le opzioni conta la separazione delle distribuzioni, non l'accuratezza media.
    Si riportano media E mediana (robusta alle code) ed escludendo la crisi COVID,
    per verificare che l'edge non stia tutto in pochi spike.
    """
    up_mask = (preds["p_post"] > 0.5).to_numpy()
    fwd = preds["fwd_ret_pct"].to_numpy()
    up, dn = fwd[up_mask], fwd[~up_mask]

    lo_d, hi_d = (pd.Timestamp(config.WF_CRISIS_EXCLUDE[0]),
                  pd.Timestamp(config.WF_CRISIS_EXCLUDE[1]))
    keep = ~((preds.index >= lo_d) & (preds.index <= hi_d))
    p2 = preds[keep]
    m2 = (p2["p_post"] > 0.5).to_numpy()
    f2 = p2["fwd_ret_pct"].to_numpy()
    up2, dn2 = f2[m2], f2[~m2]

    boot = _spread_bootstrap(fwd, up_mask, block, n_boot)
    ci_low, ci_high = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (None, None))

    def _sp(u, d):
        return (float(np.mean(u)) - float(np.mean(d))) if len(u) and len(d) else None

    def _msp(u, d):
        return (float(np.median(u) - np.median(d))) if len(u) and len(d) else None

    r = lambda x, nd=3: round(x, nd) if x is not None else None  # noqa: E731
    return {"fwd_ret_when_up_pct": r(float(up.mean())) if len(up) else None,
            "fwd_ret_when_down_pct": r(float(dn.mean())) if len(dn) else None,
            "spread_pct": r(_sp(up, dn)),
            "median_up_pct": r(float(np.median(up))) if len(up) else None,
            "median_down_pct": r(float(np.median(dn))) if len(dn) else None,
            "median_spread_pct": r(_msp(up, dn)),
            "ex_crisis_spread_pct": r(_sp(up2, dn2)),
            "ex_crisis_median_spread_pct": r(_msp(up2, dn2)),
            "spread_ci_low": r(float(ci_low)) if ci_low is not None else None,
            "spread_ci_high": r(float(ci_high)) if ci_high is not None else None,
            "crisis_window": list(config.WF_CRISIS_EXCLUDE),
            "n_up": int(len(up)), "n_down": int(len(dn))}


# ============================================================
# GAMBA VRP — griglia di percentili
# ============================================================
def validate_vrp(df_feat: pd.DataFrame, horizon: int, n_boot: int | None = None) -> dict:
    """
    La premessa della gamba short-vol regge fuori campione?

    Test causale e model-free (indipendente da EGARCH/HAR, cosi' e' una baseline
    onesta del CONCETTO):
        segnale(t) = percentile rolling del VRP proxy = VIX - realized vol trailing
        premio(t)  = VIX(t) - realized vol(t+1..t+H)     [>0 = implied era ricca]

    Nella versione precedente la soglia era assoluta (3.0 punti) e catturava il 62%
    dei giorni: lo spread risultava -0.45 con CI [-1.95, +1.22] e Spearman 0.05,
    cioe' nessuna discriminazione. Qui si scansiona una griglia di PERCENTILI e si
    riporta il migliore con la sua significativita'. Se nessun percentile produce
    uno spread positivo significativo, il gate viene dichiarato SENZA EVIDENZA e il
    forecast lo segnala nel razionale invece di fingere che funzioni.
    """
    n_boot = n_boot or config.WF_BOOTSTRAP_N
    if "SPX" not in df_feat.columns or df_feat["SPX"].notna().sum() < 300:
        return {"available": False, "reason": "SPX non disponibile"}

    vix = df_feat["VIX"].to_numpy(dtype=float)
    rv_daily = features.daily_variance(df_feat, "SPX")
    ann = np.sqrt(252.0) * 100.0
    rv_trail = (np.sqrt(rv_daily.rolling(config.REALIZED_VOL_WINDOW,
                                         min_periods=10).mean()) * ann).to_numpy()
    rv_fwd = (np.sqrt(rv_daily.rolling(horizon, min_periods=max(3, horizon // 2))
                      .mean().shift(-horizon)) * ann).to_numpy()

    vrp_proxy = vix - rv_trail
    premium = vix - rv_fwd
    ok = np.isfinite(vrp_proxy) & np.isfinite(premium)
    vp, prem = vrp_proxy[ok], premium[ok]
    rank = df_feat["vrp_proxy_rank"].to_numpy(dtype=float)[ok]

    def _bucket(sel):
        s = prem[sel]
        n = int(sel.sum())
        return {"n": n,
                "mean_premium": round(float(s.mean()), 3) if n else None,
                "median_premium": round(float(np.median(s)), 3) if n else None,
                "hit_rate": round(float((s > 0).mean() * 100), 1) if n else None}

    def _test(rich_sel, comp_sel):
        if not rich_sel.any() or not comp_sel.any():
            return None
        spread = float(prem[rich_sel].mean()) - float(prem[comp_sel].mean())
        nb = int(np.ceil(len(prem) / horizon))
        rng = np.random.default_rng(2)
        max_start = max(len(prem) - horizon, 0)
        ar = np.arange(horizon)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, max_start + 1, size=nb)
            idx = (starts[:, None] + ar).ravel()
            idx = idx[idx < len(prem)]
            r_i, c_i, p_i = rich_sel[idx], comp_sel[idx], prem[idx]
            boot[i] = ((p_i[r_i].mean() if r_i.any() else 0.0)
                       - (p_i[c_i].mean() if c_i.any() else 0.0))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"spread": round(spread, 3), "ci_low": round(float(lo), 3),
                "ci_high": round(float(hi), 3), "significant": bool(lo > 0),
                "rich": _bucket(rich_sel), "compressed": _bucket(comp_sel)}

    # --- Griglia di percentili ---
    grid = []
    ok_rank = np.isfinite(rank)
    for q in config.VRP_PCTL_GRID:
        rich_sel = ok_rank & (rank >= q)
        comp_sel = ok_rank & (rank <= (1.0 - q))
        res = _test(rich_sel, comp_sel)
        if res:
            grid.append({"pctl": q, **{k: v for k, v in res.items()
                                       if k not in ("rich", "compressed")},
                         "n_rich": res["rich"]["n"], "n_comp": res["compressed"]["n"]})

    # --- Soglia assoluta legacy, per il confronto ---
    legacy = _test(vp >= config.VRP_RICH_ABS, vp <= config.VRP_COMPRESSED_ABS)

    best = max(grid, key=lambda g: g["spread"]) if grid else None
    rho = (float(spearmanr(vp, prem).statistic)
           if len(vp) > 10 and np.std(vp) > 0 else float("nan"))
    any_sig = any(g["significant"] for g in grid)

    return {"available": True, "horizon": int(horizon), "n": int(ok.sum()),
            "grid": grid, "best": best, "legacy_absolute": legacy,
            "spearman_vrp_premium": round(rho, 3) if np.isfinite(rho) else None,
            "any_significant": bool(any_sig),
            "verdict": ("il gate VRP ha evidenza OOS con soglia a percentile"
                        if any_sig else
                        "NESSUN percentile produce uno spread di premio significativo: "
                        "la gamba short-vol resta carry senza edge dimostrato"),
            "note": "VRP proxy causale = VIX - realized vol trailing (model-free, "
                    "Garman-Klass); premio = VIX - realized vol forward su H."}


# ============================================================
# MULTI-ORIZZONTE
# ============================================================
def run_all(df_raw: pd.DataFrame, horizons: list[int] | None = None,
            ablation: bool = True) -> dict:
    """
    Esegue il walk-forward su tutti gli orizzonti + ablazione stagionale + gamba VRP.
    Ogni orizzonte ha embargo, tabelle e calibratore propri.
    """
    horizons = horizons or config.FORECAST_HORIZONS
    df_feat = regime.classify_regime(features.compute_features(df_raw))

    out = {"horizons": {}, "feature_coverage": features.feature_summary(df_feat)}
    for h in horizons:
        log.info("--- walk-forward H=%d ---", h)
        res = run_walkforward(df_feat, horizon=h)
        entry = {k: v for k, v in res.items() if k != "predictions"}
        entry["reliability"] = res["reliability"].to_dict(orient="records")
        entry["selective"] = res["selective"].to_dict(orient="records")

        if ablation:
            # Ablazione come confronto tra il peso ATTIVO e un peso alternativo.
            # Con SEASONAL_WEIGHT=0 il vecchio confronto "con/senza stagionalita'"
            # sarebbe diventato identico a se stesso: cosi' invece il report continua
            # a rispondere alla domanda "cosa succederebbe riaccendendo il prior?".
            alt_w = config.SEASONAL_WEIGHT_ALT
            res_alt = run_walkforward(df_feat, horizon=h, seasonal_weight=alt_w,
                                      use_seasonal=alt_w > 0)
            m, ma = res["metrics"], res_alt["metrics"]
            entry["ablation"] = {
                "active_weight": config.SEASONAL_WEIGHT,
                "alt_weight": alt_w,
                "active_acc": m["accuracy"], "alt_acc": ma["accuracy"],
                "delta_acc": round(m["accuracy"] - ma["accuracy"], 4),
                "active_bss": m["brier_skill_score"], "alt_bss": ma["brier_skill_score"],
                "delta_bss": round(m["brier_skill_score"] - ma["brier_skill_score"], 4),
            }

        entry["_predictions"] = res["predictions"]
        out["horizons"][h] = entry
        mm, ss = res["metrics"], res["significance"]
        log.info("H=%2d | acc %.3f vs base %.3f (edge %+.3f) | CI [%.3f, %.3f] | "
                 "sig=%s | BSS %+.4f | AUC %.3f | calib=%s",
                 h, mm["accuracy"], mm["majority_baseline_acc"], mm["edge_vs_majority"],
                 ss["acc_ci_low"], ss["acc_ci_high"], ss["significant_5pct"],
                 mm["brier_skill_score"], mm["auc"], res["calibrator"]["apply"])

    out["vrp_validation"] = validate_vrp(df_feat, config.PRIMARY_HORIZON)
    return out
