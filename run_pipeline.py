"""
run_pipeline.py — Pipeline notturna: calcola tutto, salva gli snapshot, notifica.

    1. Scarica i dati da CBOE + EODHD e li riconcilia
    2. Freshness guard: se i dati sono stantii NON produce segnali direzionali
    3. Feature -> regime -> VRP -> stagionalita' -> forecast -> event study
    4. Salva gli snapshot che la dashboard legge (parquet + json)
    5. Confronta con lo stato precedente e invia l'alert Telegram se serve

La dashboard NON ricalcola: legge. Cosi' i numeri mostrati e quelli notificati
coincidono per costruzione, il caricamento e' istantaneo e l'app non ha bisogno
di nessuna chiave.

    python run_pipeline.py                # run standard
    python run_pipeline.py --no-alerts    # calcola e salva, senza Telegram
    python run_pipeline.py --digest       # invia comunque lo status
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from src import (alerts, config, data, eventstudy, features, forecast, regime,
                 seasonality, vrp as vrp_mod)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")


def _load_validation() -> dict:
    """Snapshot di validazione, se presente. La pipeline ne legge due cose."""
    try:
        return json.loads(config.VALIDATION_JSON.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.info("Validazione non disponibile (%s): niente calibratore, "
                 "gate VRP non certificato.", e)
        return {}


def _reliability(val: dict, horizon: int) -> dict:
    """
    Estratto compatto della validazione OOS, incorporato nello snapshot.

    Serve a rispondere dentro l'alert alla domanda "quanto mi fido di questo numero"
    senza aprire la dashboard, e mantiene l'unica fonte di verita': alert e dashboard
    leggono lo stesso snapshot invece di ricalcolare ciascuno per conto suo.
    """
    e = (val.get("horizons") or {}).get(str(horizon))
    if not e:
        return {}
    m, s, sep = e["metrics"], e["significance"], e["separation"]
    vv = val.get("vrp_validation") or {}
    return {
        "horizon": horizon,
        "accuracy": m["accuracy"],
        "baseline": m["majority_baseline_acc"],
        "edge": m["edge_vs_majority"],
        "significant": s["significant_5pct"],
        "ci_low": s["acc_ci_low"], "ci_high": s["acc_ci_high"],
        "n_oos": m["n_oos"], "effective_n": s["effective_n"],
        "fwd_up": sep.get("fwd_ret_when_up_pct"),
        "fwd_down": sep.get("fwd_ret_when_down_pct"),
        "spread": sep.get("spread_pct"),
        "spread_ex_crisis": sep.get("ex_crisis_spread_pct"),
        "vrp_gate_validated": vv.get("any_significant"),
        "oos_start": e["params"]["oos_start"], "oos_end": e["params"]["oos_end"],
        "asof": val.get("asof"),
    }


def _calibrators(val: dict) -> dict:
    """
    Mappa {orizzonte: calibratore}. Ogni orizzonte usa il PROPRIO: il cancello
    held-out puo' ammetterlo su un orizzonte e respingerlo su un altro, e applicare
    la mappa dell'orizzonte primario a tutti significa calibrare un modello con la
    mappa di un altro. I calibratori respinti vengono passati comunque (con
    apply=False) cosi' la dashboard puo' spiegare perche' non sono attivi.
    """
    out = {}
    for h, entry in (val.get("horizons") or {}).items():
        cal = (entry or {}).get("calibrator")
        if cal:
            out[int(h)] = cal
    return out


def _save_timeseries(df_feat: pd.DataFrame, fc: dict, vrp_out: dict,
                     years: int = 8) -> None:
    """Serie storica con feature e probabilita', per i grafici della dashboard."""
    df = df_feat.copy()
    df = df.join(vrp_out["vol_series"]).join(vrp_out["vrp_series"])
    for h, s in fc["p_series"].items():
        df[f"p_up_{h}"] = s

    keep = ["VIX", "VIX3M", "VIX9D", "VVIX", "SPX", "SKEW", "MOVE", "VXN",
            "ts_ratio", "ts_short", "vix_rank", "vvix_rank", "vvix_z", "vix_range",
            "vix_range_z", "stress_score", "regime", "realized_vol", "exp_vol_ann",
            "vrp", "vrp_proxy", "vrp_proxy_rank"]
    keep += [f"p_up_{h}" for h in fc["p_series"]]
    df = df[[c for c in keep if c in df.columns]]
    df = df[df.index >= df.index.max() - pd.Timedelta(days=365 * years)]
    df.to_parquet(config.TIMESERIES_PARQUET)
    log.info("Salvato %s (%d righe, %d colonne)", config.TIMESERIES_PARQUET.name,
             len(df), df.shape[1])


def _save_seasonality(seas: dict) -> None:
    level = seas["level_curve"].add_prefix("lvl_")
    fwd = seas["fwd_curve"].add_prefix("fwd_")
    pd.concat([level, fwd], axis=1).to_parquet(config.SEASONALITY_PARQUET)


def main(send_alerts: bool = True, digest: bool = False) -> dict:
    log.info("=== KQ Volatility Engine — pipeline start ===")

    # 1) Dati
    df_raw, quality = data.build_dataset()
    config.DATA_QUALITY_JSON.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2) Freshness
    fresh = data.check_freshness(df_raw)
    log.info("Freshness: %s", fresh)

    # 3) Modelli
    df_feat = regime.classify_regime(features.compute_features(df_raw))
    regime_latest = regime.latest_state(df_feat)
    episodes = regime.regime_episodes(df_feat)
    episodes.to_parquet(config.EPISODES_PARQUET)

    # Durata del regime in corso e ultimo regime DIVERSO: contesto che rende leggibile
    # l'alert ("Calmo da 12 sessioni, prima Transizione") senza aprire la dashboard.
    regime_latest |= regime.current_context(df_feat, episodes)

    vrp_out = vrp_mod.estimate_vrp(df_feat)
    seas = seasonality.build_seasonality(df_raw)

    val = _load_validation()
    calibrators = _calibrators(val)
    vrp_validated = (val.get("vrp_validation", {}) or {}).get("any_significant")

    fc = forecast.build_forecast(df_feat, regime_latest, vrp_out, seas,
                                 calibrators=calibrators, vrp_validated=vrp_validated)
    log.info("Calibrazione per orizzonte: %s", fc.get("calibrated_by_horizon"))

    # 4) Event study (baseline e FDR cablati per costruzione)
    ev = eventstudy.build_event_study(df_feat)
    if not ev["forward"].empty:
        ev["forward"].to_parquet(config.EVENTS_PARQUET)
    analogues = {h: eventstudy.current_analogues(df_feat, h)
                 for h in config.FORECAST_HORIZONS}

    # 5) Snapshot
    snapshot = {
        "asof": df_raw.index.max().date().isoformat(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "freshness": fresh,
        "data_quality": {"rows": quality["rows"], "first": quality["first"],
                         "last": quality["last"],
                         "anomalies_total": quality["anomalies_total"],
                         "series": list(quality["series"])},
        "regime": regime_latest,
        "vrp": {"latest": vrp_out["latest"]},
        "seasonality": {"priors": {str(k): v for k, v in seas["priors"].items()},
                        "scan": {str(k): v for k, v in seas["scan"].items()},
                        "current_doy": int(df_raw.index.max().dayofyear)},
        "forecast": {k: v for k, v in fc.items() if k != "p_series"},
        "analogues": {str(k): v for k, v in analogues.items()},
        "reliability": _reliability(val, config.PRIMARY_HORIZON),
        "event_study": ev["summary"],
        "params": {"horizons": config.FORECAST_HORIZONS,
                   "primary_horizon": config.PRIMARY_HORIZON,
                   "event_horizons": config.EVENT_HORIZONS,
                   "data_start": config.DATA_START,
                   "rv_estimator": config.RV_ESTIMATOR,
                   "seasonal_weight": config.SEASONAL_WEIGHT,
                   "seasonality_lookback_years": config.SEASONALITY_LOOKBACK_YEARS,
                   "scan_windows": config.SEASONALITY_SCAN_WINDOWS,
                   "neutral_band": config.NEUTRAL_BAND},
    }
    config.SNAPSHOT_JSON.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    _save_timeseries(df_feat, fc, vrp_out)
    _save_seasonality(seas)

    log.info("Regime=%s | Bias=%s (%s) | P(up) %s | VRP=%s (%s)",
             regime_latest["regime_label"], fc["operational"]["bias"],
             fc["operational"]["conviction"],
             " ".join(f"{h}gg {v['p_up']:.0%}"
                      for h, v in sorted(fc["per_horizon"].items(), key=lambda x: int(x[0]))),
             vrp_out["latest"]["vrp"], vrp_out["latest"]["state"])

    # 6) Alert
    result = {"sent": False, "reasons": []}
    if not fresh["is_fresh"]:
        log.warning("Dati stantii: nessun alert direzionale.")
        if send_alerts:
            alerts.send_telegram(
                f"*Dati VIX stantii* (ultimo {fresh['last_data_date']}, "
                f"atteso {fresh['expected_session']}). Calcolo sospeso per sicurezza.")
    elif send_alerts:
        result = alerts.process_alerts(snapshot, force_digest=digest)
        log.info("Alert: %s", result)

    log.info("=== pipeline done ===")
    return {"snapshot": snapshot, "alert": result}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="KQ Volatility Engine — pipeline notturna")
    ap.add_argument("--no-alerts", action="store_true", help="non inviare Telegram")
    ap.add_argument("--digest", action="store_true", help="invia comunque lo status")
    args = ap.parse_args()
    main(send_alerts=not args.no_alerts, digest=args.digest)
