"""
validate.py — Validazione walk-forward OOS su tutti gli orizzonti.

Pesante e quasi statica: si lancia periodicamente (Action settimanale), non ogni notte.
Produce data/validation.json + data/oos_predictions.parquet, che la dashboard mostra
e da cui la pipeline preleva due cose:

    * il CALIBRATORE, ma solo se ha superato il cancello held-out;
    * il verdetto sulla gamba VRP, che il forecast dichiara nel razionale.

    python validate.py                      # tutti gli orizzonti
    python validate.py --horizons 20        # uno solo
    python validate.py --no-ablation        # salta il confronto con/senza stagionalita'
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from src import config, data, walkforward

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("validate")


def _verdict(m: dict, sig: dict) -> str:
    edge = m["edge_vs_majority"]
    if edge >= 0.03 and m["brier_skill_score"] > 0 and sig["significant_5pct"]:
        return ("VERDE — batte le baseline, BSS positivo e edge significativo anche "
                "correggendo per la sovrapposizione dei label.")
    if edge >= 0.03 and m["brier_skill_score"] > 0:
        return ("GIALLO — edge positivo e BSS>0 ma non significativo una volta corretto "
                "per la sovrapposizione: il 95% CI tocca la baseline.")
    if edge > 0:
        return "GIALLO — edge marginale sulle baseline: usare solo dove la conviction e' alta."
    return ("ROSSO — non batte la baseline su questo orizzonte/periodo: "
            "non tradare il segnale grezzo.")


def main(horizons: list[int] | None = None, ablation: bool = True) -> dict:
    horizons = horizons or config.FORECAST_HORIZONS
    log.info("=== Walk-forward OOS — orizzonti %s ===", horizons)

    df_raw, quality = data.build_dataset()
    log.info("Dataset: %d righe  %s -> %s", len(df_raw),
             df_raw.index.min().date(), df_raw.index.max().date())

    res = walkforward.run_all(df_raw, horizons=horizons, ablation=ablation)

    snapshot = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asof": df_raw.index.max().date().isoformat(),
        "data": {"rows": quality["rows"], "first": quality["first"],
                 "last": quality["last"], "anomalies_total": quality["anomalies_total"]},
        "feature_coverage": res["feature_coverage"],
        "vrp_validation": res["vrp_validation"],
        "horizons": {},
    }

    frames = []
    for h, entry in res["horizons"].items():
        preds = entry.pop("_predictions")
        preds = preds.assign(horizon=h)
        frames.append(preds)
        snapshot["horizons"][str(h)] = {
            **{k: v for k, v in entry.items()},
            "verdict": _verdict(entry["metrics"], entry["significance"]),
        }

    pd.concat(frames).to_parquet(config.OOS_PREDICTIONS_PARQUET)
    config.VALIDATION_JSON.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    # --- Riepilogo a schermo ---
    log.info("")
    log.info("%-4s %-9s %-9s %-9s %-18s %-6s %-7s %-8s", "H", "ACC", "BASE", "EDGE",
             "CI 95%", "SIG", "BSS", "CALIB")
    for h in sorted(snapshot["horizons"], key=int):
        e = snapshot["horizons"][h]
        m, s, c = e["metrics"], e["significance"], e["calibrator"]
        log.info("%-4s %-9.4f %-9.4f %+-9.4f [%.3f, %.3f]     %-6s %+-7.4f %-8s",
                 h, m["accuracy"], m["majority_baseline_acc"], m["edge_vs_majority"],
                 s["acc_ci_low"], s["acc_ci_high"], s["significant_5pct"],
                 m["brier_skill_score"], c.get("apply"))
    vv = res["vrp_validation"]
    if vv.get("available"):
        log.info("VRP: %s", vv["verdict"])
        if vv.get("best"):
            b = vv["best"]
            log.info("  miglior percentile %.2f -> spread %+.3f CI [%.3f, %.3f] sig=%s",
                     b["pctl"], b["spread"], b["ci_low"], b["ci_high"], b["significant"])
        if vv.get("legacy_absolute"):
            lg = vv["legacy_absolute"]
            log.info("  soglia assoluta legacy (3.0 punti) -> spread %+.3f CI [%.3f, %.3f] "
                     "sig=%s  [n_ricco=%d n_compresso=%d]",
                     lg["spread"], lg["ci_low"], lg["ci_high"], lg["significant"],
                     lg["rich"]["n"], lg["compressed"]["n"])
    log.info("Salvati %s e %s", config.VALIDATION_JSON.name,
             config.OOS_PREDICTIONS_PARQUET.name)
    log.info("=== done ===")
    return snapshot


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validazione walk-forward OOS")
    ap.add_argument("--horizons", type=int, nargs="*", default=None)
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    main(horizons=args.horizons, ablation=not args.no_ablation)
