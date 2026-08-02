"""
alerts.py — Notifiche Telegram: complete, leggibili, decidibili senza aprire nulla.

Criterio di progetto: **l'alert deve bastare a decidere.** Chi lo legge sul telefono
deve capire cosa dice il sistema, quanto ci crede, cosa aspettarsi in concreto e
quanto quel giudizio ha funzionato storicamente — senza aprire la dashboard.

Da qui tre vincoli:

  1. LA PRIMA RIGA E' LA NOTIFICA. Telegram mostra in anteprima solo l'inizio: la
     riga uno porta bias, orizzonte e conviction, cioe' la decisione. Tutto il resto
     e' profondita' per quando si apre il messaggio.

  2. COSA E' CAMBIATO VA PRIMA DI COSA C'E'. Un alert nasce da un cambiamento: il
     confronto con lo stato precedente sta in cima, con i valori prima -> dopo.

  3. OGNI NUMERO COL SUO PARAGONE. La probabilita' accanto alla banda neutra,
     l'accuratezza accanto alla baseline, il VRP accanto al suo percentile, i livelli
     attesi accanto al VIX di oggi.

Formato HTML invece di Markdown: il Markdown legacy di Telegram va in errore su
underscore e asterischi che compaiono normalmente in testo generato (nomi di bucket
come `mid|contango`, razionali con apostrofi). L'HTML richiede di scappare solo tre
caratteri ed e' molto piu' robusto. Resta comunque il fallback in testo semplice.

Lo stato persiste in state/state.json (i runner Actions sono effimeri) e serve sia al
debounce sia a costruire il blocco "cosa e' cambiato".
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)

# Etichette e icone: coerenti con la dashboard, cosi' il colpo d'occhio e' lo stesso.
_BIAS = {
    "LONG_VOL": ("🔴", "LONG VOL", "compra convessita'"),
    "SHORT_VOL": ("🟢", "SHORT VOL", "vendi premio"),
    "FLAT": ("⚪", "FLAT", "astensione"),
}
_CONV_ICON = {"ALTA": "★★★", "MEDIA": "★★☆", "BASSA": "★☆☆", "NULLA": "—"}
_REGIME_ICON = {0: "🟦", 1: "🟨", 2: "🟥"}


# ============================================================
# STATO
# ============================================================
def load_state(path: Path = config.STATE_JSON) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_state(state: dict, path: Path = config.STATE_JSON) -> None:
    Path(path).write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")


# ============================================================
# UTILITY DI FORMATTAZIONE
# ============================================================
def _esc(x) -> str:
    """Scappa i tre caratteri che contano per il parser HTML di Telegram."""
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _pct(x, nd: int = 0) -> str:
    return "n/d" if x is None else f"{float(x) * 100:.{nd}f}%"


def _num(x, nd: int = 2, sign: bool = False) -> str:
    if x is None:
        return "n/d"
    return f"{float(x):+.{nd}f}" if sign else f"{float(x):.{nd}f}"


def _arrow(p: float) -> str:
    """Freccia direzionale che tiene conto della banda di astensione."""
    band = config.NEUTRAL_BAND
    if p > 0.5 + band:
        return "↑"
    if p < 0.5 - band:
        return "↓"
    return "→"


def _plural(n: int, singolare: str, plurale: str) -> str:
    return f"{n} {singolare if n == 1 else plurale}"


# ============================================================
# BLOCCHI DEL MESSAGGIO
# ============================================================
def _block_headline(snap: dict) -> list[str]:
    f = snap["forecast"]
    op = f["operational"]
    icon, name, hint = _BIAS.get(op["bias"], ("⚪", op["bias"], ""))
    # Prima riga = anteprima della notifica: deve portare la decisione, non un trattino.
    if op["bias"] == "FLAT":
        head = f"{icon} <b>{name}</b> — {_esc(hint)} · {f['horizon_primary']}gg"
        sub = f"nessun setup — dati al {_esc(snap.get('asof', ''))}"
    else:
        conv = _CONV_ICON.get(op["conviction"], op["conviction"])
        head = (f"{icon} <b>{name}</b> · {f['horizon_primary']}gg · "
                f"{conv} {op['conviction']}")
        sub = f"{_esc(hint)} — dati al {_esc(snap.get('asof', ''))}"
    return [head, f"<i>{sub}</i>"]


def _block_changes(reasons: list[str], prev: dict, snap: dict) -> list[str]:
    if not reasons:
        return []
    f = snap["forecast"]
    lines = ["", "⚡ <b>COSA E' CAMBIATO</b>"]
    lines += [f"• {_esc(r)}" for r in reasons]
    # Contesto quantitativo del cambiamento, non solo l'etichetta.
    if prev.get("p_up_primary") is not None:
        d = f["p_up_primary"] - prev["p_up_primary"]
        lines.append(f"• P(su) {_pct(prev['p_up_primary'])} → "
                     f"{_pct(f['p_up_primary'])} ({d * 100:+.0f} punti)")
    return lines


def _block_regime(snap: dict) -> list[str]:
    r = snap["regime"]
    icon = _REGIME_ICON.get(r.get("regime_idx", 1), "")
    dur = r.get("days_in_regime")
    since = f" — da {_plural(dur, 'sessione', 'sessioni')}" if dur else ""
    lines = ["", f"{icon} <b>REGIME</b>  {_esc(r['regime_label'])}{since}"]
    if r.get("prev_regime_label"):
        lines.append(f"<i>prima: {_esc(r['prev_regime_label'])} per "
                     f"{_plural(r.get('prev_regime_days', 0), 'sessione', 'sessioni')}</i>")
    lines.append(
        f"VIX <b>{_num(r.get('vix'))}</b> (p{_pct(r.get('vix_rank'))[:-1]}) · "
        f"VIX3M {_num(r.get('vix3m'))} · ratio {_num(r.get('ts_ratio'), 3)}")
    vvix_z = r.get("vvix_z")
    lines.append(
        f"VVIX {_num(r.get('vvix'))}"
        + (f" (z {_num(vvix_z, 2, sign=True)})" if vvix_z is not None else "")
        + (f" · range {_num(r.get('vix_range'), 1)}%" if r.get("vix_range") else "")
        + f" · stress {_num(r.get('stress_score'))}")
    return lines


def _block_forecast(snap: dict) -> list[str]:
    f = snap["forecast"]
    ph = f["per_horizon"]
    lines = ["", "🎯 <b>P(VIX SU)</b>"]
    lines.append("   ".join(
        f"<b>{h}gg</b> {_pct(ph[h]['p_up'])} {_arrow(ph[h]['p_up'])}"
        for h in sorted(ph, key=int)))
    lines.append(f"<i>banda di astensione ±{_pct(config.NEUTRAL_BAND)} attorno al 50%</i>")

    # Cosa aspettarsi in livelli, non in probabilita': e' la parte azionabile.
    an = (snap.get("analogues") or {}).get(str(f["horizon_primary"])) or {}
    if an.get("levels"):
        lv = an["levels"]
        lines.append(
            f"Atteso a {f['horizon_primary']}gg: mediana <b>{lv['median']}</b> · "
            f"banda {lv['p25']}–{lv['p75']} (10-90: {lv['p10']}–{lv['p90']})")
        lines.append(f"<i>da {an['n']} giorni storici analoghi — {_esc(an.get('basis', ''))}</i>")
    return lines


def _block_premium(snap: dict) -> list[str]:
    v = snap["vrp"]["latest"]
    if v.get("vrp") is None and v.get("vrp_egarch") is None:
        return []
    lines = ["", "💎 <b>PREMIO (VRP)</b>"]
    rank = v.get("vrp_rank")
    lines.append(
        f"<b>{_num(v.get('vrp'), 2, sign=True)}</b> punti · {_esc(v.get('state', 'n/d'))}"
        + (f" (percentile {_pct(rank)})" if rank is not None else ""))
    eg, har = v.get("vrp_egarch"), v.get("vrp_har")
    if eg is not None or har is not None:
        agree = "concordi ✔" if v.get("agree") else "<b>discordi ⚠</b>"
        lines.append(f"EGARCH {_num(eg, 2, sign=True)} / HAR {_num(har, 2, sign=True)} → {agree}")
    return lines


def _block_why(snap: dict) -> list[str]:
    f = snap["forecast"]
    op = f["operational"]
    r = snap["regime"]
    lines = ["", "🔍 <b>PERCHE'</b>"]

    flags = r.get("flags") or {}
    if flags:
        names = {"term_structure": "term struct", "vix_level": "livello",
                 "vvix": "vol-of-vol", "trend": "trend"}
        lines.append(" · ".join(f"{names.get(k, k)} {_num(v, 2)}"
                                for k, v in flags.items() if v is not None))
    ph = f["per_horizon"].get(str(f["horizon_primary"]), {})
    if ph.get("bucket"):
        lines.append(f"bucket <code>{_esc(ph['bucket'])}</code> · n {ph.get('n_bucket', 0)}")
    if op.get("rationale"):
        lines.append(f"<i>{_esc(op['rationale'])}</i>")

    conv = f.get("conviction") or {}
    if conv:
        lines.append(f"<i>conviction {conv.get('score')} = edge {conv.get('edge')} · "
                     f"accordo {conv.get('agreement')} · stabilita' {conv.get('stability')}</i>")
    return lines


def _block_reliability(snap: dict) -> list[str]:
    rel = snap.get("reliability") or {}
    if not rel:
        return ["", "📊 <i>Validazione OOS non disponibile: esegui validate.py.</i>"]

    sig = "significativo ✔" if rel.get("significant") else "<b>NON significativo ⚠</b>"
    lines = ["", f"📊 <b>AFFIDABILITA'</b> (OOS {rel['horizon']}gg)"]
    lines.append(f"{_pct(rel.get('accuracy'), 1)} contro baseline "
                 f"{_pct(rel.get('baseline'), 1)} — {sig}")
    if rel.get("spread") is not None:
        lines.append(f"Quando dice su: VIX {_num(rel.get('fwd_up'), 1, sign=True)}% · "
                     f"quando dice giu': {_num(rel.get('fwd_down'), 1, sign=True)}% "
                     f"(spread {_num(rel.get('spread'), 1)}%)")
    lines.append(f"<i>n {rel.get('n_oos')} (N efficace {rel.get('effective_n')}) · "
                 f"{_esc(rel.get('oos_start'))} → {_esc(rel.get('oos_end'))}</i>")
    return lines


def _block_caveats(snap: dict) -> list[str]:
    """Le avvertenze che cambiano una decisione. Se non ce ne sono, il blocco sparisce."""
    out = []
    f = snap["forecast"]
    rel = snap.get("reliability") or {}
    fr = snap.get("freshness") or {}
    dq = snap.get("data_quality") or {}

    if not fr.get("is_fresh", True):
        out.append(f"⛔ Dati stantii (lag {_plural(fr.get('lag_days', 0), 'sessione', 'sessioni')}): "
                   "segnali direzionali sospesi.")
    if (f["operational"]["bias"] == "SHORT_VOL"
            and rel.get("vrp_gate_validated") is False):
        out.append("⚠ Il gate VRP non ha evidenza fuori campione: questa e' una gamba "
                   "di <b>carry</b>, non un edge dimostrato. Dimensiona di conseguenza.")
    if not f.get("calibrated"):
        out.append("ℹ Probabilita' non calibrata: la mappa isotonica non ha superato "
                   "il test held-out e non viene applicata.")
    if dq.get("anomalies_total"):
        out.append("ℹ " + _plural(dq["anomalies_total"], "divergenza", "divergenze")
                   + " fra le fonti, risolta su CBOE."
                   if dq["anomalies_total"] == 1 else
                   f"ℹ {dq['anomalies_total']} divergenze fra le fonti, risolte su CBOE.")

    return ["", *out] if out else []


# ============================================================
# COMPOSIZIONE
# ============================================================
def format_status(snap: dict, reasons: list[str] | None = None,
                  prev: dict | None = None, compact: bool = False) -> str:
    """
    Messaggio completo. `compact=True` tiene solo titolo, regime, forecast e avvertenze
    — la forma giusta per il digest quotidiano quando non e' cambiato nulla.
    """
    reasons = reasons or []
    prev = prev or {}

    parts = _block_headline(snap)
    parts += _block_changes(reasons, prev, snap)
    parts += _block_regime(snap)
    parts += _block_forecast(snap)
    if not compact:
        parts += _block_premium(snap)
        parts += _block_why(snap)
        parts += _block_reliability(snap)
    parts += _block_caveats(snap)

    parts += ["", "<i>Ricerca quantitativa. Nessun consiglio finanziario, "
                  "nessuna esecuzione automatica.</i>"]
    return "\n".join(parts)


def diff_and_reasons(snap: dict, prev: dict) -> list[str]:
    """Perche' notificare adesso: si reagisce ai CAMBIAMENTI, non al passare del tempo."""
    if prev.get("regime_idx") is None:
        return ["primo run / stato iniziale"]

    reasons = []
    f = snap["forecast"]
    reg = snap["regime"]

    if reg["regime_idx"] != prev.get("regime_idx"):
        reasons.append(f"REGIME: {config.REGIME_LABELS[prev['regime_idx']]} "
                       f"→ {reg['regime_label']}")
    if f["operational"]["bias"] != prev.get("bias"):
        reasons.append(f"BIAS: {prev.get('bias')} → {f['operational']['bias']}")
    prev_p = prev.get("p_up_primary")
    if prev_p is not None and abs(f["p_up_primary"] - prev_p) >= config.ALERT_PROB_DELTA:
        reasons.append("probabilita' in movimento oltre soglia")
    if f["operational"]["conviction"] == "ALTA" and prev.get("conviction") != "ALTA":
        reasons.append("conviction salita ad ALTA")
    return reasons


# ============================================================
# INVIO
# ============================================================
def _strip_html(text: str) -> str:
    """Versione leggibile senza tag, per il fallback in testo semplice."""
    import re
    return re.sub(r"<[^>]+>", "", text).replace("&amp;", "&") \
        .replace("&lt;", "<").replace("&gt;", ">")


def send_telegram(text: str) -> bool:
    """
    Invio robusto: HTML e, se il parser lo rifiuta, testo semplice. Un sistema di
    alert non puo' perdere una notifica per un carattere fuori posto.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("Credenziali Telegram assenti: alert solo loggato.\n%s",
                    _strip_html(text))
        return False

    url = config.TELEGRAM_API.format(token=token)
    base = {"chat_id": chat_id, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json={**base, "text": text, "parse_mode": "HTML"},
                          timeout=20)
        if r.ok:
            return True
        log.warning("Telegram HTML %s: %s — riprovo in testo semplice.",
                    r.status_code, r.text[:200])
    except Exception as e:  # noqa: BLE001
        log.warning("Telegram HTML fallito (%s): riprovo in testo semplice.", e)

    try:
        r = requests.post(url, json={**base, "text": _strip_html(text)}, timeout=20)
        r.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Invio Telegram fallito anche in testo semplice: %s", e)
        return False


def process_alerts(snap: dict, force_digest: bool = False) -> dict:
    prev = load_state()
    reasons = diff_and_reasons(snap, prev)

    sent = False
    if reasons:
        sent = send_telegram(format_status(snap, reasons, prev, compact=False))
    elif force_digest:
        sent = send_telegram(format_status(snap, [], prev, compact=True))

    f = snap["forecast"]
    save_state({
        "asof": snap.get("asof"),
        "regime_idx": snap["regime"]["regime_idx"],
        "bias": f["operational"]["bias"],
        "conviction": f["operational"]["conviction"],
        "p_up_primary": f["p_up_primary"],
        "last_run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_alert_utc": (datetime.now(timezone.utc).isoformat(timespec="seconds")
                           if sent else prev.get("last_alert_utc")),
    })
    return {"sent": sent, "reasons": reasons}
