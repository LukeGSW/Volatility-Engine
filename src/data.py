"""
data.py — Strato dati a DOPPIA FONTE con riconciliazione.

Perche' non basta una fonte sola. La verifica del 2026-08-02 su 22.000 osservazioni
ha mostrato che CBOE ed EODHD servono la stessa serie (VVIX identico su 5073 date,
VIX3M 1 divergenza su 4242, VIX 8 su 9241) ma che EODHD contiene tick sporchi isolati
— il peggiore, +2.61 punti sul VIX il 2026-02-06. Un errore di quella taglia sposta
vix_rank, vix_ma_ratio e stress_score abbastanza da generare un falso cambio di regime
e un alert sbagliato. Nessuno dei due sistemi originali se ne sarebbe accorto.

Strategia:
    1. Si scarica da entrambe le fonti.
    2. Sul close si confrontano data per data: dove divergono oltre tolleranza si
       registra un'anomalia e VINCE CBOE (autoritativo sui valori ufficiali).
    3. EODHD riempie cio' che CBOE non ha: lo S&P 500 cash, il VIX3M pre-2009
       (segmento ex-VXV, verificato autentico) e le serie cross-asset.
    4. Si legge l'OHLC completo, non solo il close: il range giornaliero del VIX ha
       mediana 7.0% del livello ed e' informazione che il close butta via.

Modulo PURO: nessuna dipendenza da Streamlit. Lo usano pipeline e validazione.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (KriterionQuant/volatility-engine)"}
_TIMEOUT = 60
_OHLC = ["open", "high", "low", "close"]


# ============================================================
# HTTP
# ============================================================
def _http_get(url: str) -> str | None:
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 — si degrada, non si crasha
        log.warning("GET fallito (%s): %s", type(e).__name__, str(e)[:120])
        return None


# ============================================================
# FONTE 1 — CBOE (CSV pubblici, con OHLC)
# ============================================================
def fetch_cboe(name: str) -> pd.DataFrame:
    """
    Serie CBOE come DataFrame con colonne open/high/low/close (le mancanti = close).

    Header variabile: VIX/VIX3M/VIX9D -> DATE,OPEN,HIGH,LOW,CLOSE ; VVIX/SKEW -> DATE,<NOME>.
    """
    body = _http_get(config.CBOE_URL.format(name=name))
    if not body:
        return pd.DataFrame(columns=_OHLC)

    rows = list(csv.reader(io.StringIO(body)))
    if len(rows) < 2:
        return pd.DataFrame(columns=_OHLC)

    header = [c.strip().upper() for c in rows[0]]
    try:
        i_date = next(i for i, c in enumerate(header) if "DATE" in c)
    except StopIteration:
        return pd.DataFrame(columns=_OHLC)

    has_ohlc = all(c in header for c in ("OPEN", "HIGH", "LOW", "CLOSE"))
    if "CLOSE" in header:
        i_close = header.index("CLOSE")
    elif name.upper() in header:
        i_close = header.index(name.upper())
    else:
        i_close = len(header) - 1
    idx = {"close": i_close}
    if has_ohlc:
        idx |= {"open": header.index("OPEN"), "high": header.index("HIGH"),
                "low": header.index("LOW")}

    records = []
    for r in rows[1:]:
        if len(r) <= max(idx.values()):
            continue
        try:
            d = datetime.strptime(r[i_date].strip(), "%m/%d/%Y").date()
        except ValueError:
            continue
        rec = {"date": pd.Timestamp(d)}
        ok = True
        for k, i in idx.items():
            try:
                rec[k] = float(r[i])
            except (ValueError, IndexError):
                ok = False
                break
        if ok:
            records.append(rec)

    if not records:
        return pd.DataFrame(columns=_OHLC)

    df = pd.DataFrame(records).set_index("date").sort_index()
    for c in _OHLC:
        if c not in df.columns:
            df[c] = df["close"]
    log.info("CBOE %-6s: %5d barre  %s -> %s", name, len(df),
             df.index[0].date(), df.index[-1].date())
    return df[_OHLC]


# ============================================================
# FONTE 2 — EODHD
# ============================================================
def fetch_eodhd(name: str, api_key: str) -> pd.DataFrame:
    """Serie EODHD come DataFrame open/high/low/close."""
    ticker = config.EODHD_TICKERS.get(name)
    if not ticker:
        return pd.DataFrame(columns=_OHLC)

    params = urllib.parse.urlencode({"api_token": api_key, "fmt": "json", "period": "d"})
    body = _http_get(f"{config.EODHD_BASE}/eod/{ticker}?{params}")
    if not body:
        return pd.DataFrame(columns=_OHLC)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return pd.DataFrame(columns=_OHLC)
    if not isinstance(data, list) or not data:
        return pd.DataFrame(columns=_OHLC)

    df = pd.DataFrame(data)
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(columns=_OHLC)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for c in _OHLC:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    # Niente riempimento: una barra senza close e' una barra che non esiste. Un
    # forward-fill qui inventerebbe sessioni e falserebbe i rendimenti forward.
    df = df.dropna(subset=["close"])
    if df.empty:
        log.warning("EODHD %s: nessuna barra valida dopo il parsing.", name)
        return pd.DataFrame(columns=_OHLC)

    # OHLC assente o degenere (high == low, tipico delle serie storiche piu' vecchie)
    # -> si ripiega sul close, cosi' i consumatori a valle vedono un range nullo
    # invece di un valore inventato.
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])

    log.info("EODHD %-6s: %5d barre  %s -> %s", name, len(df),
             df.index[0].date(), df.index[-1].date())
    return df[_OHLC]


# ============================================================
# RICONCILIAZIONE
# ============================================================
def reconcile(name: str, cboe: pd.DataFrame, eod: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fonde le due fonti. CBOE vince ovunque esista; EODHD estende e riempie.
    Ritorna (DataFrame OHLC, lista anomalie).
    """
    if cboe.empty and eod.empty:
        return pd.DataFrame(columns=_OHLC), []
    if cboe.empty:
        return eod.copy(), []
    if eod.empty:
        return cboe.copy(), []

    common = cboe.index.intersection(eod.index)
    anomalies: list[dict] = []
    if len(common):
        c = cboe.loc[common, "close"]
        e = eod.loc[common, "close"]
        diff = (c - e).abs()
        rel = diff / c.abs().replace(0.0, np.nan)
        bad = (diff > config.RECONCILE_TOL_ABS) & (rel > config.RECONCILE_TOL_REL)
        for d in common[bad.fillna(False).to_numpy()]:
            anomalies.append({
                "series": name,
                "date": pd.Timestamp(d).date().isoformat(),
                "cboe": round(float(c.loc[d]), 4),
                "eodhd": round(float(e.loc[d]), 4),
                "diff": round(float(e.loc[d] - c.loc[d]), 4),
                "resolved_to": "CBOE",
            })

    # CBOE prioritario riga per riga; EODHD copre le date che CBOE non ha.
    merged = cboe.combine_first(eod).sort_index()
    if anomalies:
        log.warning("%s: %d divergenze oltre tolleranza, risolte su CBOE "
                    "(max |diff| = %.4f)", name, len(anomalies),
                    max(abs(a["diff"]) for a in anomalies))
    return merged[_OHLC], anomalies


# ============================================================
# DATASET COMPLETO
# ============================================================
def build_dataset(api_key: str | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Costruisce il dataset allineato.

    Returns:
        (df, quality) dove df ha DatetimeIndex e, per ogni serie X disponibile,
        le colonne X, X_open, X_high, X_low. 'quality' documenta fonti, anomalie
        e copertura — viene committato e mostrato in dashboard.
    """
    api_key = api_key if api_key is not None else config.eodhd_key()
    if not api_key:
        log.warning("EODHD_API_KEY assente: niente SPX ne' serie cross-asset. "
                    "La pipeline puo' comunque calcolare regime e stagionalita'.")

    wanted = list(dict.fromkeys(config.REQUIRED_SERIES + config.OPTIONAL_SERIES))
    frames: dict[str, pd.DataFrame] = {}
    all_anomalies: list[dict] = []
    sources: dict[str, dict] = {}

    for name in wanted:
        cboe = fetch_cboe(name) if name in config.CBOE_SERIES else pd.DataFrame(columns=_OHLC)
        eod = fetch_eodhd(name, api_key) if (api_key and name in config.EODHD_TICKERS) \
            else pd.DataFrame(columns=_OHLC)
        merged, anomalies = reconcile(name, cboe, eod)
        all_anomalies.extend(anomalies)

        if merged.empty:
            if name in config.REQUIRED_SERIES:
                raise RuntimeError(
                    f"Serie obbligatoria '{name}' non disponibile da nessuna fonte. "
                    "Impossibile procedere."
                )
            log.warning("Serie opzionale '%s' non disponibile: si prosegue senza.", name)
            continue

        frames[name] = merged
        sources[name] = {
            "cboe_bars": int(len(cboe)),
            "eodhd_bars": int(len(eod)),
            "merged_bars": int(len(merged)),
            "cboe_last": cboe.index[-1].date().isoformat() if len(cboe) else None,
            "eodhd_last": eod.index[-1].date().isoformat() if len(eod) else None,
            "first": merged.index[0].date().isoformat(),
            "last": merged.index[-1].date().isoformat(),
            "divergences": sum(1 for a in anomalies),
        }

    # --- Assemblaggio ---
    out = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    for name, df in frames.items():
        out = out.join(df["close"].rename(name), how="outer")
        # OHLC solo dove serve davvero (VIX per range/gap, SPX per Garman-Klass)
        if name in ("VIX", "SPX", "VIX3M", "VIX9D"):
            for c in ("open", "high", "low"):
                out = out.join(df[c].rename(f"{name}_{c}"), how="outer")

    out = out.sort_index()
    out = out[out.index >= pd.Timestamp(config.DATA_START)]

    # Il dataset comincia dove comincia la TERM STRUCTURE, non dove comincia il VIX.
    # Senza VIX3M il rapporto VIX/VIX3M e' indefinito: tenere quelle righe significa
    # farle scivolare nel bucket "contango" per default e classificare la crisi 2008
    # come mercato calmo. Con EODHD il taglio cade al 2007-11-13, con la sola CBOE al
    # 2009-09-18: in entrambi i casi ogni riga ha una term structure vera.
    if "VIX3M" in out.columns and out["VIX3M"].notna().any():
        first_ts = out.index[out["VIX3M"].notna()][0]
        dropped = int((out.index < first_ts).sum())
        if dropped:
            log.info("Scartate %d righe prima del primo VIX3M (%s): term structure "
                     "indefinita.", dropped, pd.Timestamp(first_ts).date())
        out = out[out.index >= first_ts]

    # Ancoraggio al VIX: e' la serie piu' fresca e sempre presente. Le altre
    # possono ritardare di una sessione; ffill limitato, poi si taglia sul VIX.
    for c in out.columns:
        if not c.startswith("VIX") or c == "VIX":
            continue
    fill_cols = [c for c in out.columns if c != "VIX"]
    out[fill_cols] = out[fill_cols].ffill(limit=3)
    out = out.dropna(subset=["VIX"])

    # OHLC degenere (high == low) -> range non misurabile, si marca NaN.
    for name in ("VIX", "VIX3M", "VIX9D", "SPX"):
        hi, lo = f"{name}_high", f"{name}_low"
        if hi in out.columns and lo in out.columns:
            degenerate = out[hi] <= out[lo]
            if degenerate.any():
                out.loc[degenerate, [hi, lo]] = np.nan

    quality = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(out)),
        "first": out.index[0].date().isoformat(),
        "last": out.index[-1].date().isoformat(),
        "series": sources,
        "anomalies_total": len(all_anomalies),
        "anomalies": sorted(all_anomalies, key=lambda a: -abs(a["diff"]))[:50],
        "eodhd_available": bool(api_key),
    }
    log.info("Dataset: %d righe, %s -> %s | serie: %s | divergenze: %d",
             len(out), quality["first"], quality["last"],
             ", ".join(frames), len(all_anomalies))
    return out, quality


# ============================================================
# FRESHNESS GUARD
# ============================================================
def _expected_last_session(now_utc: datetime | None = None, buffer_hours: int = 2) -> pd.Timestamp:
    """Ultima sessione la cui chiusura (+ buffer) e' gia' passata."""
    now_utc = now_utc or datetime.now(timezone.utc)
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar(config.MARKET_CALENDAR)
        sched = cal.schedule(
            start_date=(now_utc - timedelta(days=15)).date().isoformat(),
            end_date=(now_utc + timedelta(days=1)).date().isoformat(),
        )
        closes = sched["market_close"]
        done = closes[closes + pd.Timedelta(hours=buffer_hours) <= pd.Timestamp(now_utc)]
        if len(done):
            return pd.Timestamp(done.index[-1]).normalize()
    except Exception as e:  # noqa: BLE001
        log.info("pandas_market_calendars non disponibile (%s): uso busday.", e)

    d = np.busday_offset(now_utc.date(), 0, roll="backward")
    if now_utc.hour < (21 + buffer_hours) % 24:
        d = np.busday_offset(d, -1, roll="backward")
    return pd.Timestamp(d).normalize()


def check_freshness(df: pd.DataFrame, now_utc: datetime | None = None) -> dict:
    """Verifica che l'ultimo dato corrisponda all'ultima sessione completata."""
    last_data = pd.Timestamp(df.index[-1]).normalize()
    expected = _expected_last_session(now_utc)
    try:
        lag = int(np.busday_count(last_data.date(), expected.date()))
    except Exception:  # noqa: BLE001
        lag = (expected - last_data).days
    lag = max(lag, 0)

    result = {
        "is_fresh": bool(lag <= config.MAX_STALE_TRADING_DAYS),
        "last_data_date": last_data.date().isoformat(),
        "expected_session": expected.date().isoformat(),
        "lag_days": lag,
    }
    if not result["is_fresh"]:
        log.error("DATI STANTII: ultimo=%s atteso=%s lag=%d sessioni",
                  result["last_data_date"], result["expected_session"], lag)
    return result
