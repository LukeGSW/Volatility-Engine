"""
config.py — Single source of truth del KQ Volatility Engine.

Fonde i due file di parametri dei repository originali (vix-regime-forecaster/config.py
e KQ-VVIX-Dashboard/params.py). Regola: se un numero conta, vive QUI. Nessun altro
modulo contiene costanti magiche, cosi' backtest, pipeline live, alert e dashboard non
possono divergere — la causa classica della rottura di parita'.
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# PERCORSI
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
STATE_DIR = ROOT_DIR / "state"
DATA_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

SNAPSHOT_JSON = DATA_DIR / "latest.json"                      # stato corrente
TIMESERIES_PARQUET = DATA_DIR / "timeseries.parquet"          # serie + feature
SEASONALITY_PARQUET = DATA_DIR / "seasonality.parquet"        # curve stagionali
EVENTS_PARQUET = DATA_DIR / "events.parquet"                  # event study VVIX
EPISODES_PARQUET = DATA_DIR / "episodes.parquet"              # storico regimi
DATA_QUALITY_JSON = DATA_DIR / "data_quality.json"            # riconciliazione fonti
VALIDATION_JSON = DATA_DIR / "validation.json"                # walk-forward OOS
OOS_PREDICTIONS_PARQUET = DATA_DIR / "oos_predictions.parquet"
STATE_JSON = STATE_DIR / "state.json"                         # memoria alert

# ============================================================
# SORGENTI DATI — doppia fonte con riconciliazione
# ============================================================
# CBOE: gratuito, autoritativo sui valori ufficiali, porta l'OHLC.
# EODHD: copre SPX, estende il VIX3M al 2007 (pre-CBOE) e serve le serie cross-asset.
# Verifica empirica 2026-08-02: le due fonti coincidono (VVIX 0 divergenze su 5073,
# VIX3M 1 su 4242, VIX 8 su 9241) tranne un cluster su VIX9D a dicembre 2018.
# In caso di conflitto vince CBOE; ogni divergenza viene loggata: un tick sporco da
# +2.61 punti sul VIX (visto il 2026-02-06) basta a simulare un cambio di regime.
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
CBOE_SERIES = ["VIX", "VIX3M", "VIX9D", "VVIX", "SKEW"]

EODHD_BASE = "https://eodhd.com/api"
EODHD_TICKERS = {
    "VIX": "VIX.INDX",
    "VIX3M": "VIX3M.INDX",
    "VIX9D": "VIX9D.INDX",
    "VVIX": "VVIX.INDX",
    "SKEW": "SKEW.INDX",
    "SPX": "GSPC.INDX",      # solo EODHD (CBOE non espone lo S&P cash)
    # Cross-asset: candidate feature per la Fase 2, oggi solo raccolte e mostrate.
    "MOVE": "MOVE.INDX",
    "VXN": "VXN.INDX",
    "OVX": "OVX.INDX",
}
# Serie indispensabili: senza queste la pipeline si ferma. Tutte e tre da CBOE,
# quindi il sistema resta funzionante anche senza chiave EODHD.
REQUIRED_SERIES = ["VIX", "VIX3M", "VVIX"]
# Serie facoltative: se mancano si degrada elegantemente.
# SPX e' opzionale ma FORTEMENTE raccomandato: senza, niente realized vol, niente
# VRP e niente rendimenti forward dell'S&P nell'event study. Regime, posterior
# direzionale e stagionalita' funzionano comunque (dipendono solo dal complesso VIX).
OPTIONAL_SERIES = ["SPX", "VIX9D", "SKEW", "MOVE", "VXN", "OVX"]

def eodhd_key() -> str | None:
    """Chiave EODHD dall'ambiente. Serve solo alla pipeline, mai alla dashboard."""
    return os.environ.get("EODHD_API_KEY") or None

# Tolleranza di riconciliazione: divergenza segnalata se supera ENTRAMBE le soglie.
RECONCILE_TOL_ABS = 0.10     # punti indice
RECONCILE_TOL_REL = 0.005    # 0.5%

# Inizio del dataset: e' il primo dato VIX3M disponibile (EODHD, ex-VXV).
# Verificato autentico: nessun forward-fill, 96.3% di copertura calendario,
# giunzione col segmento CBOE +2.31%. Include la crisi 2008.
DATA_START = "2007-11-13"

# ============================================================
# ORIZZONTI
# ============================================================
# Ogni orizzonte ha il SUO walk-forward: tabelle, embargo e calibrazione separati.
FORECAST_HORIZONS = [5, 10, 20]
PRIMARY_HORIZON = 20

# Orizzonti dell'event study (piu' lunghi: si guarda il comportamento, non si prevede).
EVENT_HORIZONS = [5, 10, 20, 60, 90]

# ============================================================
# FINESTRE DELLE FEATURE
# ============================================================
SHORT_WINDOWS = [3, 5, 10]
MEDIUM_WINDOWS = [20, 50]
RANK_LOOKBACK = 252 * 2          # percentili rolling ~2 anni (no look-ahead)
REALIZED_VOL_WINDOW = 21         # finestra realized vol
VVIX_Z_WINDOW = 90               # finestra log-zScore VVIX (dal repo KQ)
GAP_WINDOW = 21                  # finestra per normalizzare range/gap del VIX

# ============================================================
# SOGLIE DEI FLAG DI REGIME
# ============================================================
TS_BACKWARDATION = 1.00
TS_STEEP_BACKW = 1.05
TS_DEEP_CONTANGO = 0.90

VIX_HIGH_PCTL = 0.80
VIX_LOW_PCTL = 0.20
VVIX_HIGH_PCTL = 0.80

REGIME_LABELS = ["Calmo / Contango", "Transizione / Elevato", "Stress / Backwardation"]
REGIME_COLORS = ["#26A69A", "#FFC107", "#EF5350"]
STRESS_CALM_MAX = 0.40           # sotto = regime 0
STRESS_STRESS_MIN = 0.65         # sopra = regime 2

FLAG_WEIGHTS = {
    "term_structure": 0.35,
    "vix_level": 0.25,
    "vvix": 0.20,
    "trend": 0.20,
}

# ============================================================
# VRP — soglie a PERCENTILE, non assolute
# ============================================================
# Fix del difetto bloccante B2: la soglia assoluta VRP_RICH=3.0 punti catturava 2320
# giorni su 3758 (62% del campione) e non discriminava nulla — premio realizzato
# 83.9% di hit-rate nel bucket "ricco" contro 82.9% nel "compresso", spread OOS -0.45
# con CI che attraversa lo zero. Un percentile rolling e' scale-invariant e seleziona
# per costruzione una frazione controllata di giorni.
VRP_RANK_LOOKBACK = 252 * 2
VRP_RICH_PCTL = 0.80             # sopra = premio ricco
VRP_COMPRESSED_PCTL = 0.20       # sotto = premio compresso
# Soglie assolute conservate SOLO per il confronto storico nella validazione.
VRP_RICH_ABS = 3.0
VRP_COMPRESSED_ABS = 0.0
# Griglia di percentili testata dalla validazione: se nessuno produce spread
# significativo, il gate viene dichiarato senza evidenza e la gamba declassata.
VRP_PCTL_GRID = [0.70, 0.75, 0.80, 0.85, 0.90]

# Stimatore di realized vol per l'HAR. 'gk' = Garman-Klass su OHLC giornaliero:
# 5-8 volte piu' efficiente del close-to-close, copre tutto il campione, non richiede
# dati intraday (che partirebbero dal 2015 e dimezzerebbero la finestra di validazione).
RV_ESTIMATOR = "gk"              # 'gk' | 'yz' | 'cc'

# ============================================================
# STAGIONALITA'
# ============================================================
SEASONALITY_LOOKBACK_YEARS = 10
SEASONALITY_HARMONICS = 4
SEASONALITY_BOOTSTRAP = 2000
SEASONALITY_SCAN_WINDOWS = [5, 10, 20, 30, 50]

# Peso del prior stagionale nel posterior (log-odds). AZZERATO il 2026-08-02 sulla
# base della validazione walk-forward su 4.743 barre reali: il prior peggiorava
# ACCURATEZZA e CALIBRAZIONE a tutti e tre gli orizzonti.
#     H=5   accuratezza -0.09pp   BSS -0.0036
#     H=10  accuratezza -0.20pp   BSS -0.0133
#     H=20  accuratezza -0.64pp   BSS -0.0178
# Senza il prior l'accuratezza a 20 giorni sale da 59.57% a 60.21%.
# La stagionalita' resta CALCOLATA e mostrata come contesto: quello che non fa piu'
# e' spostare la probabilita'.
SEASONAL_WEIGHT = 0.0

# Peso alternativo usato SOLO dall'ablazione della validazione. Serve a tenere la
# decisione sotto osservazione: a ogni run il report dice cosa succederebbe
# riaccendendo il prior, invece di trattare l'azzeramento come definitivo.
SEASONAL_WEIGHT_ALT = 0.5

# ============================================================
# FORECAST E BIAS OPERATIVO
# ============================================================
NEUTRAL_BAND = 0.07              # zona di astensione attorno a 0.5
CONVICTION_HIGH = 0.75
CONVICTION_MED = 0.50
STABILITY_WINDOW = 5             # giorni su cui misurare la stabilita' del segnale

BIAS_LABELS = {
    "LONG_VOL": "LONG VOLATILITA'",
    "SHORT_VOL": "SHORT VOLATILITA'",
    "FLAT": "FLAT / ASTENSIONE",
}

# ============================================================
# EVENT STUDY VVIX (dal repo KQ-VVIX-Dashboard)
# ============================================================
VVIX_Z_UPPER = 2.5               # soglia overbought
VVIX_Z_LOWER = -2.0              # soglia oversold
VVIX_COOLDOWN = 20               # anti-clustering, ORA per tipo di segnale
MIN_EVENTS = 5                   # sotto questa N le statistiche non si mostrano
N_BOOT = 2000
BOOT_SEED = 12345

# ============================================================
# VALIDAZIONE WALK-FORWARD
# ============================================================
WF_MIN_TRAIN_YEARS = 5
WF_EMBARGO = None                # None -> usa l'orizzonte H
WF_REFIT_EVERY = 21
WF_SELECTIVE_TAUS = [0.0, 0.05, 0.10, 0.15, 0.20]
WF_ROLLING_WINDOW = 63
WF_BOOTSTRAP_BLOCK = None        # None -> usa H
WF_BOOTSTRAP_N = 2000
WF_CRISIS_EXCLUDE = ("2020-02-15", "2020-05-01")

# Calibrazione isotonica: si applica SOLO se dimostra di migliorare su held-out.
# Fix del difetto bloccante B1: nel sistema precedente la mappa veniva applicata live
# mentre il suo stesso test held-out mostrava Brier 0.2360 -> 0.2398 e log-loss
# 0.6656 -> 0.6747 (entrambi peggiori), schiacciando il posterior dentro la banda
# neutra e zittendo il segnale.
CALIB_MIN_POINTS = 100
CALIB_HELDOUT_SPLIT = 0.6
CALIB_MIN_IMPROVEMENT = 0.0      # il Brier held-out deve MIGLIORARE, non pareggiare
CALIB_MAX_NODES = 50             # nodi max della mappa: evita di memorizzare i punti

# ============================================================
# FRESHNESS E ALERT
# ============================================================
MAX_STALE_TRADING_DAYS = 1
MARKET_CALENDAR = "CBOE_Index_Options"
ALERT_PROB_DELTA = 0.10
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
