"""
KQ Volatility Engine — Kriterion Quant.

Fusione di KQ-VVIX-Dashboard (event study sugli estremi del VVIX) e
vix-regime-forecaster (forecast probabilistico del VIX) in un unico sistema.

Moduli PURI: nessuna dipendenza da Streamlit, cosi' gli stessi calcoli girano
identici nella pipeline notturna, nella validazione e nella dashboard.

    config       parametri, single source of truth
    data         doppia fonte CBOE + EODHD con riconciliazione
    features     feature causali (rolling, no look-ahead)
    regime       stress score, etichette e storico degli episodi
    cond_model   estimatore condizionale condiviso live <-> validazione
    seasonality  prior stagionale de-regimizzato
    vrp          EGARCH + HAR su varianza Garman-Klass, soglie a percentile
    forecast     posterior multi-orizzonte, conviction, bias operativo
    eventstudy   motore event study generalizzato con baseline e FDR
    stats        block-bootstrap unico e correzione per test multipli
    calibration  isotonica con cancello di ammissione held-out
    walkforward  validazione out-of-sample multi-orizzonte
    alerts       Telegram con stato persistente e debounce
    charts       grafici Plotly (unico modulo che presume un contesto grafico)
"""

__version__ = "1.0.0"
__author__ = "Kriterion Quant"
