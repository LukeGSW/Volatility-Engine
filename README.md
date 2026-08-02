# KQ Volatility Engine — Kriterion Quant

Regime della volatilità dell'S&P 500 e **forecast probabilistico del VIX** a 5, 10 e 20
giorni. Nasce dalla fusione di due progetti: `KQ-VVIX-Dashboard` (event study sugli
estremi del VVIX) e `vix-regime-forecaster` (forecast probabilistico con validazione
walk-forward).

> Strumento di **ricerca**. Non è consulenza finanziaria e non esegue ordini.
> Il target del modello è il **VIX spot**: la traduzione in strumenti è a carico
> dell'operatore.

---

## Cosa fa

Un unico oggetto risponde a tre domande diverse:

1. **In che regime siamo — e in quali siamo stati.** Stress score continuo da quattro
   flag trasparenti (term structure, percentile VIX, vol-of-vol, trend), etichette di
   regime e **storico degli episodi**: quanto sono durati, con che VIX sono cominciati
   e finiti, che massimo hanno toccato.
2. **Dove va il VIX.** Posterior probabilistico su tre orizzonti, ciascuno con la
   propria validazione out-of-sample. La *forma* della curva di probabilità è già
   un'interpretazione.
3. **Cosa aspettarsi in concreto.** Distribuzione empirica del VIX atteso, ricavata dai
   giorni storici in condizioni analoghe a oggi — nessun modello di mezzo.

Il tutto si chiude in un **bias operativo** qualitativo (`LONG_VOL` / `SHORT_VOL` /
`FLAT`) con conviction, e in un event study sugli estremi del VVIX con la statistica
onesta cablata per costruzione.

## Architettura

```
   CBOE (primaria, con OHLC) ─┐
                              ├─► riconciliazione ─► run_pipeline.py (notturno)
   EODHD (SPX, VIX3M pre-2009,│                       feature → regime → VRP →
   cross-asset) ──────────────┘                       stagionalità → forecast
                                                             │
                                    data/*.parquet + latest.json + state/state.json
                                                             │
                                              ┌──────────────┴──────────────┐
                                     app.py (Streamlit)            alert Telegram
                                     legge gli snapshot            stessi numeri
```

La pipeline **calcola e committa**; la dashboard **legge**. I numeri della dashboard e
quelli dell'alert coincidono per costruzione e l'app non ha bisogno di alcun secret.

## Dati

| Serie | CBOE | EODHD | Note |
|---|---|---|---|
| VIX | ✅ OHLC dal 1990 | ✅ | OHLC reale dal 1992 |
| VIX3M | ✅ dal 2009-09 | ✅ **dal 2007-11** | EODHD estende alla crisi 2008 |
| VIX9D | ✅ dal 2011 | ✅ | OHLC reale dal 2014 |
| VVIX | ✅ dal 2006 | ✅ | close only |
| SKEW | ✅ dal 1990 | ✅ | close only |
| SPX | — | ✅ dal 1927 | necessario per realized vol e VRP |
| MOVE · VXN · OVX | — | ✅ | candidate feature, oggi solo raccolte |

**Doppia fonte con riconciliazione.** Le due fonti coincidono (VVIX 0 divergenze su
5.073 date, VIX3M 1 su 4.242, VIX 8 su 9.241) ma EODHD contiene tick sporchi isolati —
il peggiore, **+2.61 punti sul VIX**, basterebbe a simulare un cambio di regime. Ogni
divergenza oltre tolleranza viene registrata e risolta su CBOE.

Il dataset comincia dal **primo VIX3M disponibile**: senza term structure il rapporto
VIX/VIX3M è indefinito, e tenere quelle righe significherebbe classificare la crisi
2008 come mercato calmo.

## Setup

```bash
pip install -r requirements.txt

# La chiave serve SOLO alla pipeline (per SPX, VIX3M pre-2009 e cross-asset).
# Senza, il sistema funziona lo stesso su dati CBOE ma perde VRP e forward SPX.
export EODHD_API_KEY="la-tua-chiave"        # PowerShell: $env:EODHD_API_KEY = "..."

python validate.py                # validazione walk-forward (lenta, ~5 min)
python run_pipeline.py --no-alerts   # calcola gli snapshot
streamlit run app.py              # dashboard
```

### GitHub Actions

| Workflow | Quando | Cosa fa |
|---|---|---|
| `nightly.yml` | cron notturno, mar-sab | pipeline + alert + commit snapshot |
| `validate.yml` | settimanale + manuale | walk-forward su tutti gli orizzonti |

Secrets richiesti: `EODHD_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

### Deploy su Streamlit Cloud

Collega il repo e punta ad `app.py`. **Nessun secret necessario**: la dashboard legge
soltanto gli snapshot committati.

## Metodologia — le scelte che contano

**Tutto rolling, niente look-ahead.** Percentili e z-score usano solo finestre trailing.
Il leakage possibile resta nella label e nella stima delle tabelle, ed è gestito con
purging + embargo.

**Realized vol range-based.** L'HAR-RV regredisce sulla varianza **Garman-Klass**
dall'OHLC giornaliero, 5-8 volte più efficiente del rendimento al quadrato. Niente
dati intraday: partirebbero dal 2015 e dimezzerebbero la finestra di validazione.

**Soglie VRP a percentile.** La soglia assoluta a 3 punti catturava il 62% dei giorni e
non discriminava nulla. Ora sono percentili rolling, e la validazione scansiona una
griglia: se nessuno produce spread significativo, il gate viene dichiarato **senza
evidenza** e il forecast lo scrive nel razionale.

**Calibrazione con cancello.** La mappa isotonica si applica solo se il Brier held-out
migliora **e** il 95% CI block-bootstrap del guadagno esclude lo zero. Sui dati attuali
è respinta su tutti e tre gli orizzonti: guadagno +0.0004 con CI ±0.011, mentre
sposterebbe le probabilità del 6%.

**Conviction, non confluenza.** Tre componenti indipendenti — distanza dal 50%, accordo
tra orizzonti, stabilità del segno nei giorni recenti — al posto di un punteggio che
sommava voti ridondanti.

**Stagionalità disattivata (peso 0).** La validazione su 4.743 barre ha mostrato che il
prior stagionale peggiora *sia* accuratezza *sia* calibrazione a tutti e tre gli
orizzonti (a 20 giorni: accuratezza −0.64pp, BSS −0.0178). Resta calcolata e mostrata
come contesto, ma non sposta più la probabilità. L'ablazione di ogni run confronta il
peso attivo con un peso alternativo, così la decisione resta sotto osservazione invece
di diventare un dogma.

**Correzione per test multipli.** L'event study valuta decine di celle: senza controllo
del False Discovery Rate, le "regole che funzionano" sono il risultato atteso del rumore.

## Alert Telegram

Criterio: **l'alert deve bastare a decidere**, senza aprire la dashboard.

- **La prima riga è la notifica.** Telegram mostra in anteprima solo l'inizio, quindi
  la riga uno porta bias, orizzonte e conviction — la decisione. Il resto è profondità.
- **Cosa è cambiato viene prima di cosa c'è**, con i valori prima → dopo.
- **Ogni numero col suo paragone**: la probabilità accanto alla banda neutra,
  l'accuratezza accanto alla baseline, il VRP accanto al suo percentile, i livelli
  attesi accanto al VIX di oggi.

Blocchi: `REGIME` (con durata dell'episodio e regime precedente) · `P(VIX SU)` (i tre
orizzonti + i livelli VIX attesi dagli analoghi storici) · `PREMIO` (VRP, percentile,
accordo dei due stimatori) · `PERCHÉ` (flag di regime, bucket, razionale, scomposizione
della conviction) · `AFFIDABILITÀ` (accuratezza OOS, baseline, significatività,
separazione economica) · `AVVERTENZE` (solo quelle che cambiano una decisione: dati
stantii, gate VRP non validato, probabilità non calibrata, divergenze fra fonti).

Due modalità: **alert completo** quando qualcosa cambia (~1.400 caratteri) e **digest
compatto** per `--digest` quando non cambia nulla (~700). Formato HTML — il Markdown
legacy di Telegram va in errore sugli underscore e sugli asterischi che compaiono
normalmente nel testo generato — con fallback automatico in testo semplice.

## Validazione walk-forward

`validate.py` ri-stima il modello **solo sul passato** e lo testa sul futuro, lungo
tutta la storia, per ogni orizzonte.

- **Purging + embargo** (= H): i campioni il cui label forward sconfina nel test sono
  esclusi dal training.
- **Baseline esplicite**: classe maggioritaria e persistenza. Il modello deve batterle.
- **Significatività overlap-aware**: i label a H giorni si sovrappongono, la N effettiva
  è ~N/H. L'edge conta solo se il 95% CI block-bootstrap **esclude** la baseline.
- **Separazione economica**: ΔVIX forward quando dice su contro giù, in media e mediana
  ed escludendo il COVID. Per le opzioni conta la separazione, non l'accuratezza.
- **Selective prediction**: astenersi vicino al 50% aiuta davvero? Valida il segnale FLAT.
- **Ablazione stagionale**: il prior aggiunge o toglie?

## Limiti onesti

Il walk-forward elimina il leakage grossolano ma resta condizionato alle scelte di
feature e orizzonte: non garantisce performance futura. Il modello misura il **VIX
spot**, che non è direttamente tradabile — in contango il roll dei futures erode il
risultato. La gamba short-vol, finché la validazione non le dà evidenza, è **carry
esposto**, non edge dimostrato.

---

*Kriterion Quant — [kriterionquant.com](https://kriterionquant.com)*
