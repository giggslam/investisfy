# investisfy

## S&P 500 momentum scorer (`sp500_momentum.py`)

Fetches the current S&P 500 constituents, downloads daily adjusted close prices
from 2024-01-01 to today (Yahoo Finance via `yfinance`), measures each stock's
realised momentum over three horizons and writes the result to a CSV.

| Horizon | Window                                   |
|---------|------------------------------------------|
| short   | last 21 trading days (~1 month)          |
| medium  | last 63 trading days (~3 months)         |
| long    | last 252 trading days (~1 year)          |

Scores are cross-sectional decile ranks per horizon:

* rising stocks: **+1** (slowest rise) … **+10** (fastest rise)
* falling stocks: **-1** (slowest fall) … **-10** (fastest fall)
* unchanged: 0

```bash
pip install -r requirements.txt
python sp500_momentum.py                       # -> sp500_momentum.csv (2024-01-01 .. today)
python sp500_momentum.py --method regression   # slope x R^2 momentum instead of plain return
python sp500_momentum.py --start 2026-01-01 --long 0 --out sp500_momentum_2026.csv   # 2026 only, long = YTD
```

Output columns include Symbol, Security, Sector, start/end date and price,
full-period return, per-calendar-year returns (ret_2024_pct, ret_2025_pct, ret_2026_pct), period high/low, max drawdown, annualised volatility, and for each
horizon the window length, return %, momentum value and score, plus a
`composite_score` (mean of the three) and a `trend` flag (UP_ALL / DOWN_ALL / MIXED).

Generated files: `sp500_momentum.csv` (2024-01-01 .. today) and `sp500_momentum_2026.csv` (2026 YTD only).

### Point-in-time history and hindsight calibration

```bash
python sp500_momentum.py --asof 2026-04-08      # snapshot exactly as it looked on that day (no look-ahead)
python sp500_momentum.py --mode history         # daily scores for every stock + calibration + IC tables
python sp500_momentum.py --mode both --step 5   # sample the history every 5 trading days
```

`history` mode recomputes the three momentum measures and cross-sectional scores for
**every trading day using only prices up to that day**, then attaches the realised
forward return 21 and 63 trading days later (`--fwd`). Horizons use strict windows
(21/63/252 observations), so the long score starts one year after the first price.

| File | Content |
|------|---------|
| `<stem>_history.csv` | one row per (date, stock): close, short/medium/long return, momentum, score, composite, `fwd_21d_pct`, `fwd_63d_pct` (not committed, ~28 MB; regenerate with the command above) |
| `<stem>_calibration.csv` | per horizon and score bucket: n, mean/median forward return, hit rate (% > 0), continuation rate (% same sign as the score), excess vs the universe mean on the same dates, p10/p90 |
| `<stem>_ic.csv` | information coefficient: mean daily Spearman rank correlation between each measure and the forward return, std, t-stat, % positive days |
| `<stem>_ic_daily.csv` | the daily IC series behind the summary |

Downloaded prices are cached in `prices_cache.csv` (ignored by git); pass `--cache ''` to disable.

## Entry points reverse-engineered from the winners (`sp500_entry_points.py`)

Hindsight event study on the cached prices:

1. A 10% zigzag splits every price path into legs; each trough-to-peak leg gaining
   at least 30% is a **winning run** (the trough is the true, unobservable start).
2. A set of real-time entry signals (moving-average crosses, 20/50-day and 52-week
   breakouts, 10-day thrusts, momentum-score flips, "turn" signals that require the
   thrust to start from a falling state) is evaluated on every day with no look-ahead.
3. For every run: when each signal first fired after the trough (lag), the entry price
   and how much of the run was still ahead. For every firing anywhere: forward 63-day
   return, hit rate, excess vs the universe, how often it sits inside the first 20 days
   of a winning run.

```bash
python sp500_momentum.py            # builds prices_cache.csv
python sp500_entry_points.py        # -> sp500_entry_runs.csv, _signals.csv, _trough_profile.csv, _candidates.csv
python sp500_entry_points.py --reversal 0.08 --min-gain 0.40 --recent 5 --top 3
```

| File | Content |
|------|---------|
| `sp500_entry_runs.csv` | one row per winning run (trough/peak date and price, gain, length, features at the trough, each signal's lag / entry / captured %) |
| `sp500_entry_signals.csv` | signal league table: fires per stock-year, coverage, median lag, median captured %, early-in-run precision vs baseline, forward 63-day excess / hit / big-win / bust rates |
| `sp500_entry_trough_profile.csv` | median features of winners 0, 5, 10 and 20 days after the trough |
| `sp500_entry_candidates.csv` | stocks where the top-ranked signals fired in the last N trading days |
