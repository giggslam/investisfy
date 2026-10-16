# investisfy

## S&P 500 momentum scorer (`sp500_momentum.py`)

Fetches the current S&P 500 constituents, downloads daily adjusted close prices
from 2025-01-01 to today (Yahoo Finance via `yfinance`), measures each stock's
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
python sp500_momentum.py                       # -> sp500_momentum.csv (2025-01-01 .. today)
python sp500_momentum.py --method regression   # slope x R^2 momentum instead of plain return
python sp500_momentum.py --start 2026-01-01 --long 0 --out sp500_momentum_2026.csv   # 2026 only, long = YTD
```

Output columns include Symbol, Security, Sector, start/end date and price,
full-period return, per-calendar-year returns (ret_2025_pct, ret_2026_pct), period high/low, max drawdown, annualised volatility, and for each
horizon the window length, return %, momentum value and score, plus a
`composite_score` (mean of the three) and a `trend` flag (UP_ALL / DOWN_ALL / MIXED).

Generated files: `sp500_momentum.csv` (2025-01-01 .. today) and `sp500_momentum_2026.csv` (2026 YTD only).
