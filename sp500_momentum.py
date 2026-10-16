#!/usr/bin/env python3
"""
S&P 500 momentum scorer (2026 YTD).

Pipeline
--------
1. Fetch the current S&P 500 constituent list (GitHub dataset, Wikipedia fallback).
2. Download daily adjusted close prices from 2025-01-01 to today (Yahoo Finance via yfinance).
3. Using realised (ex-post) prices, measure each stock's momentum over three horizons:
       short  = last 21 trading days   (~1 month)
       medium = last 63 trading days   (~3 months)
       long   = last 252 trading days  (~1 year; capped at the data available)
   Per-calendar-year returns (ret_2025_pct, ret_2026_pct, ...) and the full-period
   return are also reported.
4. Score each horizon cross-sectionally:
       rising  stocks ->  +1 (slowest rise)  ... +10 (fastest rise)
       falling stocks ->  -1 (slowest fall)  ... -10 (fastest fall)
       flat            ->   0
   Scores are decile ranks inside the rising group and inside the falling group,
   so +10 means "top 10% of all rising stocks for that horizon" and -10 means
   "the 10% of falling stocks that fell fastest".
5. Write everything to a CSV file.

Momentum measures (``--method``)
--------------------------------
return      : simple % return over the window (default; easiest to interpret)
regression  : annualised slope of a linear regression on log(price) multiplied by
              R^2 (Clenow-style "smooth momentum"; rewards steady trends and
              penalises choppy ones)

Usage
-----
    pip install -r requirements.txt
    python sp500_momentum.py                      # 2025-01-01 .. today -> sp500_momentum.csv
    python sp500_momentum.py --method regression  # smoother momentum measure
    python sp500_momentum.py --start 2026-01-01 --long 0 --out sp500_momentum_2026.csv  # 2026 only, long = YTD
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests

CONSTITUENTS_URLS = [
    # Community-maintained mirror of the Wikipedia table (CSV).
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
]
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = {"User-Agent": "Mozilla/5.0 (compatible; sp500-momentum/1.0)"}

HORIZONS = {  # name -> trading days (None = whole period)
    "short": 21,
    "medium": 63,
    "long": 252,
}


# --------------------------------------------------------------------------- #
# 1. Constituents
# --------------------------------------------------------------------------- #
def fetch_constituents() -> pd.DataFrame:
    """Return a DataFrame with columns Symbol, Security, Sector, SubIndustry."""
    last_err: Exception | None = None
    for url in CONSTITUENTS_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df = df.rename(
                columns={
                    "GICS Sector": "Sector",
                    "GICS Sub-Industry": "SubIndustry",
                }
            )
            print(f"[constituents] {len(df)} rows from {url}")
            return df[["Symbol", "Security", "Sector", "SubIndustry"]]
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[constituents] failed {url}: {e}", file=sys.stderr)

    try:
        r = requests.get(WIKIPEDIA_URL, headers=UA, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0].rename(
            columns={"GICS Sector": "Sector", "GICS Sub-Industry": "SubIndustry"}
        )
        print(f"[constituents] {len(df)} rows from Wikipedia")
        return df[["Symbol", "Security", "Sector", "SubIndustry"]]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Could not fetch S&P 500 constituents (last error: {last_err}; wikipedia: {e})"
        ) from e


def to_yahoo_symbol(sym: str) -> str:
    """BRK.B -> BRK-B, BF.B -> BF-B (Yahoo uses '-' for share classes)."""
    return sym.strip().replace(".", "-")


# --------------------------------------------------------------------------- #
# 2. Prices
# --------------------------------------------------------------------------- #
def download_prices(
    symbols: list[str], start: str, end: str, batch: int = 100, retries: int = 3
) -> pd.DataFrame:
    """Daily adjusted close, one column per Yahoo symbol."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise SystemExit("yfinance is required: pip install yfinance") from e

    # yfinance treats `end` as exclusive -> add one day so "today" is included.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i : i + batch]
        for attempt in range(1, retries + 1):
            try:
                data = yf.download(
                    chunk,
                    start=start,
                    end=end_excl,
                    interval="1d",
                    auto_adjust=True,
                    group_by="column",
                    threads=True,
                    progress=False,
                )
                break
            except Exception as e:  # noqa: BLE001
                print(f"[prices] batch {i // batch + 1} attempt {attempt} failed: {e}", file=sys.stderr)
                if attempt == retries:
                    raise
                time.sleep(2 * attempt)

        if data is None or data.empty:
            missing.extend(chunk)
            continue
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        if not isinstance(data.columns, pd.MultiIndex):
            close.columns = chunk
        frames.append(close)
        print(f"[prices] batch {i // batch + 1}: {close.shape[1]} symbols, {close.shape[0]} days")

    if not frames:
        raise SystemExit("No price data downloaded.")
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index().dropna(how="all")

    got = set(prices.columns[prices.notna().any()])
    missing = [s for s in symbols if s not in got]
    if missing:
        print(f"[prices] no data for {len(missing)} symbols: {', '.join(missing)}", file=sys.stderr)
    return prices


# --------------------------------------------------------------------------- #
# 3. Momentum
# --------------------------------------------------------------------------- #
def momentum_return(series: pd.Series) -> float:
    """Simple % return over the window."""
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[0] - 1.0) * 100.0


def momentum_regression(series: pd.Series) -> float:
    """
    Annualised exponential-regression slope x R^2 (in % per year).
    Positive -> rising, negative -> falling; magnitude grows with speed and smoothness.
    """
    s = series.dropna()
    if len(s) < 5 or (s <= 0).any():
        return float("nan")
    y = np.log(s.to_numpy(dtype=float))
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    annualised = (math.exp(slope) ** 252 - 1.0) * 100.0
    return annualised * r2


def window(series: pd.Series, days: int | None) -> pd.Series:
    s = series.dropna()
    if days is None or len(s) <= days:
        return s
    # `days` trading days of movement needs days+1 observations.
    return s.iloc[-(days + 1) :]


def score_momentum(values: pd.Series) -> pd.Series:
    """
    Cross-sectional decile score.
      > 0 : rank among rising stocks   -> 1 (slowest) .. 10 (fastest)
      < 0 : rank among falling stocks  -> -1 (slowest) .. -10 (fastest)
      == 0 or NaN : 0 / NaN
    """
    out = pd.Series(np.nan, index=values.index, dtype="float")
    up = values[values > 0]
    dn = values[values < 0]
    if len(up):
        pct = up.rank(method="average", pct=True)  # (0, 1]
        out[up.index] = np.clip(np.ceil(pct * 10), 1, 10)
    if len(dn):
        pct = (-dn).rank(method="average", pct=True)  # faster fall -> higher pct
        out[dn.index] = -np.clip(np.ceil(pct * 10), 1, 10)
    out[values == 0] = 0
    return out.astype("Int64")


def build_table(
    constituents: pd.DataFrame, prices: pd.DataFrame, method: str
) -> pd.DataFrame:
    mom_fn = momentum_return if method == "return" else momentum_regression
    rows = []
    for _, c in constituents.iterrows():
        ysym = to_yahoo_symbol(c["Symbol"])
        if ysym not in prices.columns:
            continue
        s = prices[ysym].dropna()
        if len(s) < 2:
            continue
        row = {
            "Symbol": c["Symbol"],
            "Security": c["Security"],
            "Sector": c["Sector"],
            "SubIndustry": c["SubIndustry"],
            "start_date": s.index[0].date().isoformat(),
            "start_price": round(float(s.iloc[0]), 4),
            "end_date": s.index[-1].date().isoformat(),
            "end_price": round(float(s.iloc[-1]), 4),
            "trading_days": int(len(s)),
            "full_return_pct": round(momentum_return(s), 2),
            "period_high": round(float(s.max()), 4),
            "period_low": round(float(s.min()), 4),
            "max_drawdown_pct": round(float(((s / s.cummax()) - 1).min() * 100), 2),
        }
        # Calendar-year returns: from the last close of the prior year (or the first
        # available close) to the last close of that year.
        for yr in sorted(set(s.index.year)):
            prev = s[s.index.year < yr]
            base = float(prev.iloc[-1]) if len(prev) else float(s[s.index.year == yr].iloc[0])
            last = float(s[s.index.year == yr].iloc[-1])
            row[f"ret_{yr}_pct"] = round((last / base - 1.0) * 100.0, 2)
        daily = s.pct_change().dropna()
        row["ann_volatility_pct"] = round(float(daily.std() * math.sqrt(252) * 100), 2) if len(daily) > 1 else np.nan
        for name, days in HORIZONS.items():
            w = window(s, days)
            row[f"{name}_days"] = int(len(w) - 1)
            row[f"{name}_return_pct"] = round(momentum_return(w), 2)
            row[f"{name}_momentum"] = round(mom_fn(w), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    for name in HORIZONS:
        df[f"{name}_score"] = score_momentum(df[f"{name}_momentum"])
    df["composite_score"] = (
        df[[f"{n}_score" for n in HORIZONS]].astype("float").mean(axis=1).round(2)
    )
    df["trend"] = np.select(
        [
            (df["short_score"] > 0) & (df["medium_score"] > 0) & (df["long_score"] > 0),
            (df["short_score"] < 0) & (df["medium_score"] < 0) & (df["long_score"] < 0),
        ],
        ["UP_ALL", "DOWN_ALL"],
        default="MIXED",
    )
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2025-01-01", help="first date (YYYY-MM-DD), default 2025-01-01")
    p.add_argument("--end", default=dt.date.today().isoformat(), help="last date (YYYY-MM-DD), default today")
    p.add_argument("--out", default="sp500_momentum.csv", help="output CSV path")
    p.add_argument("--method", choices=["return", "regression"], default="return", help="momentum measure")
    p.add_argument("--short", type=int, default=21, help="short horizon in trading days (default 21)")
    p.add_argument("--medium", type=int, default=63, help="medium horizon in trading days (default 63)")
    p.add_argument("--long", type=int, default=252, help="long horizon in trading days (default 252; 0 = whole period)")
    p.add_argument("--limit", type=int, default=None, help="only process the first N symbols (testing)")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    HORIZONS["short"] = args.short
    HORIZONS["medium"] = args.medium
    HORIZONS["long"] = args.long or None

    cons = fetch_constituents()
    if args.limit:
        cons = cons.head(args.limit)
    symbols = [to_yahoo_symbol(s) for s in cons["Symbol"]]

    print(f"[prices] downloading {len(symbols)} symbols {args.start} .. {args.end}")
    prices = download_prices(symbols, args.start, args.end)

    table = build_table(cons, prices, args.method)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\nWrote {len(table)} rows -> {args.out}")
    print(f"Price window: {prices.index[0].date()} .. {prices.index[-1].date()} ({len(prices)} trading days)")
    for name in HORIZONS:
        sc = table[f"{name}_score"]
        print(f"  {name:6s}: up={int((sc > 0).sum()):3d}  down={int((sc < 0).sum()):3d}  flat={int((sc == 0).sum())}")
    yr_cols = [c for c in table.columns if c.startswith("ret_")]
    cols = ["Symbol", "short_score", "medium_score", "long_score", "composite_score", *yr_cols]
    print("\nTop 10 composite:\n", table[cols].head(10).to_string(index=False))
    print("\nBottom 10 composite:\n", table[cols].tail(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
