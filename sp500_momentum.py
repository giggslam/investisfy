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

Modes (``--mode``)
------------------
snapshot (default) : one row per stock, scores as of the last date (or ``--asof DATE``,
                     which rebuilds the table exactly as it would have looked on that day,
                     using only prices up to and including it).
history            : point-in-time scores for EVERY trading day (no look-ahead), plus the
                     realised forward return after each date. From that long table two
                     hindsight tables are derived:
                       * calibration : for each score bucket, how the next 21/63 trading
                         days actually turned out (mean/median return, hit rate,
                         continuation rate, excess vs the universe average);
                       * information coefficient (IC) : per-day Spearman rank correlation
                         between each momentum measure and the forward return, with mean,
                         t-stat and % of positive days.
both               : snapshot + history.

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
    python sp500_momentum.py --asof 2026-03-31            # snapshot as it looked on 2026-03-31
    python sp500_momentum.py --mode history               # daily scores + calibration + IC tables
    python sp500_momentum.py --mode both --step 5         # history sampled every 5 trading days
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import os
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



def load_or_download(symbols: list[str], start: str, end: str, cache: str | None) -> pd.DataFrame:
    """Use a local CSV cache of adjusted closes when it covers the request; otherwise download."""
    if cache and os.path.exists(cache):
        try:
            cached = pd.read_csv(cache, index_col=0, parse_dates=True)
            covers_start = cached.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=7)
            covers_end = cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=4)
            has_syms = set(symbols) <= set(cached.columns)
            if covers_start and covers_end and has_syms:
                print(f"[prices] using cache {cache}: {cached.shape[1]} symbols, {len(cached)} days")
                return cached.loc[start:end, symbols]
            print(f"[prices] cache {cache} does not cover the request (start={covers_start}, end={covers_end}, symbols={has_syms}); downloading")
        except Exception as e:  # noqa: BLE001
            print(f"[prices] cache unreadable ({e}); downloading", file=sys.stderr)
    print(f"[prices] downloading {len(symbols)} symbols {start} .. {end}")
    prices = download_prices(symbols, start, end)
    if cache:
        prices.to_csv(cache)
        print(f"[prices] cached -> {cache}")
    return prices


# --------------------------------------------------------------------------- #
# 4. Point-in-time history (no look-ahead) and hindsight calibration
# --------------------------------------------------------------------------- #
def rolling_return(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    """% return over the previous `days` trading days, per date and symbol."""
    return (prices / prices.shift(days) - 1.0) * 100.0


def rolling_regression(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Rolling version of momentum_regression(): annualised log-price slope x R^2 over a
    window of days+1 observations ending on each date. Vectorised per column.
    """
    n = days + 1
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    vals = prices.to_numpy(dtype=float)
    out = np.full(vals.shape, np.nan)
    if vals.shape[0] >= n:
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.log(vals)
        for j in range(vals.shape[1]):
            w = np.lib.stride_tricks.sliding_window_view(logp[:, j], n)  # (T-n+1, n)
            yc = w - w.mean(axis=1, keepdims=True)
            slope = (yc @ xc) / sxx
            ss_tot = (yc ** 2).sum(axis=1)
            ss_res = ((yc - np.outer(slope, xc)) ** 2).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
            ann = (np.exp(slope) ** 252 - 1.0) * 100.0
            out[n - 1 :, j] = ann * r2
    return pd.DataFrame(out, index=prices.index, columns=prices.columns)


def score_frame(mom: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional score for every date (each row is one day's universe)."""
    return mom.apply(lambda row: score_momentum(row).astype("float"), axis=1)


def build_history(
    constituents: pd.DataFrame,
    prices: pd.DataFrame,
    method: str,
    fwd_horizons: tuple[int, ...] = (21, 63),
    step: int = 1,
) -> pd.DataFrame:
    """
    Long table: one row per (date, symbol) with point-in-time momentum/scores computed
    from prices up to that date only, plus realised forward returns after that date.
    Windows are strict: a horizon needs days+1 observations, otherwise NaN.
    """
    frames: dict[str, pd.DataFrame] = {}
    for name, days in HORIZONS.items():
        d = days or 252
        ret = rolling_return(prices, d)
        frames[f"{name}_return_pct"] = ret
        frames[f"{name}_momentum"] = ret if method == "return" else rolling_regression(prices, d)
        frames[f"{name}_score"] = score_frame(frames[f"{name}_momentum"])
        print(f"[history] {name:6s} window={d:3d}d  first scored date={frames[f'{name}_score'].dropna(how='all').index.min().date()}")
    frames["composite_score"] = sum(frames[f"{n}_score"] for n in HORIZONS) / len(HORIZONS)
    for h in fwd_horizons:
        frames[f"fwd_{h}d_pct"] = (prices.shift(-h) / prices - 1.0) * 100.0

    idx = prices.index[::-1][::step][::-1]  # sample every `step` days, always keep the last date
    hist = pd.concat(
        {k: fr.loc[idx].rename_axis(index="date", columns="yahoo").stack() for k, fr in frames.items()},
        axis=1,
    ).reset_index()
    hist["close"] = prices.loc[idx].rename_axis(index="date", columns="yahoo").stack().to_numpy()
    hist = hist.dropna(subset=["short_score"])  # nothing scorable before the shortest window fills

    meta = constituents.assign(yahoo=constituents["Symbol"].map(to_yahoo_symbol))[
        ["yahoo", "Symbol", "Security", "Sector"]
    ]
    hist = hist.merge(meta, on="yahoo", how="left").drop(columns="yahoo")
    front = ["date", "Symbol", "Security", "Sector", "close"]
    hist = hist[front + [c for c in hist.columns if c not in front]]
    for c in hist.columns:
        if c.endswith("_pct") or c.endswith("_momentum"):
            hist[c] = hist[c].round(4)
    hist["composite_score"] = hist["composite_score"].round(2)
    return hist.sort_values(["date", "Symbol"]).reset_index(drop=True)


def calibrate(hist: pd.DataFrame, fwd_cols: list[str]) -> pd.DataFrame:
    """
    For each horizon and score bucket: how the following fwd window actually went.
      n                 observations (date x stock)
      mean_fwd_pct      average realised forward return
      median_fwd_pct
      hit_rate_pct      % with forward return > 0
      continuation_pct  % where the forward return had the SAME sign as the score
                        (rising kept rising / falling kept falling)
      excess_vs_all_pct mean forward return minus the universe average on the same dates
                        (removes the market's own direction)
    """
    rows = []
    for fwd in fwd_cols:
        base = hist.dropna(subset=[fwd]).copy()
        base["_excess"] = base[fwd] - base.groupby("date")[fwd].transform("mean")
        for name in [*HORIZONS, "composite"]:
            col = f"{name}_score"
            g = base.dropna(subset=[col]).copy()
            g["_s"] = g[col].round().astype(int)
            for s, grp in g.groupby("_s"):
                f = grp[fwd]
                rows.append(
                    {
                        "forward": fwd,
                        "horizon": name,
                        "score": int(s),
                        "n": int(len(f)),
                        "mean_fwd_pct": round(float(f.mean()), 3),
                        "median_fwd_pct": round(float(f.median()), 3),
                        "hit_rate_pct": round(float((f > 0).mean() * 100), 1),
                        "continuation_pct": round(float((np.sign(f) == np.sign(s)).mean() * 100), 1) if s != 0 else np.nan,
                        "excess_vs_all_pct": round(float(grp["_excess"].mean()), 3),
                        "p10_fwd_pct": round(float(f.quantile(0.10)), 3),
                        "p90_fwd_pct": round(float(f.quantile(0.90)), 3),
                    }
                )
    return pd.DataFrame(rows)


def information_coefficient(hist: pd.DataFrame, fwd_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily Spearman rank correlation between each momentum measure (and the composite
    score) and the realised forward return across the universe. Returns
    (daily series, summary) where summary has mean IC, std, t-stat, % positive days.
    """
    measures = [f"{n}_momentum" for n in HORIZONS] + ["composite_score"]
    daily_parts = []
    summary = []
    for fwd in fwd_cols:
        for m in measures:
            g = hist.dropna(subset=[m, fwd])
            ic = (
                g.groupby("date")
                .apply(lambda d: d[m].rank().corr(d[fwd].rank()) if len(d) >= 30 else np.nan, include_groups=False)
                .dropna()
            )
            if ic.empty:
                continue
            daily_parts.append(pd.DataFrame({"date": ic.index, "measure": m, "forward": fwd, "ic": ic.to_numpy().round(4)}))
            mean, std, n = float(ic.mean()), float(ic.std(ddof=1)), int(len(ic))
            summary.append(
                {
                    "forward": fwd,
                    "measure": m,
                    "days": n,
                    "mean_ic": round(mean, 4),
                    "std_ic": round(std, 4),
                    "t_stat": round(mean / std * math.sqrt(n), 2) if std > 0 else np.nan,
                    "pct_positive_days": round(float((ic > 0).mean() * 100), 1),
                    "first_date": ic.index.min().date().isoformat(),
                    "last_date": ic.index.max().date().isoformat(),
                }
            )
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    return daily, pd.DataFrame(summary)


def print_calibration(cal: pd.DataFrame, fwd: str) -> None:
    sub = cal[cal["forward"] == fwd]
    if sub.empty:
        return
    piv_ex = sub.pivot(index="score", columns="horizon", values="excess_vs_all_pct")
    piv_hit = sub.pivot(index="score", columns="horizon", values="hit_rate_pct")
    piv_n = sub.pivot(index="score", columns="horizon", values="n")
    order = [h for h in [*HORIZONS, "composite"] if h in piv_ex.columns]
    out = pd.concat(
        {"excess_vs_all_%": piv_ex[order], "hit_rate_%": piv_hit[order], "n": piv_n[order]}, axis=1
    ).sort_index(ascending=False)
    print(f"\nCalibration vs {fwd} (rows = point-in-time score, excess = mean fwd return minus universe mean on same dates):")
    print(out.to_string())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2025-01-01", help="first date (YYYY-MM-DD), default 2025-01-01")
    p.add_argument("--end", default=dt.date.today().isoformat(), help="last date (YYYY-MM-DD), default today")
    p.add_argument("--out", default="sp500_momentum.csv", help="output CSV path (history files use its stem)")
    p.add_argument("--method", choices=["return", "regression"], default="return", help="momentum measure")
    p.add_argument("--short", type=int, default=21, help="short horizon in trading days (default 21)")
    p.add_argument("--medium", type=int, default=63, help="medium horizon in trading days (default 63)")
    p.add_argument("--long", type=int, default=252, help="long horizon in trading days (default 252; 0 = whole period; history mode uses 252)")
    p.add_argument("--mode", choices=["snapshot", "history", "both"], default="snapshot", help="what to compute (default snapshot)")
    p.add_argument("--asof", default=None, help="snapshot as of this date (YYYY-MM-DD): only prices up to it are used")
    p.add_argument("--step", type=int, default=1, help="history mode: sample every N trading days (default 1 = daily)")
    p.add_argument("--fwd", default="21,63", help="history mode: forward-return horizons in trading days (default 21,63)")
    p.add_argument("--cache", default="prices_cache.csv", help="CSV cache of downloaded prices ('' to disable)")
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

    prices = load_or_download(symbols, args.start, args.end, args.cache or None)
    print(f"Price window: {prices.index[0].date()} .. {prices.index[-1].date()} ({len(prices)} trading days)")
    stem = os.path.splitext(args.out)[0]

    if args.mode in ("snapshot", "both"):
        px = prices.loc[: args.asof] if args.asof else prices
        if args.asof and px.empty:
            raise SystemExit(f"No prices on or before {args.asof}")
        out = args.out
        if args.asof and args.out == "sp500_momentum.csv":
            out = f"{stem}_asof_{px.index[-1].date().isoformat()}.csv"
        table = build_table(cons, px, args.method)
        table.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[snapshot] as of {px.index[-1].date()}: wrote {len(table)} rows -> {out}")
        for name in HORIZONS:
            sc = table[f"{name}_score"]
            print(f"  {name:6s}: up={int((sc > 0).sum()):3d}  down={int((sc < 0).sum()):3d}  flat={int((sc == 0).sum())}")
        yr_cols = [c for c in table.columns if c.startswith("ret_")]
        cols = ["Symbol", "short_score", "medium_score", "long_score", "composite_score", *yr_cols]
        print("\nTop 10 composite:\n", table[cols].head(10).to_string(index=False))
        print("\nBottom 10 composite:\n", table[cols].tail(10).to_string(index=False))

    if args.mode in ("history", "both"):
        fwd_h = tuple(int(x) for x in args.fwd.split(",") if x.strip())
        fwd_cols = [f"fwd_{h}d_pct" for h in fwd_h]
        hist = build_history(cons, prices, args.method, fwd_h, args.step)
        hist_path = f"{stem}_history.csv"
        hist.to_csv(hist_path, index=False, encoding="utf-8-sig")
        print(f"\n[history] {len(hist)} rows ({hist['date'].nunique()} dates x {hist['Symbol'].nunique()} symbols) -> {hist_path}")

        cal = calibrate(hist, fwd_cols)
        cal_path = f"{stem}_calibration.csv"
        cal.to_csv(cal_path, index=False, encoding="utf-8-sig")
        print(f"[history] calibration table -> {cal_path}")

        ic_daily, ic_sum = information_coefficient(hist, fwd_cols)
        ic_daily.to_csv(f"{stem}_ic_daily.csv", index=False, encoding="utf-8-sig")
        ic_sum.to_csv(f"{stem}_ic.csv", index=False, encoding="utf-8-sig")
        print(f"[history] IC tables -> {stem}_ic.csv, {stem}_ic_daily.csv")

        for fwd in fwd_cols:
            print_calibration(cal, fwd)
        print("\nInformation coefficient (daily Spearman rank corr of measure vs forward return):")
        print(ic_sum.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
