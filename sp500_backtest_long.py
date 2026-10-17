#!/usr/bin/env python3
"""
Long-horizon (2000-2024) back-test of the double +10 strategies with a point-in-time
S&P 500 universe.

Universe
--------
Daily index membership comes from the public fja05680/sp500 dataset (membership
snapshots since 1996). On each day only stocks that were index members on that day
are eligible for scoring and for entry. Tickers delisted before today (date-suffixed
in the dataset) have no Yahoo price history and are missing, so the universe is
"members that still trade" rather than the full historical index. That is far less
survivorship-biased than using today's constituents, but not bias-free; the script
also runs the same strategies on today's constituents only so the gap is visible.

Strategies (from sp500_backtest.py / docs/ANALYSIS.md section 6)
-----------------------------------------------------------------
entry  : double_10  = first day a stock has short score 10 AND long score 10 among the
         eligible universe (scores are cross-sectional deciles, see sp500_momentum.py)
A max return : 5 positions, 63-day time stop, 12% fixed stop,   priority = 10-day return
B robust     : 4 positions, 126-day cap, exit when short score < 0 (after 5 days),
               priority = 10-day return

Usage:
    python sp500_backtest_long.py --build     # first run: downloads membership + 26 years of prices (~10 min)
    python sp500_backtest_long.py             # re-run from the caches

Outputs: <stem>_summary.csv, <stem>_yearly.csv, <stem>_equity.csv, <stem>_trades_A.csv, <stem>_trades_B.csv
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

import sp500_backtest as bt
import sp500_momentum as sm


MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def build_data(prices_path: str, membership_path: str, spy_path: str, start: str, end: str) -> None:
    """Download the membership snapshots, build the membership matrix, download prices and SPY."""
    import io
    import requests

    r = requests.get(MEMBERSHIP_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d.date >= pd.Timestamp(start) - pd.Timedelta(days=600)) & (d.date <= pd.Timestamp(end) + pd.Timedelta(days=31))].sort_values("date")
    rows = [{"date": x.date, **{t.strip(): True for t in x.tickers.split(",")}} for x in d.itertuples(index=False)]
    M = pd.DataFrame(rows).set_index("date").fillna(False).astype(bool)
    M = M.loc[:, ~M.columns.duplicated()]
    M.to_pickle(membership_path)
    print(f"[build] membership: {len(M)} snapshots, {M.shape[1]} tickers -> {membership_path}")
    yahoo = sorted(set(sm.to_yahoo_symbol(t) for t in M.columns))
    px_start = (pd.Timestamp(start) - pd.Timedelta(days=600)).strftime("%Y-%m-%d")
    px = sm.download_prices(yahoo, px_start, end, batch=100, retries=3)
    px.to_csv(prices_path)
    print(f"[build] prices: {px.shape} -> {prices_path} ({int(px.notna().any().sum())} symbols with data; delisted tickers have none)")
    import yfinance as yf
    spy = yf.download("SPY", start=px_start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=True, progress=False)["Close"].squeeze()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy.to_csv(spy_path)
    print(f"[build] SPY: {len(spy)} days -> {spy_path}")


def load(prices_path: str, membership_path: str, start: str, end: str):
    px = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    px = px.loc[:end]
    px = px.loc[:, px.notna().sum() > 300]
    M = pd.read_pickle(membership_path)
    # membership columns are raw tickers; map to yahoo symbols, drop delisted-suffixed
    live = [c for c in M.columns if not ("-" in c and c.split("-")[-1].isdigit())]
    M = M[live].copy()
    M.columns = [sm.to_yahoo_symbol(c) for c in M.columns]
    M = M.loc[:, ~M.columns.duplicated()]
    common = [c for c in px.columns if c in M.columns]
    px = px[common]
    # forward-fill membership snapshots onto trading days
    member = M[common].reindex(px.index, method="ffill").fillna(False).astype(bool)
    return px, member


def build(px: pd.DataFrame, member: pd.DataFrame):
    eligible = member & px.notna()
    pxm = px.where(eligible)  # non-members -> NaN so they get no score
    short_mom = sm.rolling_return(pxm, 21)
    long_mom = sm.rolling_return(pxm, 252)
    short_score = sm.score_frame(short_mom)
    long_score = sm.score_frame(long_mom)
    ret10 = sm.rolling_return(px, 10)
    sma20 = px.rolling(20).mean()
    fires = (short_score == 10) & (long_score == 10)
    fires = fires & ~fires.shift(1, fill_value=False)
    fires = fires & eligible
    return fires, short_score, ret10, sma20, eligible


def metrics_block(nav: pd.Series) -> dict:
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1
    return {
        "total_return_pct": round((nav.iloc[-1] / nav.iloc[0] - 1) * 100, 1),
        "cagr_pct": round(((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100, 2),
        "ann_vol_pct": round(r.std() * math.sqrt(252) * 100, 1),
        "sharpe": round(r.mean() / r.std() * math.sqrt(252), 2),
        "max_dd_pct": round(dd.min() * 100, 1),
        "max_dd_date": dd.idxmin().date().isoformat(),
        "calmar": round((((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) / abs(dd.min())), 2),
        "worst_year_pct": round((nav.resample("YE").last() / nav.resample("YE").first() - 1).min() * 100, 1),
        "best_year_pct": round((nav.resample("YE").last() / nav.resample("YE").first() - 1).max() * 100, 1),
        "pct_years_positive": round(((nav.resample("YE").last() / nav.resample("YE").first() - 1) > 0).mean() * 100, 0),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prices", default="prices_long_cache.csv")
    p.add_argument("--membership", default="membership_long.pkl")
    p.add_argument("--spy", default="spy_long_cache.csv")
    p.add_argument("--start", default="2000-01-03", help="first trading day of the back-test")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--cost", type=float, default=0.001)
    p.add_argument("--offsets", type=int, default=10, help="start-date offsets (x4 days) for sensitivity")
    p.add_argument("--out", default="sp500_backtest_long")
    p.add_argument("--build", action="store_true", help="(re)download membership, prices and SPY before running")
    args = p.parse_args(argv)

    import os
    if args.build or not (os.path.exists(args.prices) and os.path.exists(args.membership)):
        build_data(args.prices, args.membership, args.spy, args.start, args.end)

    px, member = load(args.prices, args.membership, args.start, args.end)
    print(f"prices {px.shape[1]} symbols x {len(px)} days ({px.index[0].date()} .. {px.index[-1].date()})")
    print(f"eligible members per day: min {int((member & px.notna()).sum(axis=1).min())}  median {int((member & px.notna()).sum(axis=1).median())}  max {int((member & px.notna()).sum(axis=1).max())}")

    fires, short_score, ret10, sma20, eligible = build(px, member)
    dates = px.index
    start = int(dates.get_indexer([pd.Timestamp(args.start)], method="bfill")[0])
    print(f"back-test from {dates[start].date()}; double_10 fires per eligible stock-year: "
          f"{fires.iloc[start:].to_numpy().sum() / eligible.iloc[start:].sum(axis=1).mean() / ((dates[-1]-dates[start]).days/365.25):.2f}")

    close = px.to_numpy(float)
    strategies = {
        "A_max_return_5pos_63d_stop12": dict(max_pos=5, hold=63, trail=None, stop=0.12, rule_exit=None),
        "B_robust_4pos_126cap_shortneg": dict(max_pos=4, hold=126, trail=None, stop=None, rule_exit="short_neg"),
        "C_5pos_63d_nostop": dict(max_pos=5, hold=63, trail=None, stop=None, rule_exit=None),
    }
    equity = pd.DataFrame(index=dates[start:])
    summary = []
    trades_out = {}
    for name, cfg in strategies.items():
        nav, trades = bt.backtest(close, fires.to_numpy(bool), ret10.to_numpy(float), short_score.to_numpy(float),
                                  sma20.to_numpy(float), cost=args.cost, start=start, **cfg)
        nav = pd.Series(nav[start:] / nav[start], index=dates[start:])
        equity[name] = nav
        m = {"strategy": name, **metrics_block(nav), **bt.trade_stats(trades)}
        # start-date sensitivity
        cs = []
        for k in range(0, args.offsets * 4, 4):
            n2, _ = bt.backtest(close, fires.to_numpy(bool), ret10.to_numpy(float), short_score.to_numpy(float),
                                sma20.to_numpy(float), cost=args.cost, start=start + k, **cfg)
            n2 = pd.Series(n2[start + k:] / n2[start + k], index=dates[start + k:])
            cs.append(metrics_block(n2)["cagr_pct"])
        m["cagr_offsets_min"], m["cagr_offsets_median"], m["cagr_offsets_max"] = round(min(cs), 1), round(float(np.median(cs)), 1), round(max(cs), 1)
        summary.append(m)
        trades_out[name] = trades
        print(f"{name:32s} CAGR {m['cagr_pct']:6.2f}%  maxDD {m['max_dd_pct']:6.1f}% ({m['max_dd_date']})  Sharpe {m['sharpe']}  trades {m['trades']}  win {m['win_rate_pct']}%  PF {m['profit_factor']}  offsets {m['cagr_offsets_min']}/{m['cagr_offsets_median']}/{m['cagr_offsets_max']}")

    # benchmarks
    ew = (px.where(eligible).pct_change().mean(axis=1).fillna(0) + 1).cumprod()
    ew = ew.loc[dates[start]:]
    equity["equal_weight_members"] = (ew / ew.iloc[0]).to_numpy()
    summary.append({"strategy": "equal_weight_members", **metrics_block(equity["equal_weight_members"])})
    try:
        spy = pd.read_csv(args.spy, index_col=0, parse_dates=True).squeeze().reindex(dates).ffill().loc[dates[start]:]
        equity["SPY"] = (spy / spy.iloc[0]).to_numpy()
        summary.append({"strategy": "SPY", **metrics_block(equity["SPY"])})
    except Exception as e:  # noqa: BLE001
        print(f"[spy] unavailable: {e}", file=sys.stderr)
    for b in ("equal_weight_members", "SPY"):
        if b in equity:
            m = summary[-1] if b == "SPY" else summary[-2]
            print(f"{b:32s} CAGR {m['cagr_pct']:6.2f}%  maxDD {m['max_dd_pct']:6.1f}% ({m['max_dd_date']})  Sharpe {m['sharpe']}")

    # survivorship check: same strategies on today's constituents only
    cons = sm.fetch_constituents()
    today = [sm.to_yahoo_symbol(s) for s in cons["Symbol"]]
    keep = [c for c in px.columns if c in today]
    px_t = px[keep]
    elig_t = px_t.notna()
    fires_t, ss_t, r10_t, sma_t, _ = build(px_t, elig_t)
    for name, cfg in strategies.items():
        nav, trades = bt.backtest(px_t.to_numpy(float), fires_t.to_numpy(bool), r10_t.to_numpy(float), ss_t.to_numpy(float),
                                  sma_t.to_numpy(float), cost=args.cost, start=start, **cfg)
        nav = pd.Series(nav[start:] / nav[start], index=dates[start:])
        equity[f"{name}__today_constituents"] = nav
        m = {"strategy": f"{name}__today_constituents_only", **metrics_block(nav), **bt.trade_stats(trades)}
        summary.append(m)
        print(f"{name + ' [today only]':48s} CAGR {m['cagr_pct']:6.2f}%  maxDD {m['max_dd_pct']:6.1f}%  trades {m['trades']}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    yearly = (equity.resample("YE").last() / equity.resample("YE").first() - 1) * 100
    yearly.index = yearly.index.year
    yearly.round(1).to_csv(f"{args.out}_yearly.csv", encoding="utf-8-sig")
    equity.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")
    for name in ("A_max_return_5pos_63d_stop12", "B_robust_4pos_126cap_shortneg"):
        tl = pd.DataFrame(
            [{"Symbol": px.columns[j], "entry_date": dates[et].date(), "exit_date": dates[xt].date(), "hold_days": xt - et,
              "entry_price": round(epx, 2), "exit_price": round(xpx, 2), "return_pct": round(r * 100, 2), "exit_reason": reason}
             for j, et, xt, epx, xpx, r, reason in trades_out[name]]
        ).sort_values("entry_date")
        tl.to_csv(f"{args.out}_trades_{name[0]}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("\nYearly returns (%):")
    print(yearly.round(1).to_string())
    print(f"\nwrote {args.out}_summary.csv, _yearly.csv, _equity.csv, _trades_A.csv, _trades_B.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
