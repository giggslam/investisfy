#!/usr/bin/env python3
"""
Portfolio back-test of the entry signals with a small number of concurrent positions.

Rules
-----
* Universe: the cached S&P 500 adjusted closes (prices_cache.csv).
* A signal observed at the close of day t-1 is bought at the close of day t (no look-ahead).
* Equal slot sizing: each new position gets NAV / max_positions (limited by cash).
* Exits are checked at every close: time stop (--hold), trailing stop from the highest
  close since entry, fixed stop loss, and an optional rule exit (short momentum score
  turning negative, or close below the 20-day average). Exit trades at that close.
* When more candidates fire than free slots, candidates are ranked by a priority
  (strongest 10-day thrust, deepest drawdown from the 52-week high, highest long score).
* Transaction cost --cost per side (default 0.10%).

Modes
-----
--grid        : staged grid search over entry signal x max positions x holding period x
                priority (stage 1), then stops / rule exits for the best configurations
                (stage 2). Every run reports full-period, in-sample (<= --split) and
                out-of-sample (> --split) CAGR so over-fitting is visible.
--run ...     : a single configuration with a full trade log and equity curve.

Outputs: <stem>_grid.csv, <stem>_best_trades.csv, <stem>_best_equity.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

import sp500_entry_points as ep
import sp500_momentum as sm

COST = 0.001


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_everything(prices_path: str):
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    f = ep.build_features(prices)
    sig = ep.build_signals(prices, f)
    f["long_mom"] = sm.rolling_return(prices, 252)
    f["long_score"] = sm.score_frame(f["long_mom"])
    # extra entry rules built from the momentum scores
    sig["double_10"] = ep.edge((f["short_score"] == 10) & (f["long_score"] == 10))
    sig["short10_long_pos"] = ep.edge((f["short_score"] == 10) & (f["long_score"] > 0))
    sig["long10_short_turn"] = ep.edge((f["long_score"] == 10) & (f["short_score"].shift(1) < 0) & (f["short_score"] >= 3))
    return prices, f, sig


def benchmark_curves(prices: pd.DataFrame) -> pd.DataFrame:
    ew = (prices.pct_change().mean(axis=1).fillna(0) + 1).cumprod()
    out = pd.DataFrame({"equal_weight_universe": ew})
    try:
        import yfinance as yf

        spy = yf.download("SPY", start=prices.index[0].strftime("%Y-%m-%d"),
                          end=(prices.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)["Close"]
        spy = spy.squeeze()
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        out["SPY"] = (spy.reindex(prices.index).ffill() / spy.reindex(prices.index).ffill().iloc[0])
    except Exception as e:  # noqa: BLE001
        print(f"[benchmark] SPY unavailable ({e}); using equal-weight universe only", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def backtest(
    close: np.ndarray,            # (T, N)
    fires: np.ndarray,            # (T, N) bool, signal observed at close t
    priority: np.ndarray,         # (T, N) higher = buy first
    short_score: np.ndarray,      # (T, N)
    sma20: np.ndarray,            # (T, N)
    max_pos: int,
    hold: int | None,
    trail: float | None,
    stop: float | None,
    rule_exit: str | None,
    cost: float = COST,
    start: int = 0,
):
    T, N = close.shape
    cash = 1.0
    nav = np.ones(T)
    held: dict[int, list] = {}  # j -> [shares, entry_px, entry_t, high]
    trades = []
    for t in range(start, T):
        px = close[t]
        # ---- exits at this close
        for j in list(held):
            sh, epx, et, hi = held[j]
            p = px[j]
            if np.isnan(p):
                continue
            hi = max(hi, p)
            held[j][3] = hi
            reason = None
            if hold is not None and t - et >= hold:
                reason = "time"
            elif stop is not None and p / epx - 1 <= -stop:
                reason = "stop"
            elif trail is not None and p / hi - 1 <= -trail:
                reason = "trail"
            elif rule_exit == "short_neg" and t - et >= 5 and short_score[t, j] < 0:
                reason = "short_neg"
            elif rule_exit == "sma20" and t - et >= 5 and p < sma20[t, j]:
                reason = "sma20"
            if reason:
                cash += sh * p * (1 - cost)
                trades.append((j, et, t, epx, p, p / epx - 1, reason))
                del held[j]
        # ---- entries: signals from the previous close, bought at this close
        free = max_pos - len(held)
        if t > 0 and free > 0:
            cand = np.where(fires[t - 1])[0]
            cand = [j for j in cand if j not in held and not np.isnan(px[j])]
            if cand:
                cand.sort(key=lambda j: -priority[t - 1, j] if not np.isnan(priority[t - 1, j]) else math.inf)
                nav_now = cash + sum(sh * px[j] for j, (sh, *_rest) in held.items())
                slot = nav_now / max_pos
                for j in cand[:free]:
                    amt = min(slot, cash)
                    if amt <= 1e-9:
                        break
                    sh = amt * (1 - cost) / px[j]
                    cash -= amt
                    held[j] = [sh, px[j], t, px[j]]
        nav[t] = cash + sum(sh * (px[j] if not np.isnan(px[j]) else epx) for j, (sh, epx, *_r) in held.items())
    # close out open positions at the end for the trade log (NAV already marks them)
    for j, (sh, epx, et, hi) in held.items():
        trades.append((j, et, T - 1, epx, close[T - 1, j], close[T - 1, j] / epx - 1, "open"))
    return nav, trades


def metrics(nav: np.ndarray, dates: pd.DatetimeIndex, split: pd.Timestamp | None = None) -> dict:
    s = pd.Series(nav, index=dates)
    def cagr(x: pd.Series) -> float:
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        return (x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 and x.iloc[0] > 0 else np.nan
    r = s.pct_change().dropna()
    dd = (s / s.cummax() - 1).min()
    out = {
        "total_return_pct": round((s.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(cagr(s) * 100, 1),
        "max_dd_pct": round(dd * 100, 1),
        "sharpe": round(r.mean() / r.std() * math.sqrt(252), 2) if r.std() > 0 else np.nan,
        "calmar": round(cagr(s) / abs(dd), 2) if dd < 0 else np.nan,
    }
    if split is not None:
        ins, oos = s.loc[:split], s.loc[split:]
        out["is_cagr_pct"] = round(cagr(ins) * 100, 1) if len(ins) > 20 else np.nan
        out["oos_cagr_pct"] = round(cagr(oos) * 100, 1) if len(oos) > 20 else np.nan
        out["oos_max_dd_pct"] = round((oos / oos.cummax() - 1).min() * 100, 1) if len(oos) > 20 else np.nan
    return out


def trade_stats(trades: list) -> dict:
    if not trades:
        return {"trades": 0}
    r = np.array([x[5] for x in trades])
    d = np.array([x[2] - x[1] for x in trades])
    wins = r[r > 0]
    losses = r[r <= 0]
    return {
        "trades": len(trades),
        "win_rate_pct": round((r > 0).mean() * 100, 1),
        "avg_trade_pct": round(r.mean() * 100, 2),
        "avg_win_pct": round(wins.mean() * 100, 2) if len(wins) else np.nan,
        "avg_loss_pct": round(losses.mean() * 100, 2) if len(losses) else np.nan,
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if len(losses) and losses.sum() < 0 else np.nan,
        "avg_hold_days": round(d.mean(), 1),
    }


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prices", default="prices_cache.csv")
    p.add_argument("--out", default="sp500_backtest")
    p.add_argument("--cost", type=float, default=COST, help="one-way transaction cost (default 0.001)")
    p.add_argument("--split", default="2025-12-31", help="in-sample / out-of-sample split date")
    p.add_argument("--grid", action="store_true", help="run the staged grid search")
    p.add_argument("--stage2-top", type=int, default=12, help="how many stage-1 configs get the stop/exit grid")
    p.add_argument("--run", nargs="*", help="single run: entry max_pos hold trail stop rule priority (use none for null)")
    args = p.parse_args(argv)

    prices, f, sig = load_everything(args.prices)
    dates = prices.index
    split = pd.Timestamp(args.split)
    close = prices.to_numpy(dtype=float)
    short_score = f["short_score"].to_numpy(dtype=float)
    sma20 = f["sma20"].to_numpy(dtype=float)
    priorities = {
        "ret10": f["ret10"].to_numpy(dtype=float),
        "deep_dd": (-f["dd_from_hi252"]).to_numpy(dtype=float),
        "long_score": f["long_score"].to_numpy(dtype=float),
    }
    # first day with a long score so every entry rule has the same start
    start = int(np.argmax(f["long_score"].notna().any(axis=1).to_numpy()))
    print(f"universe {close.shape[1]} stocks, {dates[0].date()} .. {dates[-1].date()}, trading from {dates[start].date()}, split {split.date()}")

    bench = benchmark_curves(prices)
    bench = bench.loc[dates[start]:]
    bench = bench / bench.iloc[0]
    for col in bench.columns:
        m = metrics(bench[col].to_numpy(), bench.index, split)
        print(f"benchmark {col:24s} total {m['total_return_pct']:7.1f}%  CAGR {m['cagr_pct']:5.1f}%  maxDD {m['max_dd_pct']:6.1f}%  Sharpe {m['sharpe']}  IS {m['is_cagr_pct']}  OOS {m['oos_cagr_pct']}")

    def one(entry, max_pos, hold, trail, stop, rule, prio):
        nav, trades = backtest(close, sig[entry].to_numpy(dtype=bool), priorities[prio], short_score, sma20,
                               max_pos, hold, trail, stop, rule, args.cost, start)
        nav = nav[start:]
        m = metrics(nav / nav[0], dates[start:], split)
        m.update(trade_stats(trades))
        m.update({"entry": entry, "max_pos": max_pos, "hold": hold, "trail": trail, "stop": stop, "rule_exit": rule, "priority": prio})
        return m, nav / nav[0], trades

    if args.run:
        e, mp, hd, tr, st, ru, pr = args.run
        conv = lambda x, fn: None if x.lower() == "none" else fn(x)  # noqa: E731
        m, nav, trades = one(e, int(mp), conv(hd, int), conv(tr, float), conv(st, float), conv(ru, str), pr)
        print(pd.Series(m).to_string())
        save_best(m, nav, trades, prices, dates[start:], bench, args.out)
        return 0

    if not args.grid:
        p.print_help()
        return 0

    entries = ["thrust_after_fall", "turn_up", "ret10_gt_8", "double_10", "short10_long_pos", "long10_short_turn",
               "short_score_ge8", "breakout_52w_high", "deep_dd_cross_sma50", "short_score_flip_pos", "rebound_from_52w_low"]
    rows = []
    print("\nstage 1: entry x max_pos x hold x priority (no stops)")
    for entry, mp, hold, prio in itertools.product(entries, (3, 4, 5), (21, 42, 63, 126), priorities):
        m, _, _ = one(entry, mp, hold, None, None, None, prio)
        m["stage"] = 1
        rows.append(m)
    s1 = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False)
    print(s1.head(15)[["entry", "max_pos", "hold", "priority", "total_return_pct", "cagr_pct", "max_dd_pct", "sharpe", "is_cagr_pct", "oos_cagr_pct", "trades", "win_rate_pct"]].to_string(index=False))

    print(f"\nstage 2: stops / rule exits for the top {args.stage2_top} stage-1 configurations")
    seen = set()
    for _, r in s1.head(args.stage2_top).iterrows():
        key = (r.entry, r.max_pos, r.hold, r.priority)
        if key in seen:
            continue
        seen.add(key)
        for trail, stop, rule in itertools.product((None, 0.10, 0.15, 0.25), (None, 0.08, 0.12), (None, "short_neg", "sma20")):
            if trail is None and stop is None and rule is None:
                continue
            m, _, _ = one(r.entry, int(r.max_pos), int(r.hold), trail, stop, rule, r.priority)
            m["stage"] = 2
            rows.append(m)
        # also: no time stop, exits only by trail / rule
        for trail, stop, rule in itertools.product((0.15, 0.25), (None, 0.12), ("short_neg", "sma20", None)):
            m, _, _ = one(r.entry, int(r.max_pos), None, trail, stop, rule, r.priority)
            m["stage"] = 2
            rows.append(m)

    grid = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False).reset_index(drop=True)
    grid.to_csv(f"{args.out}_grid.csv", index=False, encoding="utf-8-sig")
    cols = ["entry", "max_pos", "hold", "trail", "stop", "rule_exit", "priority", "total_return_pct", "cagr_pct", "max_dd_pct", "sharpe", "calmar",
            "is_cagr_pct", "oos_cagr_pct", "oos_max_dd_pct", "trades", "win_rate_pct", "profit_factor", "avg_hold_days"]
    print(f"\nall {len(grid)} runs -> {args.out}_grid.csv")
    print("\nTop 20 by full-period CAGR:")
    print(grid.head(20)[cols].to_string(index=False))
    print("\nTop 10 by in-sample CAGR (2024-2025) and how they did out of sample (2026):")
    print(grid.sort_values("is_cagr_pct", ascending=False).head(10)[cols].to_string(index=False))
    print("\nTop 10 by Calmar (CAGR / max drawdown):")
    print(grid.sort_values("calmar", ascending=False).head(10)[cols].to_string(index=False))
    print("\nBest configuration per entry signal:")
    print(grid.sort_values("cagr_pct", ascending=False).groupby("entry").head(1)[cols].to_string(index=False))

    best = grid.iloc[0]
    m, nav, trades = one(best.entry, int(best.max_pos), None if pd.isna(best.hold) else int(best.hold),
                         None if pd.isna(best.trail) else float(best.trail), None if pd.isna(best.stop) else float(best.stop),
                         None if pd.isna(best.rule_exit) else best.rule_exit, best.priority)
    save_best(m, nav, trades, prices, dates[start:], bench, args.out)
    return 0


def save_best(m, nav, trades, prices, dates, bench, stem):
    eq = pd.DataFrame({"strategy": nav}, index=dates)
    for c in bench.columns:
        eq[c] = bench[c].reindex(dates).to_numpy()
    eq.to_csv(f"{stem}_best_equity.csv", encoding="utf-8-sig")
    cons = sm.fetch_constituents()
    meta = dict(zip(cons["Symbol"].map(sm.to_yahoo_symbol), cons["Symbol"]))
    full_dates = prices.index
    tl = pd.DataFrame(
        [{"Symbol": meta.get(prices.columns[j], prices.columns[j]), "entry_date": full_dates[et].date(), "exit_date": full_dates[xt].date(),
          "hold_days": xt - et, "entry_price": round(epx, 2), "exit_price": round(xpx, 2), "return_pct": round(r * 100, 2), "exit_reason": reason}
         for j, et, xt, epx, xpx, r, reason in trades]
    ).sort_values("entry_date")
    tl.to_csv(f"{stem}_best_trades.csv", index=False, encoding="utf-8-sig")
    yearly = eq["strategy"].resample("YE").last() / eq["strategy"].resample("YE").first() - 1
    print(f"\nbest configuration: {m['entry']} max_pos={m['max_pos']} hold={m['hold']} trail={m['trail']} stop={m['stop']} rule={m['rule_exit']} priority={m['priority']}")
    print(f"  total {m['total_return_pct']}%  CAGR {m['cagr_pct']}%  maxDD {m['max_dd_pct']}%  Sharpe {m['sharpe']}  trades {m['trades']}  win {m['win_rate_pct']}%  PF {m['profit_factor']}")
    print("  yearly:", {d.year: f"{v*100:.1f}%" for d, v in yearly.items()})
    print(f"  exit reasons: {tl['exit_reason'].value_counts().to_dict()}")
    print(f"  wrote {stem}_best_equity.csv, {stem}_best_trades.csv")


if __name__ == "__main__":
    sys.exit(main())
