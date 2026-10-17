#!/usr/bin/env python3
"""
Search for high-CAGR concentrated strategies on the point-in-time S&P 500 universe.

Stage 0  signal calibration: for every entry rule, the forward 21/63-day excess return of
         its firings over the whole period (tells which rules have any edge at all).
Stage 1  entry rule x max positions x holding period x priority, no stops.
Stage 2  stops / trailing / rule exits on the best stage-1 configurations.
Stage 3  multiple-testing control: the same number of RANDOM-entry strategies, so the
         best real strategy can be compared with the best of an equal number of lucky
         random ones.

Entry rules: the 15 signals from sp500_entry_points.py plus the momentum-score rules
(double_10, short10_long_pos, long10_short_turn) and reversal rules built from the
cross-sectional scores (short_score -10, deep fall with a thrust, long_score <= -8
with a 10-day thrust).

Outputs: <stem>_calibration.csv, <stem>_grid.csv, <stem>_random.csv, <stem>_best_trades.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys

import numpy as np
import pandas as pd

import sp500_backtest as bt
import sp500_backtest_long as L
import sp500_entry_points as ep
import sp500_momentum as sm


def build_all(px: pd.DataFrame, member: pd.DataFrame):
    eligible = member & px.notna()
    pxm = px.where(eligible)
    f = ep.build_features(px)
    # scores must be cross-sectional within the eligible universe
    f["short_mom"] = sm.rolling_return(pxm, 21)
    f["short_score"] = sm.score_frame(f["short_mom"])
    f["medium_mom"] = sm.rolling_return(pxm, 63)
    f["medium_score"] = sm.score_frame(f["medium_mom"])
    f["long_mom"] = sm.rolling_return(pxm, 252)
    f["long_score"] = sm.score_frame(f["long_mom"])
    sig = ep.build_signals(px, f)
    ss, ls, ms = f["short_score"], f["long_score"], f["medium_score"]
    sig["double_10"] = ep.edge((ss == 10) & (ls == 10))
    sig["short10_long_pos"] = ep.edge((ss == 10) & (ls > 0))
    sig["long10_short_turn"] = ep.edge((ls == 10) & (ss.shift(1) < 0) & (ss >= 3))
    # reversal rules
    sig["rev_short_m10"] = ep.edge(ss == -10)
    sig["rev_short_m10_medium_neg"] = ep.edge((ss == -10) & (ms < 0))
    sig["rev_long_le_m8_thrust"] = ep.edge((ls <= -8) & (f["ret10"] >= 5))
    sig["rev_long_le_m8_short_turn"] = ep.edge((ls <= -8) & (ss.shift(1) < 0) & (ss >= 3))
    sig["rev_double_m8"] = ep.edge((ss <= -8) & (ls <= -8))
    sig["rev_dd30_thrust"] = ep.edge((f["dd_from_hi252"] <= -30) & (f["ret10"] >= 8))
    sig = {k: (v & eligible) for k, v in sig.items()}
    return f, sig, eligible


def calibration(px, sig, eligible, horizons=(21, 63)) -> pd.DataFrame:
    rows = []
    fwd = {h: ((px.shift(-h) / px - 1) * 100).where(eligible) for h in horizons}
    ex = {h: fwd[h].sub(fwd[h].mean(axis=1), axis=0) for h in horizons}
    for name, s in sig.items():
        r = {"signal": name, "fires": int(s.to_numpy().sum())}
        for h in horizons:
            m = s & fwd[h].notna()
            e = ex[h].where(m).stack().dropna()
            f_ = fwd[h].where(m).stack().dropna()
            r[f"excess_{h}d_pct"] = round(float(e.mean()), 3) if len(e) else np.nan
            r[f"hit_{h}d_pct"] = round(float((f_ > 0).mean() * 100), 1) if len(f_) else np.nan
            r[f"tstat_{h}d"] = round(float(e.mean() / e.std() * math.sqrt(len(e))), 2) if len(e) > 2 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("excess_63d_pct", ascending=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prices", default="prices_long_cache.csv")
    p.add_argument("--membership", default="membership_long.pkl")
    p.add_argument("--start", default="2000-01-03")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--cost", type=float, default=0.001)
    p.add_argument("--stage2-top", type=int, default=10)
    p.add_argument("--random-runs", type=int, default=0, help="random strategies for the multiple-testing control (0 = same as stage-1 count)")
    p.add_argument("--out", default="sp500_strategy_search")
    args = p.parse_args(argv)

    px, member = L.load(args.prices, args.membership, args.start, args.end)
    f, sig, eligible = build_all(px, member)
    dates = px.index
    start = int(dates.get_indexer([pd.Timestamp(args.start)], method="bfill")[0])
    close = px.to_numpy(float)
    ss = f["short_score"].to_numpy(float)
    sma20 = f["sma20"].to_numpy(float)
    prios = {"ret10": f["ret10"].to_numpy(float), "deep_dd": (-f["dd_from_hi252"]).to_numpy(float), "long_score": f["long_score"].to_numpy(float)}
    years = (dates[-1] - dates[start]).days / 365.25
    print(f"universe {px.shape[1]} symbols, {dates[start].date()} .. {dates[-1].date()} ({years:.1f} years)")

    cal = calibration(px, sig, eligible)
    cal.to_csv(f"{args.out}_calibration.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\nStage 0: signal calibration (excess vs eligible-universe mean):")
    print(cal.to_string(index=False))

    def run(fires, prio, cfg):
        nav, trades = bt.backtest(close, fires, prio, ss, sma20, cost=args.cost, start=start, **cfg)
        nav = pd.Series(nav[start:] / nav[start], index=dates[start:])
        m = L.metrics_block(nav)
        m.update(bt.trade_stats(trades))
        return m, trades

    rows = []
    stage1 = list(itertools.product(sig.keys(), (3, 5), (21, 63, 126, 252), prios.keys()))
    print(f"\nStage 1: {len(stage1)} runs")
    for i, (entry, mp, hold, pr) in enumerate(stage1, 1):
        m, _ = run(sig[entry].to_numpy(bool), prios[pr], dict(max_pos=mp, hold=hold, trail=None, stop=None, rule_exit=None))
        m.update(entry=entry, max_pos=mp, hold=hold, trail=None, stop=None, rule_exit=None, priority=pr, stage=1)
        rows.append(m)
        if i % 40 == 0:
            print(f"  {i}/{len(stage1)} done; best so far {max(r['cagr_pct'] for r in rows):.1f}%")
    s1 = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False)
    cols = ["entry", "max_pos", "hold", "trail", "stop", "rule_exit", "priority", "cagr_pct", "max_dd_pct", "sharpe", "worst_year_pct", "pct_years_positive", "trades", "win_rate_pct", "profit_factor"]
    print("\nStage 1 top 15:")
    print(s1.head(15)[cols].to_string(index=False))

    print(f"\nStage 2: exits on the top {args.stage2_top}")
    for _, r in s1.head(args.stage2_top).iterrows():
        for trail, stop, rule in itertools.product((None, 0.25), (None, 0.12), (None, "short_neg")):
            if trail is None and stop is None and rule is None:
                continue
            m, _ = run(sig[r.entry].to_numpy(bool), prios[r.priority], dict(max_pos=int(r.max_pos), hold=int(r.hold), trail=trail, stop=stop, rule_exit=rule))
            m.update(entry=r.entry, max_pos=int(r.max_pos), hold=int(r.hold), trail=trail, stop=stop, rule_exit=rule, priority=r.priority, stage=2)
            rows.append(m)
    grid = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False).reset_index(drop=True)
    grid.to_csv(f"{args.out}_grid.csv", index=False, encoding="utf-8-sig")
    print(f"\nAll {len(grid)} runs -> {args.out}_grid.csv. Top 20 by CAGR:")
    print(grid.head(20)[cols].to_string(index=False))
    print(f"\nruns with CAGR >= 30%: {int((grid.cagr_pct >= 30).sum())}   >= 20%: {int((grid.cagr_pct >= 20).sum())}   >= 15%: {int((grid.cagr_pct >= 15).sum())}")

    # Stage 3: random strategies with matched firing rate, same configs as stage 1 shapes
    n_rand = args.random_runs or len(stage1)
    print(f"\nStage 3: {n_rand} random-entry strategies (multiple-testing control)")
    rng = np.random.default_rng(0)
    E = eligible.to_numpy(bool)
    rates = {k: v.to_numpy()[start:].sum() / max(E[start:].sum(), 1) for k, v in sig.items()}
    rrows = []
    shapes = list(itertools.product((3, 5), (21, 63, 126, 252)))
    for i in range(n_rand):
        rate = float(rng.choice(list(rates.values())))
        mp, hold = shapes[i % len(shapes)]
        rnd = (rng.random(close.shape) < rate) & E
        m, _ = run(rnd, rng.random(close.shape), dict(max_pos=mp, hold=hold, trail=None, stop=None, rule_exit=None))
        m.update(max_pos=mp, hold=hold, fire_rate=rate)
        rrows.append(m)
    rand = pd.DataFrame(rrows)
    rand.to_csv(f"{args.out}_random.csv", index=False, encoding="utf-8-sig")
    print(f"random CAGR: median {rand.cagr_pct.median():.1f}%  p90 {rand.cagr_pct.quantile(.9):.1f}%  p99 {rand.cagr_pct.quantile(.99):.1f}%  max {rand.cagr_pct.max():.1f}%   (best real stage-1: {s1.cagr_pct.max():.1f}%)")
    print(f"real stage-1 runs above the random 99th percentile: {int((s1.cagr_pct > rand.cagr_pct.quantile(.99)).sum())} of {len(s1)}")

    best = grid.iloc[0]
    _, trades = run(sig[best.entry].to_numpy(bool), prios[best.priority],
                    dict(max_pos=int(best.max_pos), hold=int(best.hold), trail=None if pd.isna(best.trail) else float(best.trail),
                         stop=None if pd.isna(best.stop) else float(best.stop), rule_exit=None if pd.isna(best.rule_exit) else best.rule_exit))
    tl = pd.DataFrame([{"Symbol": px.columns[j], "entry_date": dates[et].date(), "exit_date": dates[xt].date(), "hold_days": xt - et,
                        "entry_price": round(epx, 2), "exit_price": round(xpx, 2), "return_pct": round(r * 100, 2), "exit_reason": reason}
                       for j, et, xt, epx, xpx, r, reason in trades]).sort_values("entry_date")
    tl.to_csv(f"{args.out}_best_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\nbest: {best.entry} {int(best.max_pos)}pos/{int(best.hold)}d trail={best.trail} stop={best.stop} rule={best.rule_exit} prio={best.priority}: CAGR {best.cagr_pct}% maxDD {best.max_dd_pct}% trades {int(best.trades)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
