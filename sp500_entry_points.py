#!/usr/bin/env python3
"""
Reverse-engineer entry points from the winners (hindsight event study).

1. WINNING RUNS (hindsight): a percentage zigzag (default 10% reversal) splits each
   stock's price path into legs. Every trough->peak leg that gained at least
   --min-gain (default 30%) is a "winning run". The trough is the true start of the
   rise; it is only knowable after the fact.
2. OBSERVABLE SIGNALS (real time): a set of entry rules that use only prices up to
   each day (moving-average crosses, breakouts, rebounds from the 52-week low,
   momentum-score flips, and "turn" signals: a 10-day thrust starting from a falling
   state). Each fires on the FIRST day its condition becomes true.
3. MATCH: for every winning run, the first day each signal fired after the trough:
   lag in days, entry price, how much of the run was still ahead (captured %).
4. PRECISION: every firing of each signal, inside or outside a winning run, is scored
   with its realised forward 63-day return (hit rate, excess vs the universe mean, big
   win rate, bust rate) and how often it fires.
5. CANDIDATES: stocks where the best-ranked signals fired in the last --recent days.

Outputs (stem from --out, default sp500_entry):
    <stem>_runs.csv        one row per winning run with features at the trough and
                           every signal's first trigger (lag / entry / captured %)
    <stem>_signals.csv     signal league table (coverage, lag, captured, precision)
    <stem>_trough_profile.csv  what the stock looked like at T0, T0+5, +10, +20
    <stem>_candidates.csv  recent firings of the top signals (actionable list)

Usage:
    python sp500_entry_points.py                       # uses prices_cache.csv
    python sp500_entry_points.py --reversal 0.08 --min-gain 0.40 --recent 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import sp500_momentum as sm

FWD = 63  # forward window (trading days) used to judge a firing


# --------------------------------------------------------------------------- #
# 1. hindsight runs
# --------------------------------------------------------------------------- #
def zigzag_pivots(v: np.ndarray, rev: float) -> tuple[list[tuple[str, int]], int | None]:
    """Percentage zigzag. Returns confirmed pivots [('L'|'H', idx), ...] and the index of
    the running high of an unfinished up-leg (None if the last leg is down)."""
    n = len(v)
    hi = lo = 0
    trend = 0
    piv: list[tuple[str, int]] = []
    for i in range(1, n):
        if v[i] > v[hi]:
            hi = i
        if v[i] < v[lo]:
            lo = i
        if trend >= 0 and v[i] <= v[hi] * (1 - rev):
            piv.append(("H", hi))
            trend, lo, hi = -1, i, i
        elif trend <= 0 and v[i] >= v[lo] * (1 + rev):
            piv.append(("L", lo))
            trend, hi, lo = 1, i, i
    return piv, (hi if trend == 1 else None)


def find_runs(prices: pd.DataFrame, rev: float, min_gain: float) -> pd.DataFrame:
    rows = []
    for sym in prices.columns:
        s = prices[sym].dropna()
        if len(s) < 30:
            continue
        v = s.to_numpy(dtype=float)
        idx = s.index
        piv, open_hi = zigzag_pivots(v, rev)
        legs = [(a, b, False) for (t1, a), (t2, b) in zip(piv, piv[1:]) if t1 == "L" and t2 == "H"]
        if piv and piv[-1][0] == "L" and open_hi is not None:
            legs.append((piv[-1][1], open_hi, True))
        for a, b, ongoing in legs:
            gain = v[b] / v[a] - 1.0
            if gain < min_gain:
                continue
            rows.append(
                {
                    "yahoo": sym,
                    "trough_date": idx[a],
                    "trough_price": round(float(v[a]), 4),
                    "peak_date": idx[b],
                    "peak_price": round(float(v[b]), 4),
                    "gain_pct": round(gain * 100, 2),
                    "run_days": int(b - a),
                    "ongoing": ongoing,
                    "ia": a,
                    "ib": b,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. observable features and signals (no look-ahead)
# --------------------------------------------------------------------------- #
def edge(cond: pd.DataFrame) -> pd.DataFrame:
    """True only on the first day a condition becomes true."""
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(1, fill_value=False)


def build_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    f: dict[str, pd.DataFrame] = {}
    c = prices
    f["sma20"] = c.rolling(20).mean()
    f["sma50"] = c.rolling(50).mean()
    f["sma200"] = c.rolling(200).mean()
    for d in (5, 10, 20, 63):
        f[f"ret{d}"] = (c / c.shift(d) - 1) * 100
    f["hi252"] = c.rolling(252, min_periods=120).max()
    f["lo252"] = c.rolling(252, min_periods=120).min()
    f["dd_from_hi252"] = (c / f["hi252"] - 1) * 100  # <= 0
    f["up_from_lo252"] = (c / f["lo252"] - 1) * 100  # >= 0
    f["pct_vs_sma20"] = (c / f["sma20"] - 1) * 100
    f["pct_vs_sma50"] = (c / f["sma50"] - 1) * 100
    f["pct_vs_sma200"] = (c / f["sma200"] - 1) * 100
    f["sma20_slope5"] = (f["sma20"] / f["sma20"].shift(5) - 1) * 100
    f["hi20_prev"] = c.shift(1).rolling(20).max()
    f["hi50_prev"] = c.shift(1).rolling(50).max()
    f["hi252_prev"] = c.shift(1).rolling(252, min_periods=120).max()
    # momentum scores from the main scorer (cross-sectional, point-in-time)
    f["short_mom"] = sm.rolling_return(c, 21)
    f["short_score"] = sm.score_frame(f["short_mom"])
    f["medium_mom"] = sm.rolling_return(c, 63)
    f["medium_score"] = sm.score_frame(f["medium_mom"])
    return f


def build_signals(c: pd.DataFrame, f: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    above20 = c > f["sma20"]
    above50 = c > f["sma50"]
    s: dict[str, pd.DataFrame] = {}
    s["cross_sma20"] = edge(above20)
    s["cross_sma50"] = edge(above50)
    s["sma20_over_sma50"] = edge(f["sma20"] > f["sma50"])
    s["breakout_20d_high"] = edge(c > f["hi20_prev"])
    s["breakout_50d_high"] = edge(c > f["hi50_prev"])
    s["breakout_52w_high"] = edge(c > f["hi252_prev"])
    s["rebound_from_52w_low"] = edge((f["up_from_lo252"] <= 20) & (f["ret10"] >= 8))
    s["ret10_gt_8"] = edge(f["ret10"] >= 8)
    s["short_score_flip_pos"] = edge((f["short_score"].shift(1) < 0) & (f["short_score"] >= 3))
    s["short_score_ge8"] = edge(f["short_score"] >= 8)
    s["trend_confirm"] = edge(above20 & (f["sma20"] > f["sma50"]) & (f["sma20_slope5"] > 0) & (f["ret20"] > 0))
    s["pullback_resume"] = edge(above20 & (f["sma20"] > f["sma50"]) & (c.shift(1) < f["sma20"].shift(1)) & (f["pct_vs_sma50"] > 0))
    s["deep_dd_cross_sma50"] = edge(above50 & (f["dd_from_hi252"].shift(1) <= -25))
    # "turn" signals: a 10-day thrust that starts from a falling state
    s["thrust_after_fall"] = edge((f["ret10"] >= 8) & (f["short_score"].shift(1) <= -6))
    s["turn_up"] = edge((f["ret10"] >= 8) & (f["dd_from_hi252"] <= -15) & (c.shift(5) < f["sma50"].shift(5)))
    return s


# --------------------------------------------------------------------------- #
# 3 + 4. match signals to runs, score every firing
# --------------------------------------------------------------------------- #
def match_runs(runs: pd.DataFrame, prices: pd.DataFrame, f: dict, signals: dict, early: int) -> pd.DataFrame:
    runs = runs.copy()
    pos = {sym: prices[sym].dropna() for sym in prices.columns}
    feat_at_trough = ["dd_from_hi252", "up_from_lo252", "ret20", "ret63", "pct_vs_sma50", "pct_vs_sma200", "short_score", "medium_score"]
    out_rows = []
    for r in runs.itertuples(index=False):
        sym = r.yahoo
        s = pos[sym]
        dates = s.index
        a, b = r.ia, r.ib
        t0 = dates[a]
        row = {}
        for k in feat_at_trough:
            row[f"{k}_at_trough"] = round(float(f[k].at[t0, sym]), 2) if pd.notna(f[k].at[t0, sym]) else np.nan
        for name, sig in signals.items():
            fired = sig[sym].reindex(dates[a + 1 : b + 1])
            hit = fired[fired].index
            if len(hit):
                t = hit[0]
                lag = int(dates.get_loc(t) - a)
                entry = float(s.at[t])
                remaining = r.peak_price / entry - 1
                row[f"{name}_lag"] = lag
                row[f"{name}_entry"] = round(entry, 4)
                row[f"{name}_captured_pct"] = round(remaining / (r.gain_pct / 100) * 100, 1)
                row[f"{name}_remaining_gain_pct"] = round(remaining * 100, 2)
            else:
                row[f"{name}_lag"] = np.nan
                row[f"{name}_entry"] = np.nan
                row[f"{name}_captured_pct"] = np.nan
                row[f"{name}_remaining_gain_pct"] = np.nan
        out_rows.append(row)
    return pd.concat([runs.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)


def score_signals(runs_m: pd.DataFrame, prices: pd.DataFrame, signals: dict, runs: pd.DataFrame, early: int) -> pd.DataFrame:
    fwd = (prices.shift(-FWD) / prices - 1) * 100
    fwd_excess = fwd.sub(fwd.mean(axis=1), axis=0)
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    n_stocks = prices.shape[1]
    completed = runs_m[~runs_m["ongoing"]]

    # mark (date, symbol) cells that lie inside a winning run (trough, peak]
    in_run = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    early_run = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for r in runs.itertuples(index=False):
        in_run.loc[r.trough_date : r.peak_date, r.yahoo] = True
        in_run.at[r.trough_date, r.yahoo] = False
        s_idx = prices[r.yahoo].dropna().index
        e = min(r.ia + early, r.ib)
        early_run.loc[s_idx[r.ia + 1] : s_idx[e], r.yahoo] = True
    base_in = float(in_run.to_numpy().mean() * 100)
    base_early = float(early_run.to_numpy().mean() * 100)

    rows = []
    for name, sig in signals.items():
        fires = sig & fwd.notna()
        n = int(fires.to_numpy().sum())
        fw = fwd.where(fires).stack().dropna()
        ex = fwd_excess.where(fires).stack().dropna()
        inrun = in_run.where(fires).stack().dropna()
        erun = early_run.where(fires).stack().dropna()
        lag = completed[f"{name}_lag"]
        cap = completed[f"{name}_captured_pct"]
        rows.append(
            {
                "signal": name,
                "fires": n,
                "fires_per_stock_year": round(n / n_stocks / years, 2),
                "coverage_pct": round(lag.notna().mean() * 100, 1),
                f"coverage_within_{early}d_pct": round((lag <= early).mean() * 100, 1),
                "median_lag_days": float(lag.median()) if lag.notna().any() else np.nan,
                "median_captured_pct": round(float(cap.median()), 1) if cap.notna().any() else np.nan,
                "median_remaining_gain_pct": round(float(completed[f"{name}_remaining_gain_pct"].median()), 1),
                "in_winning_run_pct": round(float(inrun.mean() * 100), 1) if len(inrun) else np.nan,
                f"early_in_run_{early}d_pct": round(float(erun.mean() * 100), 1) if len(erun) else np.nan,
                f"early_lift_vs_base": round(float(erun.mean() * 100) / base_early, 2) if len(erun) and base_early else np.nan,
                f"fwd{FWD}_mean_pct": round(float(fw.mean()), 2),
                f"fwd{FWD}_median_pct": round(float(fw.median()), 2),
                f"fwd{FWD}_excess_pct": round(float(ex.mean()), 2),
                f"fwd{FWD}_hit_rate_pct": round(float((fw > 0).mean() * 100), 1),
                f"fwd{FWD}_big_win_ge15_pct": round(float((fw >= 15).mean() * 100), 1),
                f"fwd{FWD}_bust_le_m10_pct": round(float((fw <= -10).mean() * 100), 1),
            }
        )
    tbl = pd.DataFrame(rows)
    # rank: precision (excess) and usefulness (coverage x captured) both matter
    # rank: how often a firing is EARLY in a winning run and how it pays, with a smaller
    # weight on how much of the run is left and how many runs it catches at all
    tbl["rank_score"] = (
        tbl[f"fwd{FWD}_excess_pct"].rank(pct=True) + tbl[f"early_in_run_{early}d_pct"].rank(pct=True)
        + tbl["median_captured_pct"].rank(pct=True) * 0.5 + tbl["coverage_pct"].rank(pct=True) * 0.5
    ).round(3)
    print(f"baseline: a random day is inside a winning run {base_in:.1f}% of the time, in its first {early} days {base_early:.1f}%")
    return tbl.sort_values("rank_score", ascending=False).reset_index(drop=True)


def trough_profile(runs: pd.DataFrame, prices: pd.DataFrame, f: dict, offsets=(0, 5, 10, 20)) -> pd.DataFrame:
    keys = ["ret5", "ret10", "ret20", "pct_vs_sma20", "pct_vs_sma50", "pct_vs_sma200", "dd_from_hi252", "up_from_lo252", "short_score", "medium_score"]
    rows = []
    for off in offsets:
        vals = {k: [] for k in keys}
        above = {"above_sma20": [], "above_sma50": [], "sma20_gt_sma50": []}
        for r in runs.itertuples(index=False):
            s = prices[r.yahoo].dropna()
            i = r.ia + off
            if i > r.ib or i >= len(s):
                continue
            t = s.index[i]
            for k in keys:
                vals[k].append(f[k].at[t, r.yahoo])
            above["above_sma20"].append(s.iat[i] > f["sma20"].at[t, r.yahoo])
            above["above_sma50"].append(s.iat[i] > f["sma50"].at[t, r.yahoo])
            above["sma20_gt_sma50"].append(f["sma20"].at[t, r.yahoo] > f["sma50"].at[t, r.yahoo])
        row = {"days_after_trough": off, "n": len(vals["ret5"])}
        for k in keys:
            row[f"{k}_median"] = round(float(np.nanmedian(vals[k])), 2)
        for k, v in above.items():
            row[f"{k}_pct"] = round(float(np.mean(v)) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def recent_candidates(prices, f, signals, top_signals, recent, meta) -> pd.DataFrame:
    last = prices.index[-1]
    window = prices.index[-recent:]
    rows = []
    for sym in prices.columns:
        fired = {name: signals[name][sym].reindex(window) for name in top_signals}
        hits = {n: s[s].index for n, s in fired.items()}
        hits = {n: d for n, d in hits.items() if len(d)}
        if not hits:
            continue
        rows.append(
            {
                "yahoo": sym,
                "last_close": round(float(prices[sym].iat[-1]), 2),
                "signals_fired": ", ".join(f"{n}@{d[-1].date()}" for n, d in hits.items()),
                "n_top_signals": len(hits),
                "ret20_pct": round(float(f["ret20"].at[last, sym]), 2),
                "pct_vs_sma50": round(float(f["pct_vs_sma50"].at[last, sym]), 2),
                "pct_vs_sma200": round(float(f["pct_vs_sma200"].at[last, sym]), 2),
                "dd_from_hi252_pct": round(float(f["dd_from_hi252"].at[last, sym]), 2),
                "up_from_lo252_pct": round(float(f["up_from_lo252"].at[last, sym]), 2),
                "short_score": f["short_score"].at[last, sym],
                "medium_score": f["medium_score"].at[last, sym],
                "short_score_20d_ago": f["short_score"].shift(20).at[last, sym],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.merge(meta, on="yahoo", how="left").drop(columns="yahoo")
    front = ["Symbol", "Security", "Sector"]
    df = df[front + [c for c in df.columns if c not in front]]
    return df.sort_values(["n_top_signals", "ret20_pct"], ascending=[False, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prices", default="prices_cache.csv", help="adjusted-close CSV from sp500_momentum.py")
    p.add_argument("--reversal", type=float, default=0.10, help="zigzag reversal fraction (default 0.10)")
    p.add_argument("--min-gain", type=float, default=0.30, help="minimum trough->peak gain to count as a winning run (default 0.30)")
    p.add_argument("--early", type=int, default=20, help="'early entry' threshold in days after the trough (default 20)")
    p.add_argument("--recent", type=int, default=10, help="candidate list: signals fired in the last N trading days (default 10)")
    p.add_argument("--top", type=int, default=3, help="number of top-ranked signals used for the candidate list")
    p.add_argument("--out", default="sp500_entry", help="output file stem")
    args = p.parse_args(argv)

    if not os.path.exists(args.prices):
        raise SystemExit(f"{args.prices} not found; run sp500_momentum.py first to build the price cache")
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True).sort_index()
    cons = sm.fetch_constituents()
    meta = cons.assign(yahoo=cons["Symbol"].map(sm.to_yahoo_symbol))[["yahoo", "Symbol", "Security", "Sector"]]
    print(f"prices: {prices.shape[1]} symbols, {prices.index[0].date()} .. {prices.index[-1].date()} ({len(prices)} days)")

    runs = find_runs(prices, args.reversal, args.min_gain)
    print(f"winning runs (>= {args.min_gain:.0%} after {args.reversal:.0%} zigzag): {len(runs)} "
          f"({int(runs['ongoing'].sum())} still ongoing), median gain {runs['gain_pct'].median():.1f}%, median length {runs['run_days'].median():.0f}d")

    f = build_features(prices)
    signals = build_signals(prices, f)
    runs_m = match_runs(runs, prices, f, signals, args.early)
    tbl = score_signals(runs_m, prices, signals, runs, args.early)
    prof = trough_profile(runs, prices, f)

    top_signals = tbl["signal"].head(args.top).tolist()
    cands = recent_candidates(prices, f, signals, top_signals, args.recent, meta)

    runs_out = runs_m.merge(meta, on="yahoo", how="left").drop(columns=["yahoo", "ia", "ib"])
    front = ["Symbol", "Security", "Sector"]
    runs_out = runs_out[front + [c for c in runs_out.columns if c not in front]].sort_values("trough_date")
    runs_out.to_csv(f"{args.out}_runs.csv", index=False, encoding="utf-8-sig")
    tbl.to_csv(f"{args.out}_signals.csv", index=False, encoding="utf-8-sig")
    prof.to_csv(f"{args.out}_trough_profile.csv", index=False, encoding="utf-8-sig")
    cands.to_csv(f"{args.out}_candidates.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("\nWhat a winning run looks like N days after its (hindsight) trough:")
    print(prof.to_string(index=False))
    print("\nSignal league table (completed runs only for coverage/lag/captured; all firings for precision):")
    show = ["signal", "fires_per_stock_year", "coverage_pct", f"coverage_within_{args.early}d_pct", "median_lag_days", "median_captured_pct",
            "in_winning_run_pct", f"early_in_run_{args.early}d_pct", "early_lift_vs_base", f"fwd{FWD}_excess_pct", f"fwd{FWD}_hit_rate_pct", f"fwd{FWD}_big_win_ge15_pct", f"fwd{FWD}_bust_le_m10_pct", "rank_score"]
    print(tbl[show].to_string(index=False))
    print(f"\nTop signals used for candidates: {top_signals}")
    print(f"Candidates (fired in last {args.recent} trading days): {len(cands)}")
    if not cands.empty:
        print(cands.head(25).to_string(index=False))
    print(f"\nWrote {args.out}_runs.csv, {args.out}_signals.csv, {args.out}_trough_profile.csv, {args.out}_candidates.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
