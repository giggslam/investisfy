#!/usr/bin/env python3
"""
Monthly "hold the N strongest momentum stocks" back-test on the point-in-time S&P 500 universe.

Sections 6-8 traded the +10/+10 signal with slot filling and 63-day holds. This script tests the
plainer rule the question actually asks: at every month end rank the eligible members by momentum,
buy the top N equal-weighted at the next close, hold for a month, repeat.

Rankings (all computed from prices up to the month-end close, members only):
  short     21-day return                (the repo's short-term momentum)
  medium    63-day return                (medium-term)
  long      252-day return               (long-term)
  12_1      252-day return skipping the last 21 days (classic academic momentum)
  composite mean of the percentile ranks of short, medium and long

Execution: signal at month-end close t, trades at the next trading day's close t+1, 0.1% cost per unit
of weight traded. Within the month the weights drift with prices (buy and hold). A stock that delists
mid-month is carried at its last price (zero return) until the next rebalance. Eligibility = S&P 500
member on the signal date (historical membership) with a price and 252 days of history.

Periods reported: 2000-2024 (point-in-time universe), 2010-2019, 2020-2024, plus 2025-01..2026-10 on
today's constituents (prices_cache.csv) so it can be compared with section 8.2.
Controls: SPY, QQQ, equal-weight members (monthly rebalanced), and 200 random "pick N members at
random each month" portfolios.

Outputs: <out>_summary.csv, <out>_yearly.csv, <out>_periods.csv, <out>_equity.csv,
         <out>_random.csv, docs/top5_monthly.png
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import sp500_backtest_long as L
import sp500_beat_qqq as B

COST = 0.001


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().to_numpy())


def rank_frames(px: pd.DataFrame, eligible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # momentum from each stock's own history, then masked to the members eligible on the signal date
    r21 = (px / px.shift(21) - 1).where(eligible)
    r63 = (px / px.shift(63) - 1).where(eligible)
    r252 = (px / px.shift(252) - 1).where(eligible)
    r12_1 = (px.shift(21) / px.shift(252) - 1).where(eligible)
    comp = (r21.rank(axis=1, pct=True) + r63.rank(axis=1, pct=True) + r252.rank(axis=1, pct=True)) / 3
    return {"short": r21, "medium": r63, "long": r252, "12_1": r12_1, "composite": comp}


def run_portfolio(px: pd.DataFrame, picks: dict[pd.Timestamp, list[str]], cost: float = COST) -> pd.Series:
    """
    picks: signal date -> list of tickers (equal weight). Trades at the next trading day's close.
    Returns a daily NAV series starting at the first execution date (NAV = 1 there).
    """
    idx = px.index
    pxf = px.ffill()
    rets = pxf.pct_change().to_numpy()
    rets[~np.isfinite(rets)] = 0.0
    cols = {c: i for i, c in enumerate(px.columns)}
    sig_dates = sorted(picks)
    pos = {d: idx.get_indexer([d])[0] for d in sig_dates}
    exec_pos = [pos[d] + 1 for d in sig_dates if pos[d] + 1 < len(idx)]
    nav = np.full(len(idx), np.nan)
    w = np.zeros(len(px.columns))
    value = 1.0
    for k, e in enumerate(exec_pos):
        d = sig_dates[k]
        target = np.zeros_like(w)
        names = [n for n in picks[d] if n in cols]
        if names:
            target[[cols[n] for n in names]] = 1.0 / len(names)
        # weights at close e before trading are the drifted weights (already in w); trade to target
        value *= 1 - np.abs(target - w).sum() * cost
        w = target.copy()
        nav[e] = value
        stop = exec_pos[k + 1] if k + 1 < len(exec_pos) else len(idx)
        for t in range(e + 1, stop):
            g = 1 + rets[t]
            pv = w * g
            tot = pv.sum() if w.sum() > 0 else 1.0
            value *= tot / w.sum() if w.sum() > 0 else 1.0
            w = pv / tot * w.sum() if w.sum() > 0 else w
            nav[t] = value
    s = pd.Series(nav, index=idx).dropna()
    return s / s.iloc[0]


def period_cagr(nav: pd.Series, a: str, b: str) -> float:
    s = nav.loc[a:b]
    if len(s) < 2:
        return np.nan
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--sizes", default="5,10,20,50")
    p.add_argument("--random-runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="sp500_top5_monthly")
    args = p.parse_args(argv)
    sizes = [int(x) for x in args.sizes.split(",")]

    px, member = L.load("prices_long_cache.csv", "membership_long.pkl", args.start, args.end)
    px = px.replace(0.0, np.nan)  # a handful of zero closes in the cache (DEC 2021-2023) would be infinite returns
    eligible = member & px.notna() & px.shift(252).notna()
    ranks = rank_frames(px, eligible)
    mes = [d for d in month_ends(px.index) if d >= pd.Timestamp(args.start) - pd.Timedelta(days=40)]
    mes = [d for d in mes if ranks["long"].loc[d].notna().sum() >= 50]
    print(f"universe {px.shape[1]} tickers, {len(mes)} month-ends {mes[0].date()}..{mes[-1].date()}")

    curves: dict[str, pd.Series] = {}
    for name, fr in ranks.items():
        for n in sizes:
            picks = {d: fr.loc[d].dropna().nlargest(n).index.tolist() for d in mes}
            curves[f"{name}_top{n}"] = run_portfolio(px, picks)
    # controls
    ew_picks = {d: eligible.loc[d][eligible.loc[d]].index.tolist() for d in mes}
    curves["equal_weight_members"] = run_portfolio(px, ew_picks)
    spy = pd.read_csv("spy_long_cache.csv", index_col=0, parse_dates=True).squeeze().sort_index()
    qqq = B.load_series("QQQ")
    first = curves["composite_top5"].index[0]
    for nm, s in (("SPY", spy), ("QQQ", qqq)):
        s = s.reindex(px.index).ffill().loc[first:args.end]
        curves[nm] = s / s.iloc[0]
    eq = pd.DataFrame(curves).loc[args.start:args.end]
    eq = eq / eq.iloc[0]
    eq.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")

    # random controls: pick N members at random each month
    rng = np.random.default_rng(args.seed)
    rand_rows = []
    for n in sizes:
        for k in range(args.random_runs):
            picks = {d: list(rng.choice(ew_picks[d], size=min(n, len(ew_picks[d])), replace=False)) for d in mes}
            nav = run_portfolio(px, picks).loc[args.start:args.end]
            rand_rows.append({"n": n, "run": k, "cagr_pct": round(B.cagr(nav) * 100, 2),
                              "cagr_2020_2024_pct": round(period_cagr(nav, "2020-01-01", args.end) * 100, 2),
                              "max_dd_pct": round((nav / nav.cummax() - 1).min() * 100, 1)})
    rand = pd.DataFrame(rand_rows)
    rand.to_csv(f"{args.out}_random.csv", index=False, encoding="utf-8-sig")

    # 2025-2026 on today's constituents
    px2 = pd.read_csv("prices_cache.csv", index_col=0, parse_dates=True).sort_index().replace(0.0, np.nan)
    elig2 = px2.notna() & px2.shift(252).notna()
    ranks2 = rank_frames(px2, elig2)
    mes2 = [d for d in month_ends(px2.index) if d >= pd.Timestamp("2024-12-01") and ranks2["long"].loc[d].notna().sum() >= 50]
    short_curves = {}
    for name, fr in ranks2.items():
        for n in sizes:
            picks = {d: fr.loc[d].dropna().nlargest(n).index.tolist() for d in mes2}
            short_curves[f"{name}_top{n}"] = run_portfolio(px2, picks)
    short_curves["equal_weight_members"] = run_portfolio(px2, {d: elig2.loc[d][elig2.loc[d]].index.tolist() for d in mes2})
    f2 = short_curves["composite_top5"].index[0]
    for nm, s in (("SPY", B.load_series("SPY")), ("QQQ", qqq)):
        s = s.reindex(px2.index).ffill().loc[f2:"2026-10-16"]
        short_curves[nm] = s / s.iloc[0]
    eq2 = pd.DataFrame(short_curves).dropna(how="all")
    eq2 = eq2 / eq2.iloc[0]
    eq2.to_csv(f"{args.out}_equity_2025_2026.csv", encoding="utf-8-sig")

    rows = []
    for c in eq.columns:
        m = L.metrics_block(eq[c])
        rows.append({"strategy": c, **m,
                     "cagr_2000_2009_pct": round(period_cagr(eq[c], "2000-01-01", "2009-12-31") * 100, 2),
                     "cagr_2010_2019_pct": round(period_cagr(eq[c], "2010-01-01", "2019-12-31") * 100, 2),
                     "cagr_2020_2024_pct": round(period_cagr(eq[c], "2020-01-01", "2024-12-31") * 100, 2),
                     "cagr_2025_2026_pct": round(B.cagr(eq2[c]) * 100, 2) if c in eq2 else np.nan,
                     "max_dd_2025_2026_pct": round((eq2[c] / eq2[c].cummax() - 1).min() * 100, 1) if c in eq2 else np.nan})
    summary = pd.DataFrame(rows)
    for n in sizes:
        rn = rand[rand.n == n]
        summary.loc[summary.strategy.str.endswith(f"_top{n}"), "random_cagr_p10"] = round(rn.cagr_pct.quantile(0.1), 2)
        summary.loc[summary.strategy.str.endswith(f"_top{n}"), "random_cagr_median"] = round(rn.cagr_pct.median(), 2)
        summary.loc[summary.strategy.str.endswith(f"_top{n}"), "random_cagr_p90"] = round(rn.cagr_pct.quantile(0.9), 2)
        summary.loc[summary.strategy.str.endswith(f"_top{n}"), "random_2020_2024_median"] = round(rn.cagr_2020_2024_pct.median(), 2)
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    yearly = (eq.resample("YE").last() / eq.resample("YE").first() - 1) * 100
    yearly.index = yearly.index.year
    yearly.round(1).to_csv(f"{args.out}_yearly.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    cols = ["strategy", "cagr_pct", "ann_vol_pct", "sharpe", "max_dd_pct", "worst_year_pct", "cagr_2000_2009_pct", "cagr_2010_2019_pct", "cagr_2020_2024_pct",
            "cagr_2025_2026_pct", "random_cagr_median", "random_cagr_p90"]
    print(summary[cols].to_string(index=False))
    print(yearly.loc[2019:, [c for c in yearly.columns if c.endswith("top5") or c in ("SPY", "QQQ", "equal_weight_members")]].round(1).to_string())
    chart(eq, summary, sizes, "docs/top5_monthly.png")
    print(f"wrote {args.out}_summary.csv, _yearly.csv, _equity.csv, _equity_2025_2026.csv, _random.csv, docs/top5_monthly.png")
    return 0


def chart(eq: pd.DataFrame, summary: pd.DataFrame, sizes: list[int], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    labels = {"short_top5": "短期 21 日前 5", "medium_top5": "中期 63 日前 5", "long_top5": "長期 252 日前 5", "12_1_top5": "12-1 前 5",
              "composite_top5": "三階段合計前 5", "equal_weight_members": "成員等權", "SPY": "SPY", "QQQ": "QQQ"}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [2.2, 1]})
    series = ["composite_top5", "12_1_top5", "long_top5", "medium_top5", "short_top5", "equal_weight_members", "SPY", "QQQ"]
    colors = pal + ["#555555", "#999999"]
    for c, col in zip(series, colors):
        s = eq[c]
        ax.plot(s.index, s, color=col, lw=1.4 if "top5" in c else 1.1, ls="-" if "top5" in c else "--")
    ax.set_yscale("log")
    ax.set_title("每月換成動能最高 5 隻（時點成份股，2000 至 2024）：五種動能定義對基準", loc="left", fontsize=13)
    ax.set_ylabel("淨值（倍，對數座標）")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ends = [(eq[c].iloc[-1], c, col) for c, col in zip(series, colors)]
    ends.sort(key=lambda t: t[0])
    ys = []
    for v, c, col in ends:
        y = math.log(v)
        if ys and y - ys[-1] < 0.12:
            y = ys[-1] + 0.12
        ys.append(y)
        ax.annotate(f"{labels[c]}  {v:.1f}x", (eq.index[-1], math.exp(y)), xytext=(6, 0), textcoords="offset points", color=col, fontsize=9, va="center")
    ax.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=365 * 3))
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}x" for t in ticks])
    # bottom: CAGR by ranking x N
    rankings = ["composite", "12_1", "long", "medium", "short"]
    names = {"composite": "三階段合計", "12_1": "12-1", "long": "長期 252 日", "medium": "中期 63 日", "short": "短期 21 日"}
    width = 0.8 / len(sizes)
    x = np.arange(len(rankings))
    for j, n in enumerate(sizes):
        vals = [summary.loc[summary.strategy == f"{r}_top{n}", "cagr_pct"].iloc[0] for r in rankings]
        bars = ax2.bar(x + (j - (len(sizes) - 1) / 2) * width, vals, width, color=pal[j], label=f"前 {n} 隻")
        for b, v in zip(bars, vals):
            ax2.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -10), textcoords="offset points", ha="center", fontsize=8)
    for nm, col, ls in (("SPY", "#555555", "--"), ("equal_weight_members", "#999999", ":")):
        v = summary.loc[summary.strategy == nm, "cagr_pct"].iloc[0]
        ax2.axhline(v, color=col, ls=ls, lw=1)
        ax2.annotate(f"{labels[nm]} {v:.1f}%", (len(rankings) - 0.5, v), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=9, color=col)
    ax2.set_xticks(x); ax2.set_xticklabels([names[r] for r in rankings])
    ax2.set_ylabel("2000 至 2024 年 CAGR（%）")
    ax2.set_title("持倉數從 5 放寬到 50：集中度對報酬的影響", loc="left", fontsize=12)
    ax2.legend(frameon=False, ncol=len(sizes), loc="upper left")
    ax2.grid(axis="y", alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
