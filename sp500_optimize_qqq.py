#!/usr/bin/env python3
"""
Second-generation cash-allowed rules: can finer-grained exposure control beat QQQ more often?

The first generation (sp500_beat_qqq.py) was all-in / all-out and lost to QQQ in 82-88% of
random entries because every exit in the 2009-2024 bull cost a few points. This script tests
rules that scale exposure instead of switching it, and rules with fewer exits:

  voltarget_T_W      QQQ weight = min(1, T / realised W-day volatility)        (no leverage)
  partial_sma200_X   QQQ weight 1 above SMA200, X below (instead of 0)
  vote_ensemble      weight = share of 5 trend rules that are "on" (SMA100/200/250, 6m/12m momentum)
  asym_trend         exit only when close < SMA200 AND SMA200 falling; re-enter when close > SMA50
  breadth_regime     hold QQQ while > 40% of S&P 500 members are above their own 200-day SMA
                     (point-in-time members 2000-2024, today's constituents 2025-2026)
  gem_4asset_N       best of QQQ / SPY / TLT / GLD by N-day return if above T-bills, else cash
  blend_50_50        50% QQQ buy-and-hold + 50% qqq_sma_200_monthly, rebalanced daily
  voltarget+sma200m  vol-target weight, and 0 when QQQ < SMA200 at month end
All rules: signal at close t-1, executed at close t, 0.10% per side, cash earns T-bills.

Evaluation: same entry-window grid (3/5/10 years) and the same 5,000 paired random draws
(1-20 year holds, seed 42) as sections 9.2 and 9.5.

Outputs: <stem>_summary.csv, <stem>_entry_windows.csv, <stem>_mc_summary.csv, <stem>_mc_by_hold.csv,
         <stem>_equity.csv, docs/optimize_qqq.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import sp500_beat_qqq as B
import sp500_random_entry_mc as MC

COST = 0.001


def load_etf(t: str) -> pd.Series:
    return pd.read_csv(f"{t.lower()}_full_cache.csv", index_col=0, parse_dates=True).squeeze().sort_index()


def breadth_series(idx: pd.DatetimeIndex) -> pd.Series:
    """% of S&P 500 members above their own 200-day SMA; PIT members to 2024, today's list after."""
    import sp500_backtest_long as L

    px, member = L.load("prices_long_cache.csv", "membership_long.pkl", "1999-01-01", "2024-12-31")
    above = (px > px.rolling(200).mean()) & member & px.notna()
    b1 = above.sum(axis=1) / (member & px.notna()).sum(axis=1).replace(0, np.nan)
    px2 = pd.read_csv("prices_cache.csv", index_col=0, parse_dates=True)
    above2 = (px2 > px2.rolling(200).mean()) & px2.notna()
    b2 = above2.sum(axis=1) / px2.notna().sum(axis=1).replace(0, np.nan)
    b = pd.concat([b1, b2.loc[b1.index[-1] + pd.Timedelta(days=1):]])
    return b.reindex(idx).ffill()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2000-03-01")
    p.add_argument("--end", default="2026-10-16")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="sp500_optimize_qqq")
    p.add_argument("--chart-only", action="store_true", help="redraw docs/optimize_qqq.png from the saved CSVs")
    args = p.parse_args(argv)
    if args.chart_only:
        eq = pd.read_csv(f"{args.out}_equity.csv", index_col=0, parse_dates=True)
        chart(eq, pd.read_csv(f"{args.out}_summary.csv"), "docs/optimize_qqq.png")
        print("wrote docs/optimize_qqq.png")
        return 0

    qqq, spy, tlt, gld, irx = load_etf("QQQ"), load_etf("SPY"), load_etf("TLT"), load_etf("GLD"), B.load_series("IRX")
    idx = qqq.loc[: args.end].index
    etf = pd.DataFrame({"QQQ": qqq, "SPY": spy, "TLT": tlt, "GLD": gld}).reindex(idx).ffill()
    cash_daily = (irx.reindex(idx).ffill() / 100 / 252).fillna(0.0)
    tb_cum = (1 + cash_daily).cumprod()
    start_ts = pd.Timestamp(args.start)
    zero = pd.DataFrame(0.0, index=idx, columns=etf.columns)
    strategies, desc, cashpct = {}, {}, {}

    def add(name, weights, d):
        nav = B.etf_timing(etf, weights, cash_daily, COST).loc[start_ts:]
        strategies[name] = nav / nav.iloc[0]
        desc[name] = d
        w = weights.loc[start_ts:]
        cashpct[name] = round(float((1 - w.sum(axis=1)).mean() * 100), 1)

    q = etf["QQQ"]
    sma50, sma100, sma200, sma250 = (q.rolling(n).mean() for n in (50, 100, 200, 250))
    month_end = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("last") == pd.Series(idx, index=idx)
    add("QQQ_buy_hold", zero.assign(QQQ=1.0), "buy and hold QQQ")
    add("SPY_buy_hold", zero.assign(SPY=1.0), "buy and hold SPY")
    sig_m = (q > sma200).where(month_end.to_numpy()).ffill().fillna(False)
    add("qqq_sma_200_monthly", zero.assign(QQQ=sig_m.astype(float)), "reference: QQQ > SMA200 at month end, else cash")

    # 1. volatility targeting (no leverage)
    for T in (0.15, 0.20, 0.25, 0.30):
        for W in (21, 63):
            vol = q.pct_change().rolling(W).std() * math.sqrt(252)
            w = (T / vol).clip(upper=1.0).fillna(0.0)
            add(f"voltarget_{int(T*100)}_{W}", zero.assign(QQQ=w), f"QQQ weight = min(1, {int(T*100)}% / {W}-day realised vol)")
    # 2. partial de-risking
    for X in (0.25, 0.5):
        w = pd.Series(np.where(q > sma200, 1.0, X), index=idx)
        add(f"partial_sma200_{int(X*100)}", zero.assign(QQQ=w), f"QQQ weight 1 above SMA200, {X:.2f} below")
        wm = pd.Series(np.where(sig_m, 1.0, X), index=idx)
        add(f"partial_sma200m_{int(X*100)}", zero.assign(QQQ=wm), f"QQQ weight 1 above SMA200 (month end), {X:.2f} below")
    # 3. ensemble vote
    votes = pd.concat([(q > sma100), (q > sma200), (q > sma250),
                       (q / q.shift(126) - 1 > tb_cum / tb_cum.shift(126) - 1),
                       (q / q.shift(252) - 1 > tb_cum / tb_cum.shift(252) - 1)], axis=1).astype(float)
    add("vote_ensemble", zero.assign(QQQ=votes.mean(axis=1).fillna(0.0)), "QQQ weight = share of 5 trend rules that are on")
    add("vote_ensemble_monthly", zero.assign(QQQ=votes.mean(axis=1).where(month_end.to_numpy()).ffill().fillna(0.0)), "vote ensemble evaluated at month end")
    # 4. asymmetric trend: slow exit, fast re-entry
    on = np.zeros(len(idx)); state = 1
    c, m200, m200_20, m50 = q.to_numpy(), sma200.to_numpy(), sma200.shift(20).to_numpy(), sma50.to_numpy()
    for t in range(len(idx)):
        if np.isnan(m200[t]) or np.isnan(m200_20[t]):
            on[t] = 0; continue
        if state == 1 and c[t] < m200[t] and m200[t] < m200_20[t]:
            state = 0
        elif state == 0 and c[t] > m50[t]:
            state = 1
        on[t] = state
    add("asym_trend", zero.assign(QQQ=on), "exit when close < SMA200 and SMA200 falling; re-enter when close > SMA50")
    # 5. breadth regime
    try:
        br = breadth_series(idx)
        for thr in (0.3, 0.4, 0.5):
            add(f"breadth_gt_{int(thr*100)}", zero.assign(QQQ=(br > thr).astype(float)), f"QQQ while > {int(thr*100)}% of S&P 500 members above their 200-day SMA")
        add("breadth_40_or_sma200", zero.assign(QQQ=((br > 0.4) | (q > sma200)).astype(float)), "QQQ while breadth > 40% OR QQQ > SMA200")
        add("breadth_40_and_sma200", zero.assign(QQQ=((br > 0.4) & (q > sma200)).astype(float)), "QQQ while breadth > 40% AND QQQ > SMA200")
    except Exception as e:  # noqa: BLE001
        print(f"[breadth] skipped: {e}", file=sys.stderr)
    # 6. four-asset dual momentum
    for n in (126, 252):
        mom = etf / etf.shift(n) - 1
        tb = tb_cum / tb_cum.shift(n) - 1
        momf = mom.fillna(-np.inf)
        best = momf.idxmax(axis=1)
        bestv = momf.max(axis=1)
        w = pd.DataFrame(0.0, index=idx, columns=etf.columns)
        ok = (bestv > tb).to_numpy() & np.isfinite(bestv.to_numpy())
        for t, b, good in zip(idx, best, ok):
            if good:
                w.at[t, b] = 1.0
        add(f"gem_4asset_{n}", w, f"best of QQQ/SPY/TLT/GLD by {n}-day return if above T-bills, else cash")
    # 7. 50/50 blend of buy-and-hold and timing
    add("blend_50_50", zero.assign(QQQ=0.5 + 0.5 * sig_m.astype(float)), "50% QQQ always + 50% QQQ under the SMA200 monthly rule")
    # 8. vol target + monthly SMA200
    vol63 = q.pct_change().rolling(63).std() * math.sqrt(252)
    add("voltarget_25_63+sma200m", zero.assign(QQQ=((0.25 / vol63).clip(upper=1.0).fillna(0.0) * sig_m.astype(float))), "vol-target 25%/63d, and cash when QQQ < SMA200 at month end")
    add("voltarget_20_63+sma200m", zero.assign(QQQ=((0.20 / vol63).clip(upper=1.0).fillna(0.0) * sig_m.astype(float))), "vol-target 20%/63d, and cash when QQQ < SMA200 at month end")

    eq = pd.DataFrame(strategies).loc[start_ts:]
    eq.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")
    end_common = pd.Timestamp("2024-12-31")
    e24 = eq.loc[:end_common]; e24 = e24 / e24.iloc[0]
    rows = []
    for col in eq.columns:
        m = B.metrics(e24[col]); m.update(strategy=col, description=desc[col], avg_cash_pct=cashpct[col])
        m["cagr_to_2026_pct"] = round(B.cagr(eq[col]) * 100, 2)
        rows.append(m)
    summary = pd.DataFrame(rows)
    ew = B.entry_windows(e24, "QQQ_buy_hold")
    for h in (3, 5, 10):
        s = ew[ew.horizon_years == h].set_index("strategy")
        summary[f"win_vs_QQQ_{h}y"] = summary.strategy.map(s.win_rate_vs_bench_pct)
        summary[f"median_excess_{h}y"] = summary.strategy.map(s.median_excess_cagr_pct)
    # Monte Carlo on the same draws as section 9.5 (same seed, same QQQ index)
    rng = np.random.default_rng(args.seed)
    main_idx = eq["QQQ_buy_hold"].dropna().index
    starts, ends, years = MC.draw_windows(main_idx, args.n, 1, 20, rng)
    gq, cq, _ = MC.window_stats(eq["QQQ_buy_hold"].to_numpy(), starts, ends, years)
    mc_rows, mc_hold = [], []
    buckets = [(1, 2), (3, 5), (6, 10), (11, 15), (16, 20)]
    for col in eq.columns:
        g, cg, md = MC.window_stats(eq[col].to_numpy(), starts, ends, years)
        win = g > gq
        mc_rows.append({"strategy": col, "mc_win_vs_QQQ_pct": round(float(win.mean() * 100), 1), "mc_median_cagr_pct": round(float(np.median(cg)) * 100, 2),
                        "mc_p10_cagr_pct": round(float(np.percentile(cg, 10)) * 100, 2), "mc_worst_cagr_pct": round(float(cg.min()) * 100, 2),
                        "mc_prob_loss_pct": round(float((g < 1).mean() * 100), 1), "mc_median_excess_pct": round(float(np.median(cg - cq)) * 100, 2),
                        "mc_median_terminal_100k": round(float(np.median(100_000 * g)))})
        for lo, hi in buckets:
            mk = (years >= lo) & (years <= hi)
            mc_hold.append({"strategy": col, "hold_years": f"{lo}-{hi}", "win_vs_QQQ_pct": round(float(win[mk].mean() * 100), 1),
                            "median_cagr_pct": round(float(np.median(cg[mk])) * 100, 2), "prob_loss_pct": round(float((g[mk] < 1).mean() * 100), 1)})
    mc = pd.DataFrame(mc_rows)
    summary = summary.merge(mc, on="strategy").sort_values("mc_win_vs_QQQ_pct", ascending=False).reset_index(drop=True)
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    ew.to_csv(f"{args.out}_entry_windows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mc_hold).to_csv(f"{args.out}_mc_by_hold.csv", index=False, encoding="utf-8-sig")
    mc.to_csv(f"{args.out}_mc_summary.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 300)
    cols = ["strategy", "cagr_pct", "max_dd_pct", "sharpe", "worst_year_pct", "avg_cash_pct", "win_vs_QQQ_5y", "win_vs_QQQ_10y", "median_excess_5y",
            "mc_win_vs_QQQ_pct", "mc_median_cagr_pct", "mc_p10_cagr_pct", "mc_worst_cagr_pct", "mc_prob_loss_pct", "mc_median_terminal_100k", "cagr_to_2026_pct"]
    print(f"evaluation 2000-03-01 .. 2024-12-31 for CAGR/DD/windows; Monte Carlo {args.n} draws 1-20y to 2026-10-16 (seed {args.seed})")
    print(summary[cols].to_string(index=False))
    piv = pd.DataFrame(mc_hold).pivot(index="hold_years", columns="strategy", values="win_vs_QQQ_pct").reindex([f"{lo}-{hi}" for lo, hi in buckets])
    print("\nMC win rate vs QQQ by holding bucket (top 8 by overall):")
    print(piv[summary.strategy.head(9).tolist()].to_string())
    chart(eq, summary, f"docs/optimize_qqq.png")
    print(f"\nwrote {args.out}_summary.csv, _entry_windows.csv, _mc_summary.csv, _mc_by_hold.csv, _equity.csv, docs/optimize_qqq.png")
    return 0


def chart(eq: pd.DataFrame, summary: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "WenQuanYi" in f.name or "Noto Sans CJK" in f.name})
    if cjk:
        plt.rcParams["font.family"] = cjk[0]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    s = summary[summary.strategy != "QQQ_buy_hold"].copy()
    x = s["mc_win_vs_QQQ_pct"].to_numpy(); y = s["mc_median_cagr_pct"].to_numpy(); dd = -s["max_dd_pct"].to_numpy()
    fam = s.strategy.str.extract(r"^(voltarget|partial|vote|asym|breadth|gem|blend|qqq_sma|SPY)")[0].fillna("other")
    colors = {"voltarget": "#2a78d6", "partial": "#eb6834", "vote": "#1baf7a", "breadth": "#eda100", "gem": "#e87ba4", "asym": "#008300", "blend": "#6250d6", "qqq_sma": "#e34948", "SPY": "#52514e", "other": "#52514e"}
    for f in colors:
        mk = (fam == f).to_numpy()
        if mk.any():
            ax.scatter(x[mk], y[mk], s=40 + dd[mk] * 3, color=colors[f], alpha=0.85, edgecolor="#fcfcfb", linewidth=1, label=f)
    # merge labels that share a point (e.g. blend_50_50 == partial_sma200m_50 by construction)
    groups: dict[tuple, list] = {}
    for xi, yi, name in zip(x, y, s.strategy):
        groups.setdefault((round(xi, 1), round(yi, 2)), []).append(name)
    placed = []
    for (xi, yi), names in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        dy = 3
        for px_, py_ in placed:  # push the label up if another label sits within 1.5 x-units and 0.35 y-units
            if abs(px_ - xi) < 1.5 and abs(py_ - yi) < 0.35:
                dy += 9
        ax.annotate(" = ".join(names), (xi, yi), xytext=(5, dy), textcoords="offset points", fontsize=7.5, color="#52514e")
        placed.append((xi, yi))
    qq = summary[summary.strategy == "QQQ_buy_hold"].iloc[0]
    ax.axhline(qq.mc_median_cagr_pct, color="#c3c2b7", lw=1, ls="--")
    ax.annotate(f"QQQ median CAGR {qq.mc_median_cagr_pct}%", (1, qq.mc_median_cagr_pct), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color="#52514e")
    ax.axvline(50, color="#c3c2b7", lw=1, ls="--")
    ax.set_xlabel("share of 5,000 random entries (1-20y holds) where the rule beats QQQ, %", color="#52514e", fontsize=9)
    ax.set_ylabel("median CAGR across the draws, %", color="#52514e", fontsize=9)
    ax.set_title("Second-generation cash-allowed rules vs QQQ (marker size = max drawdown 2000-2024)", fontsize=11, loc="left", color="#0b0b0b")
    ax.grid(True, color="#e6e5e1", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, title="family", title_fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    sys.exit(main())
