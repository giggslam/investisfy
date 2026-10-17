#!/usr/bin/env python3
"""
Monte Carlo over random entry dates and random holding periods.

Draws N (entry date, holding period) pairs: holding period uniform over whole years
in [--min-years, --max-years], entry date uniform over the trading days for which the
whole holding period fits inside the equity curve. Every strategy is evaluated on the
SAME draws, so the comparison with QQQ is paired.

Per draw: CAGR of the strategy and of QQQ, excess, terminal value of --capital, max
drawdown inside the window. Aggregates overall and by holding-period bucket and entry
year. Input: the equity curves written by sp500_beat_qqq.py.

Outputs: <stem>_summary.csv, <stem>_by_hold.csv, <stem>_by_entry_year.csv, <stem>_draws.csv,
         docs/random_entry_mc.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

DEFAULT = ["QQQ_buy_hold", "SPY_buy_hold", "qqq_sma_200_monthly", "qqq_sma_250", "qqq_sma_200_dd_15",
           "qqq_absmom_189", "qqq_sma_200_band3", "stocks_rev_short_m10_5pos_63d+regime"]


def draw_windows(index: pd.DatetimeIndex, n: int, min_y: int, max_y: int, rng: np.random.Generator):
    """Return arrays (start_idx, end_idx, years) with end_idx the first trading day >= start + years."""
    T = len(index)
    years = rng.integers(min_y, max_y + 1, size=n)
    starts = np.empty(n, dtype=int)
    ends = np.empty(n, dtype=int)
    last = index[-1]
    for i, y in enumerate(years):
        # latest admissible start: start + y years <= last date
        latest = index.searchsorted(last - pd.DateOffset(years=int(y)), side="right") - 1
        s = rng.integers(0, latest + 1)
        e = index.searchsorted(index[s] + pd.DateOffset(years=int(y)))
        e = min(e, T - 1)
        starts[i], ends[i] = s, e
    return starts, ends, years


def window_stats(nav: np.ndarray, starts, ends, years):
    g = nav[ends] / nav[starts]
    cagr = g ** (1 / years) - 1
    mdd = np.empty(len(starts))
    for i, (s, e) in enumerate(zip(starts, ends)):
        w = nav[s : e + 1]
        mdd[i] = (w / np.maximum.accumulate(w) - 1).min()
    return g, cagr, mdd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--equity", default="sp500_beat_qqq_equity.csv")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--min-years", type=int, default=1)
    p.add_argument("--max-years", type=int, default=20)
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", nargs="*", default=DEFAULT)
    p.add_argument("--out", default="sp500_random_entry_mc")
    args = p.parse_args(argv)

    eq = pd.read_csv(args.equity, index_col=0, parse_dates=True)
    cols = [c for c in args.strategies if c in eq.columns]
    rng = np.random.default_rng(args.seed)
    # ETF curves run to 2026-10; stock curves end 2024-12. Use the common ETF index, and
    # for a strategy with a shorter curve, draws are redrawn on its own index.
    main_idx = eq["QQQ_buy_hold"].dropna().index
    starts, ends, years = draw_windows(main_idx, args.n, args.min_years, args.max_years, rng)
    qqq = eq["QQQ_buy_hold"].reindex(main_idx).to_numpy()
    gq, cq, mq = window_stats(qqq, starts, ends, years)

    draws = pd.DataFrame({"entry_date": main_idx[starts], "exit_date": main_idx[ends], "years": years, "QQQ_cagr": cq, "QQQ_growth": gq, "QQQ_mdd": mq})
    summary, by_hold, by_year = [], [], []
    buckets = [(1, 2), (3, 5), (6, 10), (11, 15), (16, 20)]
    for c in cols:
        s = eq[c].dropna()
        if s.index[-1] < main_idx[-1] - pd.Timedelta(days=10):
            # shorter curve: redraw on its own index with the same seed so draws are comparable in law
            idx_c = s.index
            r2 = np.random.default_rng(args.seed)
            st, en, yr = draw_windows(idx_c, args.n, args.min_years, args.max_years, r2)
            nav = s.to_numpy()
            g, cg, md = window_stats(nav, st, en, yr)
            qn = eq["QQQ_buy_hold"].reindex(idx_c).to_numpy()
            gq2, cq2, _ = window_stats(qn, st, en, yr)
            ex = cg - cq2
            win = g > gq2
            yrs_c, entry_years = yr, idx_c[st].year
            note = f"own window {idx_c[0].date()}..{idx_c[-1].date()}"
        else:
            nav = s.reindex(main_idx).to_numpy()
            g, cg, md = window_stats(nav, starts, ends, years)
            ex = cg - cq
            win = g > gq
            yrs_c, entry_years = years, main_idx[starts].year
            note = f"window {main_idx[0].date()}..{main_idx[-1].date()}"
            draws[f"{c}_cagr"] = cg
            draws[f"{c}_growth"] = g
        tv = args.capital * g
        summary.append({"strategy": c, "draws": args.n, "note": note,
                        "win_vs_QQQ_pct": round(float(win.mean() * 100), 1),
                        "median_cagr_pct": round(float(np.median(cg)) * 100, 2), "p10_cagr_pct": round(float(np.percentile(cg, 10)) * 100, 2),
                        "p90_cagr_pct": round(float(np.percentile(cg, 90)) * 100, 2), "worst_cagr_pct": round(float(cg.min()) * 100, 2),
                        "prob_loss_pct": round(float((g < 1).mean() * 100), 1), "median_excess_vs_QQQ_pct": round(float(np.median(ex)) * 100, 2),
                        "mean_excess_vs_QQQ_pct": round(float(ex.mean()) * 100, 2),
                        "median_terminal_100k": round(float(np.median(tv))), "p10_terminal_100k": round(float(np.percentile(tv, 10))),
                        "worst_terminal_100k": round(float(tv.min())), "median_max_dd_in_window_pct": round(float(np.median(md)) * 100, 1),
                        "worst_max_dd_in_window_pct": round(float(md.min()) * 100, 1)})
        for lo, hi in buckets:
            m = (yrs_c >= lo) & (yrs_c <= hi)
            if m.sum() == 0:
                continue
            by_hold.append({"strategy": c, "hold_years": f"{lo}-{hi}", "draws": int(m.sum()), "win_vs_QQQ_pct": round(float(win[m].mean() * 100), 1),
                            "median_cagr_pct": round(float(np.median(cg[m])) * 100, 2), "p10_cagr_pct": round(float(np.percentile(cg[m], 10)) * 100, 2),
                            "worst_cagr_pct": round(float(cg[m].min()) * 100, 2), "prob_loss_pct": round(float((g[m] < 1).mean() * 100), 1),
                            "median_excess_pct": round(float(np.median(ex[m])) * 100, 2), "median_terminal_100k": round(float(np.median(tv[m])))})
        for y in sorted(set(entry_years)):
            m = entry_years == y
            by_year.append({"strategy": c, "entry_year": int(y), "draws": int(m.sum()), "win_vs_QQQ_pct": round(float(win[m].mean() * 100), 1),
                            "median_cagr_pct": round(float(np.median(cg[m])) * 100, 2), "prob_loss_pct": round(float((g[m] < 1).mean() * 100), 1)})

    summary = pd.DataFrame(summary)
    by_hold = pd.DataFrame(by_hold)
    by_year = pd.DataFrame(by_year)
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    by_hold.to_csv(f"{args.out}_by_hold.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(f"{args.out}_by_entry_year.csv", index=False, encoding="utf-8-sig")
    draws.to_csv(f"{args.out}_draws.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print(f"{args.n} random (entry date, holding period) draws, holding {args.min_years}-{args.max_years} whole years, seed {args.seed}")
    print(f"draw mix: years median {np.median(years):.0f}, entry dates {main_idx[starts].min().date()} .. {main_idx[starts].max().date()}")
    print("\nOverall:")
    print(summary.drop(columns=["note"]).to_string(index=False))
    print("\nBy holding-period bucket:")
    print(by_hold.pivot(index="hold_years", columns="strategy", values="win_vs_QQQ_pct").reindex([f"{lo}-{hi}" for lo, hi in buckets]).to_string())
    print("\nMedian CAGR by bucket:")
    print(by_hold.pivot(index="hold_years", columns="strategy", values="median_cagr_pct").reindex([f"{lo}-{hi}" for lo, hi in buckets]).to_string())
    print("\nProbability of loss by bucket (%):")
    print(by_hold.pivot(index="hold_years", columns="strategy", values="prob_loss_pct").reindex([f"{lo}-{hi}" for lo, hi in buckets]).to_string())
    chart(draws, by_hold, cols, f"docs/random_entry_mc.png")
    print(f"\nwrote {args.out}_summary.csv, _by_hold.csv, _by_entry_year.csv, _draws.csv, docs/random_entry_mc.png")
    return 0


def chart(draws: pd.DataFrame, by_hold: pd.DataFrame, cols: list[str], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "WenQuanYi" in f.name or "Noto Sans CJK" in f.name})
    if cjk:
        plt.rcParams["font.family"] = cjk[0]
    plt.rcParams["axes.unicode_minus"] = False
    palette = {"QQQ_buy_hold": "#2a78d6", "SPY_buy_hold": "#eb6834", "qqq_sma_200_monthly": "#1baf7a", "qqq_sma_200_dd_15": "#eda100", "qqq_sma_250": "#e87ba4"}
    show = [c for c in palette if f"{c}_cagr" in draws.columns or c == "QQQ_buy_hold"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor="#fcfcfb")
    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    yrs = np.arange(1, 21)
    for c in show:
        col = "QQQ_cagr" if c == "QQQ_buy_hold" else f"{c}_cagr"
        med = [np.median(draws.loc[draws.years == y, col]) * 100 for y in yrs]
        p10 = [np.percentile(draws.loc[draws.years == y, col], 10) * 100 for y in yrs]
        ax.plot(yrs, med, color=palette[c], lw=2 if c == "QQQ_buy_hold" else 1.6, label=c)
        ax.plot(yrs, p10, color=palette[c], lw=1, ls="--", alpha=0.8)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_title("CAGR by holding period: median (solid) and 10th percentile (dashed), %", fontsize=11, loc="left", color="#0b0b0b")
    ax.set_xlabel("holding period, years", color="#52514e", fontsize=9)
    ax.set_ylabel("CAGR, %", color="#52514e", fontsize=9)
    ax2 = axes[1]
    ax2.set_facecolor("#fcfcfb")
    for c in show:
        if c == "QQQ_buy_hold":
            continue
        col = f"{c}_cagr"
        wr = [(draws.loc[draws.years == y, f"{c}_growth"] > draws.loc[draws.years == y, "QQQ_growth"]).mean() * 100 for y in yrs]
        ax2.plot(yrs, wr, color=palette[c], lw=1.8, label=c, marker="o", ms=3)
    ax2.axhline(50, color="#c3c2b7", lw=1)
    ax2.set_ylim(0, 100)
    ax2.set_title("Share of random entries where the rule beats QQQ, %", fontsize=11, loc="left", color="#0b0b0b")
    ax2.set_xlabel("holding period, years", color="#52514e", fontsize=9)
    for a in axes:
        a.grid(True, axis="y", color="#e6e5e1", lw=0.8)
        a.grid(False, axis="x")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#c3c2b7")
        a.tick_params(colors="#52514e", labelsize=9)
        a.set_xticks(range(1, 21, 2))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{len(draws):,} random entries, 1-20 year holds, 2000-2026", fontsize=13, color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    sys.exit(main())
