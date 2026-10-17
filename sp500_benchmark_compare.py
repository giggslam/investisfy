#!/usr/bin/env python3
"""
Compare the back-tested strategies with SPY and QQQ.

Reads the equity curves written by sp500_backtest_long.py (2000-2024) and
sp500_backtest.py (2025-2026), downloads SPY and QQQ adjusted closes (cached), and
writes metrics, yearly returns, sub-period CAGR and a two-panel chart.

Outputs:
    benchmark_compare_long_equity.csv / _metrics.csv / _yearly.csv / _periods.csv
    benchmark_compare_short_equity.csv / _metrics.csv / _yearly.csv
    docs/benchmark_comparison.png
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

LONG_COLS = {"A_max_return_5pos_63d_stop12": "A 最大報酬版", "B_robust_4pos_126cap_shortneg": "B 穩健版",
             "C_5pos_63d_nostop": "C 無停損版", "equal_weight_members": "成員等權"}
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "EW": "#eda100", "SPY": "#e87ba4", "QQQ": "#008300"}
KEYS = {"A 最大報酬版": "A", "B 穩健版": "B", "C 無停損版": "C", "成員等權": "EW", "成份股等權": "EW", "SPY": "SPY", "QQQ": "QQQ"}
LEGEND = {"A": "A max-return (5 pos, 63d, 12% stop)", "B": "B robust (4 pos, exit on short score < 0)",
          "C": "C no-stop (5 pos, 63d)", "EW": "Equal-weight members", "SPY": "SPY", "QQQ": "QQQ"}


def load_etf(ticker: str, start: str = "1999-03-01") -> pd.Series:
    path = f"{ticker.lower()}_full_cache.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    import yfinance as yf

    s = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.to_csv(path)
    return s


def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1
    yearly = nav.resample("YE").last() / nav.resample("YE").first() - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    return dict(final_x=round(nav.iloc[-1] / nav.iloc[0], 2), cagr_pct=round(cagr * 100, 2),
                ann_vol_pct=round(r.std() * math.sqrt(252) * 100, 1), sharpe=round(r.mean() / r.std() * math.sqrt(252), 2),
                max_dd_pct=round(dd.min() * 100, 1), calmar=round(cagr / abs(dd.min()), 2),
                worst_year_pct=round(yearly.min() * 100, 1), best_year_pct=round(yearly.max() * 100, 1),
                pct_years_pos=round((yearly > 0).mean() * 100, 0))


def with_benchmarks(eq: pd.DataFrame, spy: pd.Series, qqq: pd.Series) -> pd.DataFrame:
    out = eq.copy()
    for name, s in (("SPY", spy), ("QQQ", qqq)):
        b = s.reindex(eq.index).ffill()
        out[name] = (b / b.iloc[0]).to_numpy()
    return out


def chart(L: pd.DataFrame, S: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates, font_manager

    cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "WenQuanYi" in f.name or "Noto Sans CJK" in f.name})
    if cjk:
        plt.rcParams["font.family"] = cjk[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.6, 1]}, facecolor="#fcfcfb")
    panels = ((axes[0], L, "2000-01-03 to 2024-12-31 (log scale)", True, 0.10), (axes[1], S, "2025-01-02 to 2026-10-16", False, 0.22))
    ends_all = {}
    for ax, df, title, log, pad in panels:
        ax.set_facecolor("#fcfcfb")
        ends = []
        for col in df.columns:
            key = KEYS[col]
            s = df[col].dropna()
            ax.plot(s.index, s.values, color=COLORS[key], lw=2 if key in ("A", "B", "C") else 1.6, label=LEGEND[key])
            ends.append((key, s.index[-1], float(s.iloc[-1])))
        if log:
            ax.set_yscale("log")
        x0, x1 = mdates.date2num(df.index[0]), mdates.date2num(df.index[-1])
        ax.set_xlim(x0 - (x1 - x0) * 0.02, x1 + (x1 - x0) * pad)
        ax.set_title(title, fontsize=11, color="#0b0b0b", loc="left")
        ax.grid(True, axis="y", color="#e6e5e1", lw=0.8)
        ax.grid(False, axis="x")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", labelsize=9)
        ax.set_ylabel("Growth of 1 unit", color="#52514e", fontsize=9)
        ends_all[ax] = ends
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Double +10 momentum strategies vs SPY and QQQ", fontsize=13, color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.canvas.draw()
    for ax, ends in ends_all.items():  # direct labels at line ends, pushed apart when they collide
        pts = sorted(ends, key=lambda e: e[2])
        disp = [ax.transData.transform((mdates.date2num(d), v))[1] for _, d, v in pts]
        orig = list(disp)
        for i in range(1, len(disp)):
            if disp[i] - disp[i - 1] < 14:
                disp[i] = disp[i - 1] + 14
        for (key, d, v), y0, y1 in zip(pts, orig, disp):
            ax.annotate(f"{key} {v:.2f}x", xy=(d, v), xytext=(6, (y1 - y0) * 72 / fig.dpi), textcoords="offset points",
                        va="center", fontsize=9, color="#52514e")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main() -> int:
    spy, qqq = load_etf("SPY"), load_etf("QQQ")
    # long period
    eq = pd.read_csv("sp500_backtest_long_equity.csv", index_col=0, parse_dates=True)
    L = with_benchmarks(eq[list(LONG_COLS)].rename(columns=LONG_COLS), spy, qqq)
    L.to_csv("benchmark_compare_long_equity.csv", encoding="utf-8-sig")
    m = pd.DataFrame({c: metrics(L[c]) for c in L.columns}).T
    m.index.name = "series"
    m.to_csv("benchmark_compare_long_metrics.csv", encoding="utf-8-sig")
    yl = ((L.resample("YE").last() / L.resample("YE").first() - 1) * 100).round(1)
    yl.index = yl.index.year
    yl.to_csv("benchmark_compare_long_yearly.csv", encoding="utf-8-sig")
    rows = []
    for lo, hi in ((2000, 2002), (2003, 2007), (2008, 2009), (2010, 2019), (2020, 2024), (2000, 2024)):
        e = L.loc[f"{lo}-01-01":f"{hi}-12-31"]
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        rows.append({"period": f"{lo}-{hi}", **{c: round(((e[c].iloc[-1] / e[c].iloc[0]) ** (1 / yrs) - 1) * 100, 1) for c in L.columns}})
    pd.DataFrame(rows).to_csv("benchmark_compare_long_periods.csv", index=False, encoding="utf-8-sig")
    # short period
    a = pd.read_csv("sp500_backtest_maxreturn_best_equity.csv", index_col=0, parse_dates=True)
    b = pd.read_csv("sp500_backtest_robust_best_equity.csv", index_col=0, parse_dates=True)
    S = with_benchmarks(pd.concat([a["strategy"].rename("A 最大報酬版"), b["strategy"].rename("B 穩健版"),
                                   a["equal_weight_universe"].rename("成份股等權")], axis=1), spy, qqq)
    S.to_csv("benchmark_compare_short_equity.csv", encoding="utf-8-sig")
    ms = pd.DataFrame({c: metrics(S[c]) for c in S.columns}).T
    ms.index.name = "series"
    ms.to_csv("benchmark_compare_short_metrics.csv", encoding="utf-8-sig")
    ys = ((S.resample("YE").last() / S.resample("YE").first() - 1) * 100).round(1)
    ys.index = ys.index.year
    ys.to_csv("benchmark_compare_short_yearly.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("2000-2024\n", m.to_string(), "\n\n2025-2026\n", ms.to_string())
    chart(L, S, "docs/benchmark_comparison.png")
    print("\nwrote benchmark_compare_*.csv and docs/benchmark_comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
