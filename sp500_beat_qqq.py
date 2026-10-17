#!/usr/bin/env python3
"""
Can a strategy that is allowed to sit in cash beat QQQ from ANY entry date?

Capital example: 100,000. A signal seen at close t-1 is executed at close t. Cash earns
the 13-week T-bill yield (^IRX). Cost 0.10% per side on every trade.

Strategy families (all may hold cash):
  qqq_sma_N          hold QQQ while QQQ close > N-day simple moving average, else cash
  qqq_sma_N_band     same with a 3% band (buy above SMA*1.03, sell below SMA*0.97) to cut whipsaws
  qqq_absmom_N       hold QQQ while its N-day return > T-bill return over the same window, else cash
  dual_mom_N         hold the better of SPY / QQQ by N-day return if that return beats T-bills, else cash
  qqq_sma_N_dd_X     SMA filter plus a drawdown exit: cash when QQQ is X% below its 252-day high
  stocks_<rule>+regime   the concentrated stock strategies from sp500_backtest.py gated by the
                     QQQ 200-day SMA regime (exit everything and stay in cash when risk-off)
Benchmarks: QQQ and SPY buy-and-hold.

"Any entry date" test: for every start date on a 21-trading-day grid, the strategy's
and QQQ's growth over the next 3, 5 and 10 years; the share of start dates where the
strategy wins, and the worst and median excess CAGR.

Outputs: <stem>_summary.csv, <stem>_entry_windows.csv, <stem>_yearly.csv, <stem>_equity.csv,
         <stem>_examples_100k.csv, docs/beat_qqq.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import sp500_backtest as bt
import sp500_backtest_long as L
import sp500_strategy_search as S

COST = 0.001


def load_series(name: str) -> pd.Series:
    path = {"QQQ": "qqq_full_cache.csv", "SPY": "spy_full_cache.csv", "IRX": "irx_cache.csv"}[name]
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    return s.sort_index()


def etf_timing(prices: pd.DataFrame, weights: pd.DataFrame, cash_daily: pd.Series, cost: float = COST) -> pd.Series:
    """
    Daily-rebalanced ETF/cash allocation. weights[t] (decided from data up to t-1) is the
    target weight per ETF at close t; the remainder is cash. Returns NAV series.
    weights may hold at most one ETF at 1.0 or all zeros (cash), so turnover = sum |dw|.
    """
    w = weights.shift(1).fillna(0.0)  # decided at t-1, applied from close t-1 -> t
    rets = prices.pct_change().fillna(0.0)
    port = (w * rets).sum(axis=1) + (1 - w.sum(axis=1)) * cash_daily.reindex(prices.index).fillna(0.0)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    port = port - turnover * cost
    return (1 + port).cumprod()


def cagr(nav: pd.Series) -> float:
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1


def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1
    yearly = nav.resample("YE").last() / nav.resample("YE").first() - 1
    c = cagr(nav)
    return dict(final_x=round(nav.iloc[-1] / nav.iloc[0], 2), cagr_pct=round(c * 100, 2), ann_vol_pct=round(r.std() * math.sqrt(252) * 100, 1),
                sharpe=round(r.mean() / r.std() * math.sqrt(252), 2), max_dd_pct=round(dd.min() * 100, 1), calmar=round(c / abs(dd.min()), 2),
                worst_year_pct=round(yearly.min() * 100, 1), pct_years_pos=round((yearly > 0).mean() * 100, 0),
                time_in_cash_pct=np.nan)


def entry_windows(eq: pd.DataFrame, bench: str, horizons=(3, 5, 10), step: int = 21) -> pd.DataFrame:
    rows = []
    for col in eq.columns:
        if col == bench:
            continue
        for h in horizons:
            n = int(h * 252)
            idx = np.arange(0, len(eq) - n, step)
            if len(idx) == 0:
                continue
            s_end = eq[col].to_numpy()[idx + n] / eq[col].to_numpy()[idx]
            b_end = eq[bench].to_numpy()[idx + n] / eq[bench].to_numpy()[idx]
            s_c = s_end ** (1 / h) - 1
            b_c = b_end ** (1 / h) - 1
            ex = (s_c - b_c) * 100
            rows.append({"strategy": col, "horizon_years": h, "n_entry_dates": len(idx), "win_rate_vs_bench_pct": round((s_end > b_end).mean() * 100, 1),
                         "median_excess_cagr_pct": round(float(np.median(ex)), 2), "worst_excess_cagr_pct": round(float(ex.min()), 2),
                         "best_excess_cagr_pct": round(float(ex.max()), 2), "median_strategy_cagr_pct": round(float(np.median(s_c)) * 100, 2),
                         "worst_strategy_cagr_pct": round(float(s_c.min()) * 100, 2), "pct_windows_strategy_negative": round((s_end < 1).mean() * 100, 1),
                         "pct_windows_bench_negative": round((b_end < 1).mean() * 100, 1)})
    return pd.DataFrame(rows)


def chart(stem: str, picks: list[str], path: str = "docs/beat_qqq.png") -> None:
    """Two panels: growth of 100k (log) for QQQ, SPY and the picked rules; 5-year entry-window CAGR by entry date."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates, font_manager

    cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "WenQuanYi" in f.name or "Noto Sans CJK" in f.name})
    if cjk:
        plt.rcParams["font.family"] = cjk[0]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    eq = pd.read_csv(f"{stem}_equity.csv", index_col=0, parse_dates=True).loc[: "2024-12-31"]
    series = ["QQQ_buy_hold", "SPY_buy_hold"] + [c for c in picks if c in eq.columns and c not in ("QQQ_buy_hold", "SPY_buy_hold")]
    series = series[:6]
    colors = dict(zip(series, palette))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.4, 1]}, facecolor="#fcfcfb")
    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    ends = []
    for c in series:
        v = eq[c].dropna() / eq[c].dropna().iloc[0] * 100_000
        ax.plot(v.index, v.values, color=colors[c], lw=2 if c in ("QQQ_buy_hold",) else 1.6, label=c)
        ends.append((c, v.index[-1], float(v.iloc[-1])))
    ax.set_yscale("log")
    x0, x1 = mdates.date2num(eq.index[0]), mdates.date2num(eq.index[-1])
    ax.set_xlim(x0 - (x1 - x0) * 0.02, x1 + (x1 - x0) * 0.16)
    ax.set_title("100,000 invested 2000-03-01, value to 2024-12-31 (log scale)", fontsize=11, loc="left", color="#0b0b0b")
    ax.set_ylabel("Portfolio value", color="#52514e", fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    # right: 5-year CAGR by entry date for QQQ vs picks
    ax2 = axes[1]
    ax2.set_facecolor("#fcfcfb")
    n = 5 * 252
    idx = np.arange(0, len(eq) - n, 21)
    for c in series:
        v = eq[c].to_numpy()
        cg = (v[idx + n] / v[idx]) ** (1 / 5) - 1
        ax2.plot(eq.index[idx], cg * 100, color=colors[c], lw=2 if c == "QQQ_buy_hold" else 1.4, label=c)
    ax2.axhline(0, color="#c3c2b7", lw=1)
    ax2.set_title("5-year CAGR by entry date (%)", fontsize=11, loc="left", color="#0b0b0b")
    ax2.set_ylabel("CAGR over the next 5 years, %", color="#52514e", fontsize=9)
    for a in axes:
        a.grid(True, axis="y", color="#e6e5e1", lw=0.8)
        a.grid(False, axis="x")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#c3c2b7")
        a.tick_params(colors="#52514e", labelsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cash-allowed timing rules vs QQQ buy-and-hold", fontsize=13, color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.canvas.draw()
    pts = sorted(ends, key=lambda e: e[2])
    disp = [ax.transData.transform((mdates.date2num(d), v))[1] for _, d, v in pts]
    orig = list(disp)
    for i in range(1, len(disp)):
        if disp[i] - disp[i - 1] < 14:
            disp[i] = disp[i - 1] + 14
    for (c, d, v), y0, y1 in zip(pts, orig, disp):
        ax.annotate(f"{c} {v/1000:,.0f}k", xy=(d, v), xytext=(6, (y1 - y0) * 72 / fig.dpi), textcoords="offset points", va="center", fontsize=8.5, color="#52514e")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2000-03-01")
    p.add_argument("--end", default="2026-10-16")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--out", default="sp500_beat_qqq")
    p.add_argument("--no-stocks", action="store_true", help="skip the stock strategies (faster)")
    p.add_argument("--chart-only", nargs="*", help="draw docs/beat_qqq.png from saved outputs for the given strategies")
    args = p.parse_args(argv)
    if args.chart_only is not None:
        chart(args.out, args.chart_only)
        print("wrote docs/beat_qqq.png")
        return 0

    qqq, spy, irx = load_series("QQQ"), load_series("SPY"), load_series("IRX")
    idx = qqq.loc[: args.end].index
    etf = pd.DataFrame({"QQQ": qqq, "SPY": spy}).reindex(idx).ffill()
    cash_daily = (irx.reindex(idx).ffill() / 100 / 252).fillna(0.0)
    tb_cum = (1 + cash_daily).cumprod()
    start_ts = pd.Timestamp(args.start)

    strategies: dict[str, pd.Series] = {}
    desc: dict[str, str] = {}

    def add(name, weights, d):
        nav = etf_timing(etf, weights, cash_daily)
        nav = nav.loc[start_ts:]
        strategies[name] = nav / nav.iloc[0]
        desc[name] = d
        w = weights.loc[start_ts:]
        strategies[name].attrs["cash_pct"] = round(float((w.sum(axis=1) == 0).mean() * 100), 1)

    zero = pd.DataFrame(0.0, index=idx, columns=["QQQ", "SPY"])
    # benchmarks
    add("QQQ_buy_hold", zero.assign(QQQ=1.0), "buy and hold QQQ")
    add("SPY_buy_hold", zero.assign(SPY=1.0), "buy and hold SPY")
    # SMA filters
    for n in (50, 100, 150, 200, 250):
        sma = etf["QQQ"].rolling(n).mean()
        add(f"qqq_sma_{n}", zero.assign(QQQ=(etf["QQQ"] > sma).astype(float)), f"QQQ when close > SMA{n}, else cash")
        # band version with hysteresis
        on = np.zeros(len(idx)); state = 0
        c = etf["QQQ"].to_numpy(); m = sma.to_numpy()
        for t in range(len(idx)):
            if np.isnan(m[t]):
                on[t] = 0; continue
            if state == 0 and c[t] > m[t] * 1.03: state = 1
            elif state == 1 and c[t] < m[t] * 0.97: state = 0
            on[t] = state
        add(f"qqq_sma_{n}_band3", zero.assign(QQQ=on), f"QQQ when close > SMA{n}*1.03, cash when < SMA{n}*0.97")
    # absolute momentum vs T-bills
    for n in (63, 126, 189, 252):
        mom = etf["QQQ"] / etf["QQQ"].shift(n) - 1
        tb = tb_cum / tb_cum.shift(n) - 1
        add(f"qqq_absmom_{n}", zero.assign(QQQ=(mom > tb).astype(float)), f"QQQ when its {n}-day return beats T-bills, else cash")
    # dual momentum SPY / QQQ / cash
    for n in (126, 252):
        mq = etf["QQQ"] / etf["QQQ"].shift(n) - 1
        ms = etf["SPY"] / etf["SPY"].shift(n) - 1
        tb = tb_cum / tb_cum.shift(n) - 1
        best_q = (mq >= ms) & (mq > tb)
        best_s = (ms > mq) & (ms > tb)
        add(f"dual_mom_{n}", zero.assign(QQQ=best_q.astype(float), SPY=best_s.astype(float)), f"best of SPY/QQQ by {n}-day return if it beats T-bills, else cash")
    # SMA200 + drawdown exit
    sma200 = etf["QQQ"].rolling(200).mean()
    hi252 = etf["QQQ"].rolling(252).max()
    for x in (0.15, 0.20):
        on = ((etf["QQQ"] > sma200) & (etf["QQQ"] / hi252 - 1 > -x)).astype(float)
        add(f"qqq_sma_200_dd_{int(x*100)}", zero.assign(QQQ=on), f"QQQ when > SMA200 and within {int(x*100)}% of 252-day high, else cash")
    # classic monthly-evaluated rules: decide only at month end (fewer whipsaws)
    month_end = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("last") == pd.Series(idx, index=idx)
    for n in (200, 250):
        sig_m = (etf["QQQ"] > etf["QQQ"].rolling(n).mean()).where(month_end.to_numpy()).ffill().fillna(False)
        add(f"qqq_sma_{n}_monthly", zero.assign(QQQ=sig_m.astype(float)), f"QQQ when close > SMA{n} evaluated at month end only, else cash")
    mom12 = (etf["QQQ"] / etf["QQQ"].shift(252) - 1) > (tb_cum / tb_cum.shift(252) - 1)
    add("qqq_absmom_252_monthly", zero.assign(QQQ=mom12.where(month_end.to_numpy()).ffill().fillna(False).astype(float)), "QQQ when 12-month return beats T-bills, evaluated at month end, else cash")
    # SPY as the regime for QQQ
    add("qqq_when_spy_gt_sma200", zero.assign(QQQ=(etf["SPY"] > etf["SPY"].rolling(200).mean()).astype(float)), "QQQ when SPY > its SMA200, else cash")

    # stock strategies gated by the QQQ SMA200 regime (2000-2024 point-in-time universe)
    if not args.no_stocks and os.path.exists("prices_long_cache.csv"):
        px, member = L.load("prices_long_cache.csv", "membership_long.pkl", "2000-01-03", "2024-12-31")
        f, sig, elig = S.build_all(px, member)
        dates = px.index
        st = int(dates.get_indexer([start_ts], method="bfill")[0])
        regime = (etf["QQQ"] > sma200).reindex(dates).ffill().fillna(False).to_numpy(bool)
        cr = cash_daily.reindex(dates).fillna(0.0).to_numpy(float)
        close = px.to_numpy(float); ss = f["short_score"].to_numpy(float); sm20 = f["sma20"].to_numpy(float)
        prio = {"ret10": f["ret10"].to_numpy(float), "long_score": f["long_score"].to_numpy(float), "deep_dd": (-f["dd_from_hi252"]).to_numpy(float)}
        for name, entry, pr, cfg in (
            ("stocks_double10_5pos_63d_stop12+regime", "double_10", "ret10", dict(max_pos=5, hold=63, trail=None, stop=0.12, rule_exit=None)),
            ("stocks_rev_double_m8_5pos_63d+regime", "rev_double_m8", "deep_dd", dict(max_pos=5, hold=63, trail=None, stop=None, rule_exit=None)),
            ("stocks_rev_short_m10_5pos_63d+regime", "rev_short_m10", "long_score", dict(max_pos=5, hold=63, trail=None, stop=None, rule_exit=None)),
            ("stocks_double10_5pos_63d_stop12_noregime", "double_10", "ret10", dict(max_pos=5, hold=63, trail=None, stop=0.12, rule_exit=None)),
        ):
            reg = None if name.endswith("noregime") else regime
            nav, trades = bt.backtest(close, sig[entry].to_numpy(bool), prio[pr], ss, sm20, cost=COST, start=st, regime=reg, cash_rate=cr, **cfg)
            nav = pd.Series(nav[st:] / nav[st], index=dates[st:])
            strategies[name] = nav
            strategies[name].attrs["cash_pct"] = round(float((~regime[st:]).mean() * 100), 1) if reg is not None else 0.0
            strategies[name].attrs["trades"] = len(trades)
            desc[name] = f"{entry} {cfg['max_pos']} positions, {cfg['hold']}d hold" + (", cash when QQQ < SMA200" if reg is not None else "")

    # ---- evaluation over the common 2000-03 .. 2024-12 window (stocks end 2024) and full window for ETF rules
    eq_full = pd.DataFrame({k: v for k, v in strategies.items()})
    eq_full = eq_full.loc[start_ts:]
    end_common = pd.Timestamp("2024-12-31")
    eq = eq_full.loc[:end_common].dropna(axis=1, how="all")
    eq = eq / eq.iloc[0]
    rows = []
    for col in eq.columns:
        m = metrics(eq[col]); m["strategy"] = col; m["description"] = desc.get(col, "")
        m["time_in_cash_pct"] = strategies[col].attrs.get("cash_pct", 0.0)
        m["trades"] = strategies[col].attrs.get("trades", np.nan)
        m["final_value_100k"] = round(args.capital * eq[col].iloc[-1])
        # full window (to 2026-10) where available
        full = eq_full[col].dropna()
        m["cagr_to_2026_pct"] = round(cagr(full / full.iloc[0]) * 100, 2) if full.index[-1] > end_common else np.nan
        rows.append(m)
    summary = pd.DataFrame(rows)
    ew = entry_windows(eq, "QQQ_buy_hold")
    piv = ew.pivot(index="strategy", columns="horizon_years", values="win_rate_vs_bench_pct").add_prefix("win_vs_QQQ_")
    piv2 = ew.pivot(index="strategy", columns="horizon_years", values="worst_excess_cagr_pct").add_prefix("worst_excess_")
    piv3 = ew.pivot(index="strategy", columns="horizon_years", values="median_excess_cagr_pct").add_prefix("median_excess_")
    summary = summary.merge(piv, left_on="strategy", right_index=True, how="left").merge(piv2, left_on="strategy", right_index=True, how="left").merge(piv3, left_on="strategy", right_index=True, how="left")
    bench_cagr = summary.loc[summary.strategy == "QQQ_buy_hold", "cagr_pct"].iloc[0]
    summary["beats_qqq_full_period"] = summary["cagr_pct"] > bench_cagr
    summary = summary.sort_values(["win_vs_QQQ_5", "cagr_pct"], ascending=False).reset_index(drop=True)
    cols = ["strategy", "cagr_pct", "max_dd_pct", "sharpe", "calmar", "worst_year_pct", "time_in_cash_pct", "final_value_100k", "win_vs_QQQ_3", "win_vs_QQQ_5", "win_vs_QQQ_10", "median_excess_5", "worst_excess_5", "worst_excess_10", "cagr_to_2026_pct"]
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    ew.to_csv(f"{args.out}_entry_windows.csv", index=False, encoding="utf-8-sig")
    eq_full.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")
    yearly = ((eq_full.resample("YE").last() / eq_full.resample("YE").first() - 1) * 100).round(1)
    yearly.index = yearly.index.year
    yearly.to_csv(f"{args.out}_yearly.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 260)
    print(f"evaluation window {eq.index[0].date()} .. {eq.index[-1].date()}; cash earns T-bills (mean {irx.loc[start_ts:end_common].mean():.2f}%)")
    print("\nSummary (sorted by 5-year entry-window win rate vs QQQ, then CAGR):")
    print(summary[cols].to_string(index=False))

    # examples with 100k from specific entry dates
    examples = []
    top = [s for s in summary.strategy.head(6) if s != "QQQ_buy_hold"][:4] + ["QQQ_buy_hold", "SPY_buy_hold"]
    for d in ("2000-03-01", "2003-01-02", "2007-10-01", "2009-03-02", "2012-01-03", "2015-01-02", "2020-02-19", "2022-01-03"):
        ts = eq.index[eq.index.get_indexer([pd.Timestamp(d)], method="bfill")[0]]
        row = {"entry_date": ts.date()}
        for sname in top:
            row[sname] = round(args.capital * eq[sname].iloc[-1] / eq[sname].loc[ts])
        examples.append(row)
    ex = pd.DataFrame(examples)
    ex.to_csv(f"{args.out}_examples_100k.csv", index=False, encoding="utf-8-sig")
    print(f"\n{int(args.capital):,} invested on each date, value on {eq.index[-1].date()}:")
    print(ex.to_string(index=False))
    print(f"\nwrote {args.out}_summary.csv, _entry_windows.csv, _equity.csv, _yearly.csv, _examples_100k.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
