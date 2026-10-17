#!/usr/bin/env python3
"""
Out-of-sample test: do the QQQ cash-allowed timing rules generalise to assets they never saw?

Earlier passes (sp500_optimize_qqq*.py) tuned volatility targeting, a vol-spike ratio rule and
SMA timing on QQQ alone (2000-2026). Here the SAME rules with the SAME frozen parameters are
applied, unchanged, to 18 other ETFs (US size/style, regions, S&P sectors, gold, long bonds),
each traded on its own against T-bill cash with 0.10% cost per unit of weight changed.

Rules (weights decided at close t-1, applied at t):
  buy_hold                 always 100% in the asset
  vt_T_W                   min(1, T / W-day realised vol)            vt_30_21, vt_25_21, vt_20_63
  ratio_1.5_10_63          1.0 unless 10d vol > 1.5 x 63d vol, then 30% / 10d vol
  combo_ratio+vt_floor50   min(ratio rule, clip(30% / 21d vol, 0.5, 1))
  sma_200                  1 if close > SMA200 else 0, evaluated daily
  sma_200_monthly          same, evaluated at month end and held for the month
  sma_200_dd_15            1 only if close > SMA200 and close is within 15% of its 252-day high

Periods: full (first close + 260 trading days warm-up .. --end) and post-2009 (2010-01-04 .. --end,
no 2000-02 or 2008 crash inside). Per asset x rule x period: CAGR, vol, Sharpe, max DD, worst year,
avg cash, turnover, excess CAGR vs the asset's own buy-and-hold, max-DD improvement, plus a paired
random-entry Monte Carlo (same draws for every rule on an asset: holds 1-10y, --n draws, --seed).
Aggregate per rule x period across the out-of-sample assets (QQQ reported separately) with a
two-sided binomial sign test on "excess CAGR >= 0".

Outputs: <out>_metrics.csv, <out>_mc.csv, <out>_summary.csv, <out>_equity_post2009.csv,
         docs/cross_asset.png. Prices cached in cross_asset_cache.csv (--refresh to redownload).
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
CACHE = "cross_asset_cache.csv"
TICKERS = ["SPY", "DIA", "IWM", "MDY", "EFA", "EEM", "EWJ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "GLD", "TLT"]
RULES = ["buy_hold", "vt_30_21", "vt_25_21", "vt_20_63", "ratio_1.5_10_63", "combo_ratio+vt_floor50", "sma_200", "sma_200_monthly", "sma_200_dd_15"]
WARMUP = 260
MIN_YEARS = 12
POST_START = pd.Timestamp("2010-01-04")


def load_prices(end: str, refresh: bool) -> pd.DataFrame:
    """Adjusted closes for TICKERS (cached) plus QQQ from qqq_full_cache.csv."""
    if refresh or not os.path.exists(CACHE):
        import yfinance as yf

        px = yf.download(TICKERS, start="1990-01-01", auto_adjust=True, progress=False)["Close"]
        px = px.loc[:, [c for c in TICKERS if c in px.columns]]
        px.to_csv(CACHE)
        print(f"downloaded {px.shape[1]} tickers -> {CACHE}")
    px = pd.read_csv(CACHE, index_col=0, parse_dates=True).sort_index()
    qqq = pd.read_csv("qqq_full_cache.csv", index_col=0, parse_dates=True).squeeze().sort_index().rename("QQQ")
    px = pd.concat([qqq, px], axis=1).loc[:end]
    keep, dropped = [], []
    for c in px.columns:
        s = px[c].dropna()
        yrs = (s.index[-1] - s.index[0]).days / 365.25 if len(s) else 0
        (keep if yrs >= MIN_YEARS else dropped).append(c)
    if dropped:
        print(f"dropped (download failed or < {MIN_YEARS}y): {dropped}")
    return px[keep]


def build_weights(close: pd.Series) -> dict[str, pd.Series]:
    """All rule weights for one asset on its own index, clipped to [0, 1], NaN -> 0 (cash)."""
    idx = close.index
    r = close.pct_change()
    rv = {W: r.rolling(W).std() * math.sqrt(252) for W in (10, 21, 63)}
    sma200 = close.rolling(200).mean()
    dd = close / close.rolling(252).max() - 1
    month_end = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("last") == pd.Series(idx, index=idx)
    w = {"buy_hold": pd.Series(1.0, index=idx)}
    for name, T, W in (("vt_30_21", 0.30, 21), ("vt_25_21", 0.25, 21), ("vt_20_63", 0.20, 63)):
        w[name] = T / rv[W]
    ratio = pd.Series(np.where(rv[10] > 1.5 * rv[63], 0.30 / rv[10], 1.0), index=idx)
    w["ratio_1.5_10_63"] = ratio
    w["combo_ratio+vt_floor50"] = pd.Series(np.minimum(ratio.clip(0, 1), (0.30 / rv[21]).clip(lower=0.5, upper=1.0)), index=idx)
    above = close > sma200
    w["sma_200"] = above.astype(float)
    w["sma_200_monthly"] = above.where(month_end.to_numpy()).ffill().fillna(False).astype(float)
    w["sma_200_dd_15"] = (above & (dd >= -0.15)).astype(float)
    return {k: v.clip(lower=0.0, upper=1.0).fillna(0.0) for k, v in w.items()}


def period_metrics(nav: pd.Series, w: pd.Series) -> dict:
    m = B.metrics(nav)
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return {"cagr_pct": m["cagr_pct"], "ann_vol_pct": m["ann_vol_pct"], "sharpe": m["sharpe"], "max_dd_pct": m["max_dd_pct"], "worst_year_pct": m["worst_year_pct"],
            "avg_cash_pct": round(float((1 - w).mean() * 100), 1), "turnover_per_year": round(float(w.diff().abs().sum() / yrs), 2), "years": round(yrs, 1)}


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for k successes out of n with p = 0.5."""
    if n == 0:
        return float("nan")
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    lo = sum(pmf[: k + 1]); hi = sum(pmf[k:])
    return round(min(1.0, 2 * min(lo, hi)), 4)


def run(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    px = load_prices(args.end, args.refresh)
    irx = pd.read_csv("irx_cache.csv", index_col=0, parse_dates=True).squeeze().sort_index()
    met_rows, mc_rows, eq_post = [], [], {}
    check = {}
    for asset in px.columns:
        close = px[asset].dropna()
        idx = close.index
        cash_daily = (irx.reindex(idx).ffill() / 100 / 252).fillna(0.0)
        prices = close.to_frame(asset)
        weights = build_weights(close)
        navs = {k: B.etf_timing(prices, prices.assign(**{asset: v}), cash_daily, COST) for k, v in weights.items()}
        if asset == "QQQ":
            for k in ("vt_30_21", "sma_200_monthly"):
                s = navs[k].loc["2000-03-01":"2024-12-31"]
                check[k] = B.metrics(s / s.iloc[0])
        periods = {"full": (idx[WARMUP], idx[-1])}
        if idx[0] <= POST_START - pd.Timedelta(days=400):
            periods["post2009"] = (POST_START, idx[-1])
        for per, (p0, p1) in periods.items():
            sub = {k: v.loc[p0:p1] for k, v in navs.items()}
            sub = {k: v / v.iloc[0] for k, v in sub.items()}
            pidx = sub["buy_hold"].index
            bh = period_metrics(sub["buy_hold"], weights["buy_hold"].loc[p0:p1])
            rng = np.random.default_rng(args.seed)
            max_y = min(10, int((pidx[-1] - pidx[0]).days / 365.25) - 1)
            st, en, yr = MC.draw_windows(pidx, args.n, 1, max_y, rng)
            g_bh, c_bh, _ = MC.window_stats(sub["buy_hold"].to_numpy(), st, en, yr)
            for rule in RULES:
                m = period_metrics(sub[rule], weights[rule].loc[p0:p1])
                m.update(asset=asset, rule=rule, period=per, start=p0.date(), end=p1.date(), excess_cagr_pct=round(m["cagr_pct"] - bh["cagr_pct"], 2),
                         dd_improvement_pts=round(m["max_dd_pct"] - bh["max_dd_pct"], 1), final_x=round(float(sub[rule].iloc[-1]), 2))
                met_rows.append(m)
                g, cg, _ = MC.window_stats(sub[rule].to_numpy(), st, en, yr)
                ex = (cg - c_bh) * 100
                mc_rows.append({"asset": asset, "rule": rule, "period": per, "n_draws": args.n, "max_hold_years": max_y,
                                "win_pct": round(float((g > g_bh).mean() * 100), 1), "median_cagr_pct": round(float(np.median(cg)) * 100, 2),
                                "p10_cagr_pct": round(float(np.percentile(cg, 10)) * 100, 2), "worst_cagr_pct": round(float(cg.min()) * 100, 2),
                                "prob_loss_pct": round(float((g < 1).mean() * 100), 1), "median_excess_pct": round(float(np.median(ex)), 2),
                                "p10_excess_pct": round(float(np.percentile(ex, 10)), 2), "mean_excess_pct": round(float(ex.mean()), 2),
                                "bh_median_cagr_pct": round(float(np.median(c_bh)) * 100, 2), "bh_p10_cagr_pct": round(float(np.percentile(c_bh, 10)) * 100, 2)})
                if per == "post2009":
                    eq_post[f"{asset}_{rule}"] = sub[rule]
        print(f"{asset}: {len(idx)} days, full from {periods['full'][0].date()}, post2009 {'yes' if 'post2009' in periods else 'no'}")
    print("\nQQQ validation 2000-03-01..2024-12-31 (expect vt_30_21 ~9.3% / -70.7%, sma_200_monthly ~9.4% / -41.3%):")
    for k, m in check.items():
        print(f"  {k:18s} CAGR {m['cagr_pct']:.2f}%  max DD {m['max_dd_pct']:.1f}%")
    met = pd.DataFrame(met_rows)
    cols = ["asset", "rule", "period", "start", "end", "years", "final_x", "cagr_pct", "ann_vol_pct", "sharpe", "max_dd_pct", "worst_year_pct", "avg_cash_pct", "turnover_per_year",
            "excess_cagr_pct", "dd_improvement_pts"]
    met = met[cols]
    mc = pd.DataFrame(mc_rows)
    summary = aggregate(met, mc)
    eq = pd.DataFrame(eq_post)
    return met, mc, summary, eq


def aggregate(met: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for per in ("full", "post2009"):
        for rule in RULES[1:]:
            for scope in ("out_of_sample", "QQQ"):
                m = met[(met.period == per) & (met.rule == rule)]
                c = mc[(mc.period == per) & (mc.rule == rule)]
                m = m[m.asset == "QQQ"] if scope == "QQQ" else m[m.asset != "QQQ"]
                c = c[c.asset == "QQQ"] if scope == "QQQ" else c[c.asset != "QQQ"]
                c = c.set_index("asset").reindex(m.asset)
                n = len(m)
                k = int((m.excess_cagr_pct >= 0).sum())
                rows.append({"period": per, "rule": rule, "scope": scope, "n_assets": n, "n_excess_cagr_ge0": k, "n_dd_improved": int((m.dd_improvement_pts > 0).sum()),
                             "n_mc_median_excess_ge0": int((c.median_excess_pct >= 0).sum()), "n_mc_p10_improved": int((c.p10_cagr_pct > c.bh_p10_cagr_pct).sum()),
                             "n_mc_win_gt50": int((c.win_pct > 50).sum()), "mean_excess_cagr_pct": round(float(m.excess_cagr_pct.mean()), 2),
                             "median_excess_cagr_pct": round(float(m.excess_cagr_pct.median()), 2), "mean_dd_improvement_pts": round(float(m.dd_improvement_pts.mean()), 1),
                             "mean_mc_median_excess_pct": round(float(c.median_excess_pct.mean()), 2), "mean_mc_win_pct": round(float(c.win_pct.mean()), 1),
                             "sign_test_p_excess_ge0": sign_test_p(k, n), "sharpe_bh_mean": round(float(met[(met.period == per) & (met.rule == "buy_hold") & (met.asset.isin(m.asset))].sharpe.mean()), 2),
                             "sharpe_rule_mean": round(float(m.sharpe.mean()), 2)})
    return pd.DataFrame(rows)


def chart(met: pd.DataFrame, path: str = "docs/cross_asset.png") -> None:
    """Two stacked heatmaps (full period): excess CAGR vs buy-and-hold, and max-DD improvement."""
    import matplotlib

    matplotlib.use("Agg")
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#fcfcfb", "#2a78d6"])
    full = met[(met.period == "full") & (met.rule != "buy_hold")]
    assets = ["QQQ"] + [a for a in TICKERS if a in set(full.asset)]
    rules = RULES[1:]
    panels = [("excess_cagr_pct", "年化報酬差（規則 - 買進持有，百分點，完整期間）", "%"),
              ("dd_improvement_pts", "最大回撤改善（百分點，正值 = 回撤較小，完整期間）", "")]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.6), facecolor="#fcfcfb")
    for ax, (col, title, _) in zip(axes, panels):
        mat = full.pivot(index="rule", columns="asset", values=col).reindex(index=rules, columns=assets)
        v = float(np.nanmax(np.abs(mat.to_numpy())))
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        ax.set_facecolor("#fcfcfb")
        im = ax.imshow(mat.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(rules)):
            for j in range(len(assets)):
                val = mat.iat[i, j]
                if np.isnan(val):
                    continue
                dark = abs(val) > 0.55 * v
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8.2, color="#fcfcfb" if dark else "#0b0b0b")
        ax.set_xticks(range(len(assets))); ax.set_xticklabels(assets, fontsize=9)
        ax.set_yticks(range(len(rules))); ax.set_yticklabels(rules, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11.5, color="#0b0b0b", pad=10)
        ax.set_xlabel("資產（QQQ 為樣本內參考，其餘為樣本外）", fontsize=9, color="#52514e")
        ax.set_ylabel("規則", fontsize=9, color="#52514e")
        ax.axvline(0.5, color="#52514e", lw=1.2, ls="--")
        ax.set_xticks(np.arange(-0.5, len(assets), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(rules), 1), minor=True)
        ax.grid(which="minor", color="#fcfcfb", lw=1.5); ax.tick_params(which="minor", length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}".replace("\u2212", "-")))
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("QQQ 上調參的擇時規則套用到 18 檔從未見過的 ETF：橘 = 比買進持有差，藍 = 較好", fontsize=13, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--end", default="2026-10-16")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh", action="store_true", help="redownload prices into the cache")
    p.add_argument("--out", default="sp500_cross_asset")
    p.add_argument("--chart-only", action="store_true", help="draw docs/cross_asset.png from <out>_metrics.csv")
    args = p.parse_args(argv)
    if args.chart_only:
        chart(pd.read_csv(f"{args.out}_metrics.csv"))
        print("wrote docs/cross_asset.png")
        return 0
    met, mc, summary, eq = run(args)
    met.to_csv(f"{args.out}_metrics.csv", index=False, encoding="utf-8-sig")
    mc.to_csv(f"{args.out}_mc.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    eq.round(6).to_csv(f"{args.out}_equity_post2009.csv", encoding="utf-8-sig")
    chart(met)
    pd.set_option("display.width", 320)
    cols = ["period", "rule", "scope", "n_assets", "n_excess_cagr_ge0", "n_dd_improved", "n_mc_median_excess_ge0", "n_mc_p10_improved", "n_mc_win_gt50", "mean_excess_cagr_pct",
            "median_excess_cagr_pct", "mean_dd_improvement_pts", "mean_mc_median_excess_pct", "sign_test_p_excess_ge0"]
    print("\n" + summary[cols].to_string(index=False))
    print(f"\nwrote {args.out}_metrics.csv, _mc.csv, _summary.csv, _equity_post2009.csv, docs/cross_asset.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
