#!/usr/bin/env python3
"""
Third pass: refine volatility targeting on QQQ (long-only, cash allowed, no leverage).

Section 9.6 found weight = min(1, T / realised vol) keeps QQQ's median CAGR and improves the
tail, but gives back 2-8 points in rebound years and trails QQQ by 0.3-0.8 points a year in
windows without a crash. The variants here attack those two costs:

  vt_T_W            baseline: min(1, T / W-day realised vol)
  ewma_T_L          EWMA vol (lambda L) instead of a rolling window
  down_T_W          downside semi-volatility only (negative returns), so rally volatility is ignored
  vix_T             implied vol: min(1, T / VIX)
  ratio_K_S_L       exposure 1 unless S-day vol > K x L-day vol (vol expanding), then T/vol
  floor_T_W_F       baseline with a minimum weight F
  weekly_T_W        baseline rebalanced only on the last trading day of each week
  dd_T_W            baseline applied only when QQQ is more than 10% below its 252-day high
  recover_T_W       baseline, but full weight when QQQ is >= 25% below its 252-day high and
                    10-day vol < 21-day vol (volatility receding after a crash)
  band_T_W_B        baseline, trade only when |target - current| > B
  combo             min(spike rule, vol-target with a 0.5 floor): the two mechanisms stacked

Metrics: full-period 2000-03..2024-12, the same 5,000 paired random draws (1-20y, seed 42),
and the two splits that matter: draws entered after 2009, and draws whose window had no
>= 30% QQQ drawdown. Outputs: <stem>_summary.csv, <stem>_equity.csv, docs/optimize_qqq2.png
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

import sp500_beat_qqq as B
import sp500_random_entry_mc as MC

COST = 0.001


def load(t: str) -> pd.Series:
    return pd.read_csv(f"{t}_cache.csv" if t in ("irx", "vix") else f"{t}_full_cache.csv", index_col=0, parse_dates=True).squeeze().sort_index()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2000-03-01")
    p.add_argument("--end", default="2026-10-16")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="sp500_optimize_qqq2")
    args = p.parse_args(argv)

    qqq, spy, irx, vix = load("qqq"), load("spy"), load("irx"), load("vix")
    idx = qqq.loc[: args.end].index
    etf = pd.DataFrame({"QQQ": qqq, "SPY": spy}).reindex(idx).ffill()
    cash_daily = (irx.reindex(idx).ffill() / 100 / 252).fillna(0.0)
    vixs = vix.reindex(idx).ffill() / 100
    start_ts = pd.Timestamp(args.start)
    zero = pd.DataFrame(0.0, index=idx, columns=["QQQ", "SPY"])
    q = etf["QQQ"]
    r = q.pct_change()
    hi252 = q.rolling(252).max()
    dd = q / hi252 - 1
    week_end = pd.Series(idx, index=idx).groupby([idx.isocalendar().year.to_numpy(), idx.isocalendar().week.to_numpy()]).transform("last") == pd.Series(idx, index=idx)

    def rvol(W):
        return r.rolling(W).std() * math.sqrt(252)

    def dvol(W):
        neg = r.clip(upper=0.0)
        return np.sqrt((neg ** 2).rolling(W).mean() * 2) * math.sqrt(252)  # x2 so a symmetric series matches rvol

    def ewma_vol(lam):
        v = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
        return np.sqrt(v) * math.sqrt(252)

    strategies, desc, cashpct, turnover = {}, {}, {}, {}

    def add(name, w, d):
        w = w.clip(lower=0.0, upper=1.0).fillna(0.0)
        nav = B.etf_timing(etf, zero.assign(QQQ=w), cash_daily, COST).loc[start_ts:]
        strategies[name] = nav / nav.iloc[0]
        desc[name] = d
        cashpct[name] = round(float((1 - w.loc[start_ts:]).mean() * 100), 1)
        turnover[name] = round(float(w.loc[start_ts:].diff().abs().sum() / ((idx[-1] - start_ts).days / 365.25)), 2)

    add("QQQ_buy_hold", pd.Series(1.0, index=idx), "buy and hold")
    for T in (0.25, 0.30):
        for W in (21,):
            add(f"vt_{int(T*100)}_{W}", T / rvol(W), f"baseline min(1, {int(T*100)}% / {W}d realised vol)")
    for T in (0.25, 0.30, 0.35):
        for lam in (0.94, 0.97):
            add(f"ewma_{int(T*100)}_{int(lam*100)}", T / ewma_vol(lam), f"min(1, {int(T*100)}% / EWMA vol, lambda {lam})")
    for T in (0.20, 0.25, 0.30, 0.35):
        for W in (21, 42):
            add(f"down_{int(T*100)}_{W}", T / dvol(W), f"min(1, {int(T*100)}% / {W}d downside semi-vol)")
    for T in (0.25, 0.30, 0.35, 0.40):
        add(f"vix_{int(T*100)}", T / vixs, f"min(1, {int(T*100)}% / VIX)")
    for K in (1.3, 1.5):
        for S, L in ((10, 63), (21, 126)):
            expanding = rvol(S) > K * rvol(L)
            add(f"ratio_{K}_{S}_{L}", pd.Series(np.where(expanding, 0.30 / rvol(S), 1.0), index=idx), f"full unless {S}d vol > {K} x {L}d vol, then 30% / {S}d vol")
    for F in (0.3, 0.5):
        add(f"floor_30_21_{int(F*100)}", (0.30 / rvol(21)).clip(lower=F), f"baseline 30%/21d with weight floor {F}")
    for T in (0.25, 0.30):
        w = (T / rvol(21)).where(week_end.to_numpy()).ffill()
        add(f"weekly_{int(T*100)}_21", w, f"baseline {int(T*100)}%/21d rebalanced weekly")
    for T in (0.25, 0.30):
        w = pd.Series(np.where(dd < -0.10, T / rvol(21), 1.0), index=idx)
        add(f"dd_{int(T*100)}_21", w, f"baseline {int(T*100)}%/21d only when QQQ > 10% below 252d high")
    for T in (0.25, 0.30):
        base = T / rvol(21)
        recover = (dd <= -0.25) & (rvol(10) < rvol(21))
        add(f"recover_{int(T*100)}_21", pd.Series(np.where(recover, 1.0, base), index=idx), f"baseline {int(T*100)}%/21d, full weight when >= 25% below high and 10d vol < 21d vol")
    for T in (0.25, 0.30):
        tgt = (T / rvol(21)).clip(upper=1.0).fillna(0.0).to_numpy()
        w = np.zeros(len(idx)); cur = 0.0
        for t in range(len(idx)):
            if abs(tgt[t] - cur) > 0.15:
                cur = tgt[t]
            w[t] = cur
        add(f"band_{int(T*100)}_21_15", pd.Series(w, index=idx), f"baseline {int(T*100)}%/21d, trade only when |target - current| > 0.15")
    # combinations
    add("down_30_21+recover", pd.Series(np.where((dd <= -0.25) & (rvol(10) < rvol(21)), 1.0, 0.30 / dvol(21)), index=idx), "downside-vol 30%/21d with recovery override")
    add("vix_30+recover", pd.Series(np.where((dd <= -0.25) & (rvol(10) < rvol(21)), 1.0, 0.30 / vixs), index=idx), "VIX 30% with recovery override")
    add("ewma_30_94+dd10", pd.Series(np.where(dd < -0.10, 0.30 / ewma_vol(0.94), 1.0), index=idx), "EWMA 30% only when > 10% below high")
    w_ratio = pd.Series(np.where(rvol(10) > 1.5 * rvol(63), 0.30 / rvol(10), 1.0), index=idx).clip(0, 1)
    add("combo_ratio+vt_floor50", pd.Series(np.minimum(w_ratio, (0.30 / rvol(21)).clip(lower=0.5, upper=1.0)), index=idx), "min(spike rule 1.5/10/63/30%, vol-target 30%/21d with floor 0.5)")

    eq = pd.DataFrame(strategies).loc[start_ts:]
    eq.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")
    e24 = eq.loc[: "2024-12-31"]; e24 = e24 / e24.iloc[0]
    rng = np.random.default_rng(args.seed)
    st, en, yr = MC.draw_windows(eq.index, args.n, 1, 20, rng)
    qn = eq["QQQ_buy_hold"].to_numpy()
    gq, cq, _ = MC.window_stats(qn, st, en, yr)
    post = eq.index[st] >= pd.Timestamp("2009-01-01")
    crash = np.array([(qn[s_:e_ + 1] / np.maximum.accumulate(qn[s_:e_ + 1]) - 1).min() <= -0.30 for s_, e_ in zip(st, en)])
    rows = []
    for c in eq.columns:
        g, cg, _ = MC.window_stats(eq[c].to_numpy(), st, en, yr)
        ex = (cg - cq) * 100
        win = g > gq
        m = B.metrics(e24[c])
        rows.append({"strategy": c, "description": desc[c], "cagr_00_24": m["cagr_pct"], "max_dd": m["max_dd_pct"], "sharpe": m["sharpe"], "worst_year": m["worst_year_pct"],
                     "avg_cash_pct": cashpct[c], "turnover_per_year": turnover[c], "cagr_to_2026": round(B.cagr(eq[c]) * 100, 2),
                     "mc_win_pct": round(float(win.mean() * 100), 1), "mc_median_cagr": round(float(np.median(cg)) * 100, 2), "mc_p10_cagr": round(float(np.percentile(cg, 10)) * 100, 2),
                     "mc_worst_cagr": round(float(cg.min()) * 100, 1), "mc_prob_loss": round(float((g < 1).mean() * 100), 1),
                     "mc_mean_excess": round(float(ex.mean()), 2), "mc_median_excess": round(float(np.median(ex)), 2),
                     "post2009_win_pct": round(float(win[post].mean() * 100), 1), "post2009_median_excess": round(float(np.median(ex[post])), 2), "post2009_mean_excess": round(float(ex[post].mean()), 2),
                     "nocrash_win_pct": round(float(win[~crash].mean() * 100), 1), "nocrash_median_excess": round(float(np.median(ex[~crash])), 2),
                     "crash_win_pct": round(float(win[crash].mean() * 100), 1), "crash_median_excess": round(float(np.median(ex[crash])), 2)})
    summary = pd.DataFrame(rows)
    # objective: keep tail gains while cutting the bull-market cost -> rank by post-2009 mean excess subject to p10 > QQQ p10
    qp10 = summary.loc[summary.strategy == "QQQ_buy_hold", "mc_p10_cagr"].iloc[0]
    summary["tail_better"] = summary.mc_p10_cagr > qp10
    summary = summary.sort_values(["tail_better", "post2009_mean_excess"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    cols = ["strategy", "cagr_00_24", "max_dd", "avg_cash_pct", "turnover_per_year", "mc_win_pct", "mc_median_cagr", "mc_p10_cagr", "mc_worst_cagr", "mc_prob_loss", "mc_mean_excess",
            "post2009_win_pct", "post2009_mean_excess", "nocrash_median_excess", "crash_median_excess", "tail_better"]
    print(summary[cols].to_string(index=False))
    chart(summary, "docs/optimize_qqq2.png")
    print(f"\nwrote {args.out}_summary.csv, _equity.csv, docs/optimize_qqq2.png")
    return 0


def chart(summary: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "WenQuanYi" in f.name or "Noto Sans CJK" in f.name})
    if cjk:
        plt.rcParams["font.family"] = cjk[0]
    plt.rcParams["axes.unicode_minus"] = False
    s = summary[summary.strategy != "QQQ_buy_hold"].copy()
    fam = s.strategy.str.extract(r"^(vt|ewma|down|vix|ratio|floor|weekly|dd|recover|band)")[0].fillna("other")
    colors = {"vt": "#52514e", "ewma": "#2a78d6", "down": "#eb6834", "vix": "#1baf7a", "ratio": "#eda100", "floor": "#e87ba4", "weekly": "#008300", "dd": "#6250d6", "recover": "#e34948", "band": "#9a9890", "other": "#0b0b0b"}
    fig, ax = plt.subplots(figsize=(13, 6.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = s["post2009_mean_excess"].to_numpy(); y = s["mc_p10_cagr"].to_numpy()
    for f in colors:
        mk = (fam == f).to_numpy()
        if mk.any():
            ax.scatter(x[mk], y[mk], s=55, color=colors[f], alpha=0.9, edgecolor="#fcfcfb", linewidth=1, label=f)
    placed = []
    for xi, yi, name in sorted(zip(x, y, s.strategy), key=lambda t: (t[0], t[1])):
        dy = 3
        for px_, py_ in placed:
            if abs(px_ - xi) < 0.06 and abs(py_ - yi) < 0.12:
                dy += 9
        ax.annotate(name, (xi, yi), xytext=(5, dy), textcoords="offset points", fontsize=7.5, color="#52514e")
        placed.append((xi, yi))
    qq = summary[summary.strategy == "QQQ_buy_hold"].iloc[0]
    ax.axhline(qq.mc_p10_cagr, color="#c3c2b7", lw=1, ls="--")
    ax.annotate(f"QQQ p10 CAGR {qq.mc_p10_cagr}%", (x.min(), qq.mc_p10_cagr), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color="#52514e")
    ax.axvline(0, color="#c3c2b7", lw=1, ls="--")
    ax.set_xlabel("mean excess CAGR vs QQQ for draws entered after 2009 (percentage points; 0 = no bull-market cost)", color="#52514e", fontsize=9)
    ax.set_ylabel("10th-percentile CAGR across all 5,000 draws, %", color="#52514e", fontsize=9)
    ax.set_title("Volatility-targeting refinements: bull-market cost vs tail protection", fontsize=11, loc="left", color="#0b0b0b")
    ax.grid(True, color="#e6e5e1", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, title="family", title_fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    sys.exit(main())
