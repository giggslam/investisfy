#!/usr/bin/env python3
"""
Forward search (section 20): trade the audited composite score with the audited timing switch and
place the result on the section 19 grid.

Inputs produced by the two audits:
  signal_audit_scores.csv  : out-of-sample composite scores per (signal_date, ticker) from
                             sp500_signal_audit.py (anchored walk-forward; the score at month end t
                             uses only data <= t and a model fitted on years before t's year).
  sp500_skill_needed_combined.csv : the IC x hit-rate grid (section 19.3) for the expected CAGR / MDD.

Portfolios (same engine, costs and cash yield as section 15): at each month end take the top N by
score, equal weight, execute at the next close, hold a month. Optional month-end switch: when the
rule calls the next month "down" the book sits in T-bills. Baselines: 63-day momentum top N, and
random N-stock portfolios, over the same months. For every curve the realised monthly IC of the
score and the realised hit rate of the switch are also reported, next to the grid cell they imply.

Usage: python sp500_forward_search.py [--scores signal_audit_scores.csv] [--sizes 5,20]
       [--switches none,qqq_sma200,qqq_dd15,spy_sma200,breadth50,vote3] [--random-runs 50] [--out forward_search]
       [--chart docs/forward_search.png] [--chart-only]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sp500_top5_monthly as T5
from sp500_precursor_events import load_universe

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def stats(nav: pd.Series) -> tuple[float, float]:
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return nav.iloc[-1] ** (1 / yrs) - 1, (nav / nav.cummax() - 1).min()


def load_series(path: str, idx: pd.DatetimeIndex) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True)
    s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    return s.sort_index().reindex(idx).ffill()


def switch_calls(name: str, px: pd.DataFrame, member: pd.DataFrame, qqq: pd.Series, spy: pd.Series, mes: list[pd.Timestamp]) -> dict[pd.Timestamp, bool]:
    """True = hold next month. Every input is known at the month end t."""
    if name == "none":
        return {d: True for d in mes}
    q_sma = qqq.rolling(200).mean()
    s_sma = spy.rolling(200).mean()
    above200 = (px > px.rolling(200).mean()).where(member)
    breadth = above200.sum(axis=1) / member.sum(axis=1).replace(0, np.nan)
    mom121 = spy.shift(21) / spy.shift(252) - 1
    q_dd = qqq / qqq.rolling(252).max() - 1
    calls = {}
    for d in mes:
        a = bool(qqq[d] > q_sma[d]); b = bool(spy[d] > s_sma[d]); c = bool(breadth[d] > 0.5); m = bool(mom121[d] > 0); dd = bool(q_dd[d] > -0.15)
        calls[d] = {"qqq_sma200": a, "spy_sma200": b, "breadth50": c, "vote3": (b + m + c) >= 2, "qqq_dd15": dd}[name]
    return calls


def spearman(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a, b], axis=1).dropna()
    if len(j) < 10:
        return np.nan
    return float(np.corrcoef(j.iloc[:, 0].rank(), j.iloc[:, 1].rank())[0, 1])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", default="signal_audit_scores.csv")
    p.add_argument("--score-cols", default="score_ew,score_ridge")
    p.add_argument("--grid", default="sp500_skill_needed_combined.csv")
    p.add_argument("--sizes", default="5,20")
    p.add_argument("--switches", default="none,qqq_sma200,qqq_dd15,spy_sma200,breadth50,vote3")
    p.add_argument("--random-runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="forward_search")
    p.add_argument("--chart", default="docs/forward_search.png")
    p.add_argument("--chart-only", action="store_true")
    args = p.parse_args(argv)
    sizes = [int(x) for x in args.sizes.split(",")]
    switches = args.switches.split(",")
    if args.chart_only:
        chart(pd.read_csv(f"{args.out}_equity.csv", index_col=0, parse_dates=True), pd.read_csv(f"{args.out}_summary.csv"), args.chart)
        return 0

    px, member, eligible, spy = load_universe()
    idx = px.index
    mes = [d for d in T5.month_ends(idx) if d >= pd.Timestamp("2000-02-29")]
    qqq = load_series("qqq_full_cache.csv", idx)
    irx = load_series("irx_cache.csv", idx)
    cash = (irx / 100 / 252).to_numpy()
    # forward 1-month return (exec close to exec close) for realised IC / hit rate
    fwd, spy_next = {}, {}
    for k in range(len(mes) - 1):
        a = idx.get_loc(mes[k]) + 1; b = idx.get_loc(mes[k + 1]) + 1
        if b >= len(idx):
            break
        fwd[mes[k]] = (px.iloc[b] / px.iloc[a] - 1).where(eligible.loc[mes[k]]).dropna()
        spy_next[mes[k]] = spy.iloc[b] / spy.iloc[a] - 1

    sc = pd.read_csv(args.scores, parse_dates=["signal_date"])
    score_cols = [c for c in args.score_cols.split(",") if c in sc.columns]
    first = sc.signal_date.min()
    months = [d for d in mes if d >= first and d in fwd]
    print(f"scores {args.scores}: {score_cols}, {sc.signal_date.nunique()} month-ends from {first.date()}; trading {len(months)} months")
    sc = sc[sc.signal_date.isin(months)]
    by_month = {d: g.set_index("ticker") for d, g in sc.groupby("signal_date")}
    mom = T5.rank_frames(px, eligible)["medium"]

    def picks_from(series_by_month, n):
        out = {}
        for d in months:
            s = series_by_month(d)
            s = s[s.index.isin(eligible.columns[eligible.loc[d]])].dropna()
            out[d] = list(s.nlargest(n).index)
        return out

    calls = {sw: switch_calls(sw, px, member, qqq, spy, mes) for sw in switches}
    hit = {sw: float(np.mean([(calls[sw][d]) == (spy_next[d] > 0) for d in months])) for sw in switches}
    gate = lambda picks, sw: {d: (v if calls[sw][d] else []) for d, v in picks.items()}

    grid = pd.read_csv(args.grid)

    def grid_cell(n, ic, h):
        g = grid[grid.n == n]
        r = g.iloc[((g.ic - ic).abs() * 10 + (g.hit - h).abs()).argmin()]
        return r.ic, r.hit, r.cagr, r.mdd

    curves, rows = {}, []
    rng = np.random.default_rng(args.seed)
    sources = {c: (lambda d, c=c: by_month[d][c]) for c in score_cols}
    sources["momentum63"] = lambda d: mom.loc[d]
    for name, src in sources.items():
        ics = pd.Series({d: spearman(src(d), fwd[d]) for d in months})
        for n in sizes:
            base = picks_from(src, n)
            for sw in switches:
                nav = T5.run_portfolio(px, gate(base, sw), cash_daily=cash)
                key = f"{name}_top{n}_{sw}"
                curves[key] = nav
                c, m = stats(nav)
                gi, gh, gc, gm = grid_cell(n, max(ics.mean(), 0), hit[sw])
                rows.append({"source": name, "n": n, "switch": sw, "cagr": c, "mdd": m, "ic_mean": ics.mean(), "ic_t": ics.mean() / ics.std() * np.sqrt(len(ics)),
                             "ic_years_pos": float((ics.groupby(ics.index.year).mean() > 0).mean()), "hit": hit[sw],
                             "grid_ic": gi, "grid_hit": gh, "grid_cagr": gc, "grid_mdd": gm, "start": months[0].date(), "end": nav.index[-1].date()})
                print(f"{key:40s} CAGR {c*100:6.1f}%  MDD {m*100:6.1f}%  | IC {ics.mean():+.3f} (t {rows[-1]['ic_t']:.1f})  hit {hit[sw]*100:.0f}%  grid cell ({gi:.2f},{gh:.1f}) -> {gc*100:.0f}% / {gm*100:.0f}%", flush=True)
    # random controls (unfiltered and with the first non-trivial switch)
    for n in sizes:
        for sw in ["none"] + [s for s in switches if s != "none"][:1]:
            res = []
            for r in range(args.random_runs):
                picks = {d: list(rng.choice(fwd[d].index, size=min(n, len(fwd[d])), replace=False)) for d in months}
                res.append(stats(T5.run_portfolio(px, gate(picks, sw), cash_daily=cash)))
            c = float(np.median([x[0] for x in res])); m = float(np.median([x[1] for x in res]))
            rows.append({"source": "random", "n": n, "switch": sw, "cagr": c, "mdd": m, "ic_mean": 0.0, "ic_t": 0.0, "ic_years_pos": np.nan, "hit": hit[sw],
                         "cagr_p90": float(np.percentile([x[0] for x in res], 90)), "start": months[0].date(), "end": months[-1].date()})
            print(f"random_top{n}_{sw:12s} median CAGR {c*100:6.1f}%  MDD {m*100:6.1f}%  p90 CAGR {rows[-1]['cagr_p90']*100:.1f}%")
    # benchmarks over the same window
    for nm, s in (("SPY", spy), ("QQQ", qqq)):
        nav = s.loc[months[0]:].dropna(); nav = nav / nav.iloc[0]
        curves[nm] = nav
        c, m = stats(nav)
        rows.append({"source": nm, "n": 0, "switch": "none", "cagr": c, "mdd": m, "start": nav.index[0].date(), "end": nav.index[-1].date()})
        print(f"{nm:40s} CAGR {c*100:6.1f}%  MDD {m*100:6.1f}%")
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    eq = pd.DataFrame(curves)
    eq.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")
    chart(eq, summary, args.chart)
    print(f"wrote {args.out}_summary.csv, {args.out}_equity.csv, {args.chart}")
    return 0


def chart(eq: pd.DataFrame, summary: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    s = summary[summary.source.isin(["SPY", "QQQ"]) == False]
    ax = axes[1]
    labels = {"score_ew": "稽核分數（等權）", "score_ridge": "稽核分數（ridge）", "momentum63": "63 日動能", "random": "隨機"}
    marks = {"none": "o", "qqq_sma200": "s", "qqq_dd15": "P", "spy_sma200": "D", "breadth50": "^", "vote3": "v"}
    for i, (src, g) in enumerate(s.groupby("source")):
        col = PAL[i % len(PAL)] if src != "random" else "#999999"
        for _, r in g.iterrows():
            ax.scatter(-r.mdd * 100, r.cagr * 100, color=col, marker=marks.get(r.switch, "o"), s=60 if r.n == 20 else 36, alpha=0.9, edgecolor="white")
    for src, col in zip(sorted(s.source.unique()), PAL):
        ax.scatter([], [], color=PAL[sorted(s.source.unique()).index(src) % len(PAL)] if src != "random" else "#999999", label=labels.get(src, src))
    for sw, mk in marks.items():
        ax.scatter([], [], color="#555555", marker=mk, label=f"開關：{sw}")
    for nm, col in (("SPY", "#555555"), ("QQQ", "#999999")):
        r = summary[summary.source == nm].iloc[0]
        ax.scatter(-r.mdd * 100, r.cagr * 100, color=col, marker="*", s=160, label=nm)
    ax.axhline(30, color="#555555", ls=":", lw=1); ax.axvline(30, color="#555555", ls=":", lw=1)
    ax.axvspan(0, 30, color="#1baf7a", alpha=0.06)
    ax.set_xlabel("最深回撤（%）"); ax.set_ylabel("CAGR（%）")
    ax.set_title("實際回測落點（大點 = 前 20 隻，小點 = 前 5 隻）對 30% / 30% 目標區", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right"); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[0]
    keys = [c for c in eq.columns if c.endswith("_top20_none") or c.endswith("_top20_qqq_sma200")] + ["SPY", "QQQ"]
    def nice(k):
        base = {"score_ew": "稽核分數（等權）", "score_ridge": "稽核分數（ridge）", "momentum63": "63 日動能"}
        for b, lab in base.items():
            if k.startswith(b):
                rest = k[len(b) + 1:]
                n, sw = rest.split("_", 1)
                return f"{lab} {n.replace('top', '前 ')} 隻" + ("" if sw == "none" else "＋200 日線開關")
        return k
    for i, k in enumerate(keys):
        ax.plot(eq.index, eq[k], lw=1.3 if "score" in k else 1.0, ls="--" if k in ("SPY", "QQQ") else "-", color=PAL[i % len(PAL)] if k not in ("SPY", "QQQ") else ("#555555" if k == "SPY" else "#999999"))
        ax.annotate(nice(k), (eq.index[-1], eq[k].dropna().iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.5, va="center")
    ax.set_yscale("log"); ax.set_ylabel("淨值（倍，對數座標）")
    ax.set_title("前 20 隻：稽核分數對 63 日動能，無開關與 QQQ 200 日線開關", loc="left", fontsize=11)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("第 20 節：按第 19 節流程正向找訊號後，實際能做到的 CAGR 與 MDD", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
