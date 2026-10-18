#!/usr/bin/env python3
"""
How much foresight does "30% CAGR with max drawdown under 30%" require? (section 19)

Instead of searching for a rule, inject a controlled amount of hindsight and measure what comes out:

* Stock selection: at each month end the signal is  IC * z(next-month return) + sqrt(1 - IC^2) * noise,
  cross-sectionally over the point-in-time S&P 500 members (same universe and portfolio engine as
  section 15). Buy the top N equal weight at the next close, hold a month, 0.1% cost. IC is the
  rank correlation between the signal and the realised next-month return; IC = 0 is random picking,
  IC = 1 is perfect foresight. Realistic published factors sit around 0.02 to 0.05.
* Timing: QQQ month-end all-in/all-out where the next month's direction is called correctly with
  probability `hit`; 0.5 is a coin flip, the SMA rules of section 13 land near 0.55 to 0.60.

Each noisy setting is repeated over several seeds and the median CAGR / MDD is reported for 2000-2024.

Usage: python sp500_skill_needed.py [--sizes 5,20] [--seeds 5] [--out sp500_skill_needed] [--chart docs/skill_needed.png] [--chart-only]
Outputs: <out>_selection.csv, <out>_timing.csv, <chart>
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sp500_top5_monthly as T5
from sp500_precursor_events import load_universe

ICS = [0.0, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
HITS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]


def stats(nav: pd.Series) -> tuple[float, float]:
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return nav.iloc[-1] ** (1 / yrs) - 1, (nav / nav.cummax() - 1).min()


def forward_returns(px: pd.DataFrame, eligible: pd.DataFrame, mes: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.Series]:
    """Next-month return from the execution close (day after month end t) to the next execution close."""
    idx = px.index
    fwd = {}
    for k in range(len(mes) - 1):
        a = idx.get_loc(mes[k]) + 1
        b = idx.get_loc(mes[k + 1]) + 1
        if b >= len(idx):
            break
        fwd[mes[k]] = (px.iloc[b] / px.iloc[a] - 1).where(eligible.loc[mes[k]]).dropna()
    return fwd


def picks_with_ic(fwd: dict, ic: float, n: int, rng: np.random.Generator) -> dict[pd.Timestamp, list[str]]:
    picks = {}
    for d, r in fwd.items():
        rk = r.rank()
        z = (rk - rk.mean()) / rk.std()
        sig = ic * z + np.sqrt(1 - ic**2) * pd.Series(rng.standard_normal(len(z)), index=z.index)
        picks[d] = list(sig.nlargest(n).index)
    return picks


def run_selection(px, eligible, mes, sizes: list[int], seeds: int) -> pd.DataFrame:
    fwd = forward_returns(px, eligible, mes)
    rows = []
    for n in sizes:
        for ic in ICS:
            res = [stats(T5.run_portfolio(px, picks_with_ic(fwd, ic, n, np.random.default_rng(s)))) for s in range(seeds if ic < 1 else 1)]
            cagr = float(np.median([x[0] for x in res])); mdd = float(np.median([x[1] for x in res]))
            rows.append({"n": n, "ic": ic, "cagr": cagr, "mdd": mdd, "cagr_min": min(x[0] for x in res), "cagr_max": max(x[0] for x in res),
                         "mdd_worst": min(x[1] for x in res), "runs": len(res)})
            print(f"top{n:>2} IC={ic:.2f}  CAGR {cagr*100:6.1f}%  MDD {mdd*100:6.1f}%", flush=True)
    return pd.DataFrame(rows)


def run_timing(qqq: pd.Series, irx: pd.Series, mes: list[pd.Timestamp], seeds: int) -> pd.DataFrame:
    qme = qqq.reindex(mes).dropna()
    nxt = qme.pct_change().shift(-1).dropna()  # next-month QQQ return known at each month end
    cash = (irx.reindex(nxt.index).ffill().fillna(0) / 100) / 12
    up = (nxt > 0).to_numpy()
    rows = []
    for p in HITS:
        res = []
        for s in range(seeds if p < 1 else 1):
            correct = np.random.default_rng(s).random(len(nxt)) < p
            hold = np.where(correct, up, ~up)
            nav = pd.Series(np.cumprod(np.where(hold, nxt.to_numpy(), cash.to_numpy()) + 1), index=nxt.index)
            res.append(stats(nav))
        cagr = float(np.median([x[0] for x in res])); mdd = float(np.median([x[1] for x in res]))
        rows.append({"hit": p, "cagr": cagr, "mdd": mdd, "cagr_min": min(x[0] for x in res), "cagr_max": max(x[0] for x in res),
                     "mdd_worst": min(x[1] for x in res), "runs": len(res)})
        print(f"QQQ timing hit={p:.2f}  CAGR {cagr*100:6.1f}%  MDD {mdd*100:6.1f}%", flush=True)
    bh = stats(qme / qme.iloc[0])
    rows.append({"hit": np.nan, "cagr": bh[0], "mdd": bh[1], "cagr_min": bh[0], "cagr_max": bh[0], "mdd_worst": bh[1], "runs": 1})
    return pd.DataFrame(rows)


def chart(sel: pd.DataFrame, tim: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.suptitle("要多準才夠：注入已知程度的「預知」後，2000 至 2024 年的 CAGR 與最深回撤（目標：CAGR 30% 以上、MDD 30% 以內）", x=0.01, ha="left", fontsize=13)

    ax = axes[0]
    for i, ((n, g), col) in enumerate(zip(sel.groupby("n"), pal)):
        g = g[g.ic < 1].sort_values("ic")
        ax.plot(-g.mdd * 100, g.cagr * 100, "-o", color=col, lw=1.6, ms=5, label=f"每月前 {n} 隻")
        for _, r in g.iterrows():
            ax.annotate(f"IC {r.ic:.2f}", (-r.mdd * 100, r.cagr * 100), xytext=(6, 4 if i == 0 else -11), textcoords="offset points", fontsize=8, color=col)
    ax.axhline(30, color="#555555", ls=":", lw=1); ax.axvline(30, color="#555555", ls=":", lw=1)
    ax.axvspan(0, 30, ymin=0, ymax=1, color="#1baf7a", alpha=0.06)
    ax.set_xlabel("最深回撤（%）"); ax.set_ylabel("CAGR（%）")
    ax.set_title("選股：訊號 = IC × 下月真實回報 + 雜訊（IC 1.0 = 全知，未畫）", loc="left", fontsize=11)
    ax.set_xlim(0, 75); ax.set_ylim(0, 160)
    ax.text(2, 100, "目標區：MDD < 30%", fontsize=9, color="#1baf7a")
    ax.text(2, 32, "目標線 CAGR 30%", fontsize=9, color="#555555")
    ax.legend(frameon=False, loc="upper right"); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    t = tim[tim.hit.notna()].sort_values("hit")
    ax.plot(-t.mdd * 100, t.cagr * 100, "-o", color=pal[1], lw=1.6, ms=5, label="QQQ 月底全進全出")
    for _, r in t.iterrows():
        ax.annotate(f"命中 {r.hit*100:.0f}%", (-r.mdd * 100, r.cagr * 100), xytext=(5, 3), textcoords="offset points", fontsize=8, color=pal[1])
    bh = tim[tim.hit.isna()].iloc[0]
    ax.plot(-bh.mdd * 100, bh.cagr * 100, "s", color="#999999", ms=7, label="QQQ 買進持有")
    ax.annotate("買進持有", (-bh.mdd * 100, bh.cagr * 100), xytext=(-60, 6), textcoords="offset points", fontsize=8, color="#777777")
    ax.axhline(30, color="#555555", ls=":", lw=1); ax.axvline(30, color="#555555", ls=":", lw=1)
    ax.axvspan(0, 30, color="#1baf7a", alpha=0.06)
    ax.set_xlabel("最深回撤（%）"); ax.set_ylabel("CAGR（%）")
    ax.set_title("擇時：猜對下月 QQQ 方向的命中率（50% = 亂猜，均線規則約 55% 到 60%）", loc="left", fontsize=11)
    ax.set_xlim(0, 85); ax.set_ylim(0, 45)
    ax.legend(frameon=False, loc="upper right"); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="5,20")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--timing-seeds", type=int, default=20)
    p.add_argument("--out", default="sp500_skill_needed")
    p.add_argument("--chart", default="docs/skill_needed.png")
    p.add_argument("--chart-only", action="store_true")
    args = p.parse_args(argv)
    if args.chart_only:
        chart(pd.read_csv(f"{args.out}_selection.csv"), pd.read_csv(f"{args.out}_timing.csv"), args.chart)
        return 0
    px, member, eligible, spy = load_universe()
    mes = [d for d in T5.month_ends(px.index) if d >= pd.Timestamp("2000-02-29")]
    sel = run_selection(px, eligible, mes, [int(x) for x in args.sizes.split(",")], args.seeds)
    qqq = pd.read_csv("qqq_full_cache.csv", index_col=0, parse_dates=True).squeeze().sort_index().reindex(px.index).ffill()
    irx = pd.read_csv("irx_cache.csv", index_col=0, parse_dates=True).squeeze().sort_index().reindex(px.index).ffill()
    tim = run_timing(qqq, irx, mes, args.timing_seeds)
    sel.to_csv(f"{args.out}_selection.csv", index=False, encoding="utf-8-sig")
    tim.to_csv(f"{args.out}_timing.csv", index=False, encoding="utf-8-sig")
    chart(sel, tim, args.chart)
    print(f"wrote {args.out}_selection.csv, {args.out}_timing.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
