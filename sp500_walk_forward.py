#!/usr/bin/env python3
"""
Walk-forward (out-of-sample) test of the cash-allowed QQQ timing rules.

Sections 9.x tuned the rule parameters on the whole 2000-2026 sample, which is in-sample.
This script measures the overfitting haircut and parameter stability with anchored
(expanding-window) walk-forward re-selection:

  Scheme A  three blocks: IS 2000-03..2009-12 -> OOS 2010-2014
                          IS 2000-03..2014-12 -> OOS 2015-2019
                          IS 2000-03..2019-12 -> OOS 2020-01..end
  Scheme B  annual: for every year Y in 2010..end, IS 2000-03..(Y-1)-12 -> OOS year Y

Rule families and grids (weights clipped to [0, 1], NaN -> 0; weight decided at close t-1
is applied to the return t-1 -> t; 0.1% cost per unit of weight change; cash earns IRX):

  vt      min(1, T / W-day realised vol)                 T in {15..40%}, W in {10,21,42,63,126}
  ratio   1 unless S-day vol > K x L-day vol, then T/vol  K in {1.2..2.0}, (S,L) in {(5,42),(10,63),(21,126)}, T in {20%,30%}
  combo   min(ratio rule at 30%, vt 30%/21d with floor F) ratio grid x F in {0.3,0.5,0.7}
  sma     1 if close > SMA(N) else 0                       N in {100..300}, evaluated daily or at month end
  sma_dd  1 if close > SMA200 and drawdown from 252d high within band   band in {10%,15%,20%}

For each family and IS window the parameter set with the best in-sample Sharpe (primary) or
in-sample CAGR (secondary) is chosen; its weights are applied over the OOS block, and the
OOS blocks are stitched into one continuous NAV from --oos-start to --end. The first weight
of each block is the one decided at the close of the last IS day (data up to the IS end), so
no OOS return is ever governed by a parameter chosen with data after the block start; this
is asserted. References over the same OOS period: QQQ and SPY buy-and-hold, each family's
full-sample-optimised parameters (chosen on 2000-03..end, the in-sample cheat) and the fixed
parameters used in earlier sections.

Haircut = full-sample-optimised CAGR/Sharpe over the OOS period minus walk-forward OOS value.
Paired random-entry Monte Carlo (same draws for every curve, holds 1-10 years) is run inside
the OOS period.

Outputs: <out>_params.csv, <out>_equity.csv, <out>_summary.csv, docs/walk_forward.png
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

import sp500_beat_qqq as B
import sp500_optimize_qqq2 as O2
import sp500_random_entry_mc as MC

COST = 0.001
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
FAMILIES = ["vt", "ratio", "combo", "sma", "sma_dd"]
FIXED = {  # label -> (family, params) used in earlier sections
    "fixed_vt_30_21": ("vt", (0.30, 21)),
    "fixed_vt_25_21": ("vt", (0.25, 21)),
    "fixed_vt_20_63": ("vt", (0.20, 63)),
    "fixed_ratio_1.5_10_63_30": ("ratio", (1.5, 10, 63, 0.30)),
    "fixed_combo_1.5_10_63_f0.5": ("combo", (1.5, 10, 63, 0.5)),
    "fixed_sma_200_daily": ("sma", (200, "daily")),
    "fixed_sma_200_monthly": ("sma", (200, "monthly")),
    "fixed_sma_200_dd15": ("sma_dd", (0.15,)),
}


def fmt_params(family: str, p: tuple) -> str:
    if family == "vt":
        return f"T={int(round(p[0] * 100))}% W={p[1]}"
    if family == "ratio":
        return f"K={p[0]} S={p[1]} L={p[2]} T={int(round(p[3] * 100))}%"
    if family == "combo":
        return f"K={p[0]} S={p[1]} L={p[2]} floor={p[3]}"
    if family == "sma":
        return f"N={p[0]} {p[1]}"
    return f"band={int(round(p[0] * 100))}%"


def build_candidates(q: pd.Series) -> dict[str, dict[tuple, np.ndarray]]:
    """Return {family: {params: weight array over q.index}} (weights in [0,1], NaN -> 0)."""
    idx = q.index
    r = q.pct_change()
    rv = {W: (r.rolling(W).std() * math.sqrt(252)).to_numpy() for W in (5, 10, 21, 42, 63, 126)}
    sma = {N: q.rolling(N).mean().to_numpy() for N in (100, 150, 200, 250, 300)}
    dd = (q / q.rolling(252).max() - 1).to_numpy()
    month_end = (pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("last") == pd.Series(idx, index=idx)).to_numpy()
    qa = q.to_numpy()

    def clean(w):
        return np.nan_to_num(np.clip(w, 0.0, 1.0), nan=0.0)

    def vt(T, W):
        with np.errstate(divide="ignore", invalid="ignore"):
            return clean(T / rv[W])

    def ratio(K, S, L, T):
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(rv[S] > K * rv[L], T / rv[S], 1.0)
        w[np.isnan(rv[L])] = np.nan  # warm-up -> 0
        return clean(w)

    def combo(K, S, L, F):
        with np.errstate(divide="ignore", invalid="ignore"):
            base = np.clip(0.30 / rv[21], F, 1.0)
        return clean(np.minimum(ratio(K, S, L, 0.30), base))

    def sma_rule(N, mode):
        sig = (qa > sma[N]).astype(float)
        sig[np.isnan(sma[N])] = 0.0
        if mode == "monthly":
            sig = pd.Series(np.where(month_end, sig, np.nan), index=idx).ffill().fillna(0.0).to_numpy()
        return sig

    def sma_dd(band):
        w = ((qa > sma[200]) & (dd > -band)).astype(float)
        w[np.isnan(sma[200]) | np.isnan(dd)] = 0.0
        return w

    cands: dict[str, dict[tuple, np.ndarray]] = {f: {} for f in FAMILIES}
    for T in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        for W in (10, 21, 42, 63, 126):
            cands["vt"][(T, W)] = vt(T, W)
    for K in (1.2, 1.3, 1.5, 1.75, 2.0):
        for S, L in ((5, 42), (10, 63), (21, 126)):
            for T in (0.20, 0.30):
                cands["ratio"][(K, S, L, T)] = ratio(K, S, L, T)
            for F in (0.3, 0.5, 0.7):
                cands["combo"][(K, S, L, F)] = combo(K, S, L, F)
    for N in (100, 150, 200, 250, 300):
        for mode in ("daily", "monthly"):
            cands["sma"][(N, mode)] = sma_rule(N, mode)
    for band in (0.10, 0.15, 0.20):
        cands["sma_dd"][(band,)] = sma_dd(band)
    return cands


def candidate_returns(W: np.ndarray, ret: np.ndarray, cash: np.ndarray, cost: float = COST) -> np.ndarray:
    """Daily portfolio returns (T x N) for weight matrix W, same convention as B.etf_timing."""
    w = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])  # weights[t-1] applied to return t
    w_prev = np.vstack([np.zeros((1, W.shape[1])), w[:-1]])
    return w * ret[:, None] + (1 - w) * cash[:, None] - np.abs(w - w_prev) * cost


def window_scores(R: np.ndarray, idx: pd.DatetimeIndex, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """In-sample Sharpe and CAGR per candidate for NAV normalised at position a, measured to position b."""
    r = R[a + 1 : b + 1]
    sharpe = r.mean(axis=0) / r.std(axis=0, ddof=1) * math.sqrt(252)
    yrs = (idx[b] - idx[a]).days / 365.25
    growth = np.prod(1 + r, axis=0)
    return sharpe, growth ** (1 / yrs) - 1


def blocks_scheme_a(idx: pd.DatetimeIndex, start: pd.Timestamp, oos_start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    edges = [oos_start, pd.Timestamp("2015-01-01"), pd.Timestamp("2020-01-01"), end + pd.Timedelta(days=1)]
    out = []
    for i in range(3):
        os_, oe = edges[i], edges[i + 1] - pd.Timedelta(days=1)
        out.append(dict(is_start=start, is_end=os_ - pd.Timedelta(days=1), oos_start=os_, oos_end=min(oe, end)))
    return out


def blocks_scheme_b(idx: pd.DatetimeIndex, start: pd.Timestamp, oos_start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    out = []
    for y in range(oos_start.year, end.year + 1):
        os_ = max(pd.Timestamp(y, 1, 1), oos_start)
        oe = min(pd.Timestamp(y, 12, 31), end)
        out.append(dict(is_start=start, is_end=os_ - pd.Timedelta(days=1), oos_start=os_, oos_end=oe))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2000-03-01", help="in-sample start (anchor)")
    p.add_argument("--oos-start", default="2010-01-01", help="first out-of-sample day")
    p.add_argument("--end", default="2026-10-16")
    p.add_argument("--n", type=int, default=3000, help="random-entry Monte Carlo draws")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="sp500_walk_forward")
    args = p.parse_args(argv)

    qqq, spy, irx = O2.load("qqq"), O2.load("spy"), O2.load("irx")
    idx = qqq.loc[: args.end].index
    etf = pd.DataFrame({"QQQ": qqq, "SPY": spy}).reindex(idx).ffill()
    cash_daily = (irx.reindex(idx).ffill() / 100 / 252).fillna(0.0)
    zero = pd.DataFrame(0.0, index=idx, columns=["QQQ", "SPY"])
    q = etf["QQQ"]
    start, oos_start, end = pd.Timestamp(args.start), pd.Timestamp(args.oos_start), pd.Timestamp(args.end)
    pos_start = idx.searchsorted(start)
    pos_oos = idx.searchsorted(oos_start)
    pos_end = len(idx) - 1
    oos_first_day = idx[pos_oos]

    cands = build_candidates(q)
    ret = q.pct_change().fillna(0.0).to_numpy()
    cash = cash_daily.to_numpy()
    fam_keys = {f: list(cands[f].keys()) for f in FAMILIES}
    fam_W = {f: np.column_stack([cands[f][k] for k in fam_keys[f]]) for f in FAMILIES}
    fam_R = {f: candidate_returns(fam_W[f], ret, cash) for f in FAMILIES}
    print(f"candidates: {sum(len(v) for v in fam_keys.values())} ({', '.join(f'{f}={len(fam_keys[f])}' for f in FAMILIES)})")

    def pick(family: str, a: int, b: int) -> dict:
        sh, cg = window_scores(fam_R[family], idx, a, b)
        return {"sharpe": (fam_keys[family][int(np.argmax(sh))], sh.max(), cg[int(np.argmax(sh))]),
                "cagr": (fam_keys[family][int(np.argmax(cg))], sh[int(np.argmax(cg))], cg.max())}

    def nav_from_weights(w: np.ndarray) -> pd.Series:
        nav = B.etf_timing(etf, zero.assign(QQQ=w), cash_daily, COST)
        nav = nav.iloc[pos_oos:]
        return nav / nav.iloc[0]

    schemes = {"A": blocks_scheme_a(idx, start, oos_start, end), "B": blocks_scheme_b(idx, start, oos_start, end)}
    curves: dict[str, pd.Series] = {}
    param_rows = []
    picks_b: dict[tuple, list] = {}

    # --- walk-forward ---------------------------------------------------------------
    for sname, blocks in schemes.items():
        for family in FAMILIES:
            stitched = {"sharpe": np.full(len(idx), np.nan), "cagr": np.full(len(idx), np.nan)}
            chosen = {"sharpe": [], "cagr": []}
            for blk in blocks:
                a = pos_start
                b = idx.searchsorted(blk["is_end"], side="right") - 1  # last trading day <= IS end
                os_pos = idx.searchsorted(blk["oos_start"])
                oe_pos = idx.searchsorted(blk["oos_end"], side="right") - 1
                assert idx[b] < blk["oos_start"] <= idx[os_pos], "IS window must end before the OOS block"
                res = pick(family, a, b)
                for crit in ("sharpe", "cagr"):
                    params, is_sh, is_cg = res[crit]
                    # weights[t-1] governs return t: the block's first governing weight is dated the last IS day
                    stitched[crit][os_pos - 1 : oe_pos] = cands[family][params][os_pos - 1 : oe_pos]
                    chosen[crit].append((os_pos, oe_pos, params, b))
                    param_rows.append({"scheme": sname, "family": family, "criterion": crit, "is_start": blk["is_start"].date(), "is_end": idx[b].date(),
                                       "oos_block": f"{blk['oos_start'].date()}..{idx[oe_pos].date()}", "params": fmt_params(family, params),
                                       "is_sharpe": round(float(is_sh), 3), "is_cagr_pct": round(float(is_cg) * 100, 2)})
                    if sname == "B":
                        picks_b.setdefault((family, crit), []).append((blk["oos_start"].year, params))
            for crit in ("sharpe", "cagr"):
                w = stitched[crit]
                # before the OOS period use the first block's choice (chosen with data to the first IS end), so
                # the starting weight is the one that rule would actually hold on the last IS day
                first_params = chosen[crit][0][2]
                w[: pos_oos - 1] = cands[family][first_params][: pos_oos - 1]
                w[pos_end] = cands[family][chosen[crit][-1][2]][pos_end]  # last day's weight (unused by any return)
                assert not np.isnan(w).any()
                # validation: every OOS weight comes from the candidate chosen with IS data ending before that block
                for os_pos, oe_pos, params, b in chosen[crit]:
                    assert b < os_pos
                    assert np.array_equal(w[os_pos - 1 : oe_pos], cands[family][params][os_pos - 1 : oe_pos])
                curves[f"{family}_wf{sname}_{crit}"] = nav_from_weights(w)

    # --- hand check of one block (Scheme A, block 1, vt by Sharpe) --------------------
    blk = schemes["A"][0]
    b = idx.searchsorted(blk["is_end"], side="right") - 1
    res = pick("vt", pos_start, b)
    T, W = res["sharpe"][0]
    print(f"\nhand check, Scheme A block 1: IS {idx[pos_start].date()}..{idx[b].date()} best vt by Sharpe = T={T:.2f} W={W} "
          f"(IS Sharpe {res['sharpe'][1]:.3f}, IS CAGR {res['sharpe'][2]*100:.2f}%)")
    rv_manual = (q.pct_change().rolling(W).std() * math.sqrt(252))
    w_manual = (T / rv_manual).clip(0, 1).fillna(0.0)
    os_pos, oe_pos = idx.searchsorted(blk["oos_start"]), idx.searchsorted(blk["oos_end"], side="right") - 1
    wf_nav = curves["vt_wfA_sharpe"]
    w_check = B.etf_timing(etf, zero.assign(QQQ=w_manual.to_numpy()), cash_daily, COST)
    manual_block = w_check.iloc[os_pos:oe_pos + 1] / w_check.iloc[os_pos]
    wf_block = wf_nav.loc[idx[os_pos]: idx[oe_pos]] / wf_nav.loc[idx[os_pos]]
    print(f"  OOS {idx[os_pos].date()}..{idx[oe_pos].date()}: NAV from hand-built vt({T:.2f},{W}) weights = {manual_block.iloc[-1]:.4f}, "
          f"stitched walk-forward NAV over same block = {wf_block.iloc[-1]:.4f}, max |diff| = {(manual_block - wf_block).abs().max():.2e}")
    assert (manual_block - wf_block).abs().max() < 1e-9

    # --- references ------------------------------------------------------------------
    curves["QQQ_buy_hold"] = nav_from_weights(np.ones(len(idx)))
    spy_nav = etf["SPY"].iloc[pos_oos:]
    curves["SPY_buy_hold"] = spy_nav / spy_nav.iloc[0]
    full_pick = {}
    for family in FAMILIES:
        res = pick(family, pos_start, pos_end)
        for crit in ("sharpe", "cagr"):
            params, is_sh, is_cg = res[crit]
            full_pick[(family, crit)] = params
            curves[f"{family}_full_{crit}"] = nav_from_weights(cands[family][params])
            param_rows.append({"scheme": "full", "family": family, "criterion": crit, "is_start": start.date(), "is_end": idx[pos_end].date(),
                               "oos_block": "none (in-sample)", "params": fmt_params(family, params), "is_sharpe": round(float(is_sh), 3), "is_cagr_pct": round(float(is_cg) * 100, 2)})
    for label, (family, params) in FIXED.items():
        curves[label] = nav_from_weights(cands[family][params])

    eq = pd.DataFrame(curves)
    eq = eq / eq.iloc[0]
    eq.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")
    pd.DataFrame(param_rows).to_csv(f"{args.out}_params.csv", index=False, encoding="utf-8-sig")

    # --- summary with paired random-entry MC inside the OOS period ---------------------
    rng = np.random.default_rng(args.seed)
    st, en, yr = MC.draw_windows(eq.index, args.n, 1, 10, rng)
    qn = eq["QQQ_buy_hold"].to_numpy()
    gq, cq, _ = MC.window_stats(qn, st, en, yr)
    q_cagr = B.cagr(eq["QQQ_buy_hold"]) * 100
    rows = []
    for c in eq.columns:
        m = B.metrics(eq[c])
        g, cg, _ = MC.window_stats(eq[c].to_numpy(), st, en, yr)
        ex = (cg - cq) * 100
        parts = c.split("_")
        if c.startswith("fixed_"):
            fam, kind, crit = FIXED[c][0], "fixed", ""
        elif c.endswith("_buy_hold"):
            fam, kind, crit = parts[0], "benchmark", ""
        else:
            crit = parts[-1]
            kind = parts[-2]
            fam = "_".join(parts[:-2])
        rows.append({"curve": c, "family": fam, "kind": kind, "criterion": crit, "cagr_pct": m["cagr_pct"], "ann_vol_pct": m["ann_vol_pct"], "sharpe": m["sharpe"],
                     "max_dd_pct": m["max_dd_pct"], "worst_year_pct": m["worst_year_pct"], "excess_cagr_vs_qqq": round(m["cagr_pct"] - q_cagr, 2),
                     "mc_win_pct_vs_qqq": round(float((g > gq).mean() * 100), 1), "mc_median_cagr": round(float(np.median(cg)) * 100, 2),
                     "mc_p10_cagr": round(float(np.percentile(cg, 10)) * 100, 2), "mc_worst_cagr": round(float(cg.min()) * 100, 2),
                     "mc_median_excess": round(float(np.median(ex)), 2)})
    summary = pd.DataFrame(rows)
    # haircut: full-sample optimised (same criterion) minus walk-forward OOS
    full = summary[summary.kind == "full"].set_index(["family", "criterion"])
    hc_c, hc_s = [], []
    for _, r_ in summary.iterrows():
        if r_.kind in ("wfA", "wfB"):
            f_ = full.loc[(r_.family, r_.criterion)]
            hc_c.append(round(f_.cagr_pct - r_.cagr_pct, 2)); hc_s.append(round(f_.sharpe - r_.sharpe, 2))
        else:
            hc_c.append(np.nan); hc_s.append(np.nan)
    summary["haircut_cagr_pp"] = hc_c
    summary["haircut_sharpe"] = hc_s
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 320)
    print(f"\n=== {eq.index[0].date()}..{eq.index[-1].date()} ===")
    print(summary.drop(columns=["family", "kind", "criterion"]).to_string(index=False))
    print("\n=== Scheme B annual picks (Sharpe criterion) ===")
    for family in FAMILIES:
        seq = picks_b[(family, "sharpe")]
        changes = sum(1 for i in range(1, len(seq)) if seq[i][1] != seq[i - 1][1])
        print(f"{family:7s} changes={changes:2d}  " + "; ".join(f"{y}:{fmt_params(family, pp)}" for y, pp in seq))
    print("\n=== Scheme B annual picks (CAGR criterion) ===")
    for family in FAMILIES:
        seq = picks_b[(family, "cagr")]
        changes = sum(1 for i in range(1, len(seq)) if seq[i][1] != seq[i - 1][1])
        print(f"{family:7s} changes={changes:2d}  " + "; ".join(f"{y}:{fmt_params(family, pp)}" for y, pp in seq))
    print("\nfull-sample picks: " + "; ".join(f"{f}/{c}: {fmt_params(f, pp)}" for (f, c), pp in full_pick.items()))

    chart(eq, picks_b, "docs/walk_forward.png")
    print(f"\nwrote {args.out}_params.csv, {args.out}_equity.csv, {args.out}_summary.csv, docs/walk_forward.png")
    return 0


def chart(eq: pd.DataFrame, picks_b: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    ink, muted, grid_c, bg = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    fig = plt.figure(figsize=(13, 9.6), facecolor=bg)
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.2, 1], hspace=0.38, wspace=0.3, left=0.06, right=0.84, top=0.94, bottom=0.08)
    ax = fig.add_subplot(gs[0, :])
    lines = {"QQQ_buy_hold": ("QQQ 買進持有", "#52514e", 2.0, "-"), "SPY_buy_hold": ("SPY 買進持有", "#9a9890", 1.6, "--")}
    ax.axhline(1.0, color="#c3c2b7", lw=0.8)
    names = {"vt": "vt 波動目標", "ratio": "ratio 尖峰", "combo": "combo 組合", "sma": "sma 均線", "sma_dd": "sma_dd 均線+回撤"}
    for f, col in zip(FAMILIES, PALETTE):
        lines[f"{f}_wfA_sharpe"] = (f"{names[f]}·走動A", col, 1.6, "-")
    labels = []
    for c, (lab, col, lw, ls) in lines.items():
        s = eq[c]
        ax.plot(s.index, s.values, color=col, lw=lw, ls=ls)
        labels.append((math.log(s.iloc[-1]), f"{lab}  {s.iloc[-1]:.1f}x", col))
    # direct labels at line ends, pushed apart in log space
    labels.sort(key=lambda t: t[0])
    gap = 0.15
    ys = [t[0] for t in labels]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    x_end = eq.index[-1]
    for (y0, lab, col), y in zip(labels, ys):
        ax.annotate(lab, (x_end, math.exp(y)), xytext=(8, 0), textcoords="offset points", fontsize=8.5, color=col, va="center", annotation_clip=False)
    ax.set_yscale("log")
    ax.set_title(f"走動式樣本外測試：{eq.index[0].year}–{eq.index[-1].year} 淨值（對數座標，起點 = 1；每個區塊的參數只用區塊開始前的資料挑選）", fontsize=11, loc="left", color=ink)
    ax.set_ylabel("淨值（倍）", color=muted, fontsize=9)
    ax.grid(True, which="major", color=grid_c, lw=0.8)
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels(["1x", "2x", "4x", "8x", "16x"])
    ax.set_xlim(eq.index[0], eq.index[-1])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=9)

    # bottom: Scheme B annual picks as step plots (one measure per panel, no dual axes)
    panels = [("vt", lambda p: p[0] * 100, "vt 目標波動 T（%）", PALETTE[0], (10, 45)),
              ("vt", lambda p: p[1], "vt 波動視窗 W（日）", PALETTE[0], (0, 140)),
              ("ratio", lambda p: p[0], "ratio 尖峰倍數 K", PALETTE[1], (1.0, 2.2))]
    for j, (fam, fn, title, col, ylim) in enumerate(panels):
        axb = fig.add_subplot(gs[1, j])
        for crit, ls, alpha, lab in (("sharpe", "-", 1.0, "依 IS Sharpe"), ("cagr", ":", 0.8, "依 IS CAGR")):
            seq = picks_b[(fam, crit)]
            yrs = [y for y, _ in seq] + [seq[-1][0] + 1]
            vals = [fn(pp) for _, pp in seq] + [fn(seq[-1][1])]
            axb.step(yrs, vals, where="post", color=col, lw=1.8, ls=ls, alpha=alpha, label=lab)
        axb.set_title(f"每年重選（走動 B）：{title}", fontsize=9.5, loc="left", color=ink)
        axb.set_ylim(*ylim)
        axb.set_xlim(yrs[0], yrs[-1])
        axb.set_xticks(list(range(yrs[0], yrs[-1] + 1, 4)))
        axb.grid(True, color=grid_c, lw=0.8)
        for sp in ("top", "right"):
            axb.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axb.spines[sp].set_color("#c3c2b7")
        axb.tick_params(colors=muted, labelsize=8.5)
        if j == 0:
            axb.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.06, 0.015, "實線 = 依樣本內 Sharpe 挑選；點線 = 依樣本內 CAGR 挑選。每年的參數只用 2000-03 到前一年年底的資料決定。", fontsize=8.5, color=muted)
    fig.savefig(path, dpi=150, facecolor=bg)


if __name__ == "__main__":
    sys.exit(main())
