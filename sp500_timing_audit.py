#!/usr/bin/env python3
"""
Forward audit of month-end market-timing rules: how often do they call the NEXT month's direction? (section 19, step 1)

Section 19.2 / 19.3 price timing in one number, the hit rate on next-month direction (55-60% for SMA rules, 70%
needed next to a selection IC of 0.15, ~85% for 30% CAGR with MDD under 30%). This script measures that number
directly for ~30 published rule families instead of inferring it from equity curves.

At every month end t (signal dates 1999-12 .. 2024-11, so the target months are 2000-01 .. 2024-12, n = 300;
sub-period target months 2010-01 .. 2024-12, n = 180) each rule emits a call, up or down, from data <= t only.
The call is scored against the sign of the target index's return from the execution close (first trading day
after t) to the next month's execution close, separately for SPY and QQQ. Index-based rules are computed on the
target index itself; breadth rules use the point-in-time S&P 500 universe of section 15; cross-asset rules compare
the target index with IEF / GLD and the 3-month T-bill (^IRX).

Reported per rule x target x period: n, hit rate, 95% Wilson interval, hit rate on up-months (sensitivity) and
on down-months (specificity), share of months called up, base rate of up-months, exact two-sided binomial
p-values vs 50% and vs the base rate, months needed to separate the observed rate from 50% at 95% confidence
(n = (1.96 * 0.5 / (p - 0.5))^2, section 19.8), and the CAGR / MDD of the month-end switch (hold the index when
called up, otherwise T-bills at IRX/100/12 per month, no cost, as in section 19.2) next to buy-and-hold over the
same months. Pre-registered thresholds, fixed before running: "real timing signal" = Wilson lower bound > 50%
AND point estimate >= 60%; "grid-passing" = hit rate >= 70% (section 19.3).

Null: 200 coin-flip callers (seed 42) and 200 "always-up with random noise" callers (per-caller flip probability
uniform in [0, 0.5]) scored on the same months; 95th / 99th percentiles of their hit rates are reported per
target and period, and the number of real rules above them.

Usage: python sp500_timing_audit.py [--start 2000-01-01] [--end 2024-12-31] [--null-runs 200] [--seed 42]
                                    [--out timing_audit] [--chart docs/timing_audit.png] [--chart-only]
Outputs: <out>_rules.csv, <out>_null.csv, <out>_summary.csv, <chart>
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import sp500_top5_monthly as T5
from sp500_precursor_events import load_universe

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREY = "#8a8a8a"
LIGHT = "#cfcfcf"
REAL_SIGNAL_MIN = 0.60   # pre-registered: point estimate >= 60% and Wilson lower bound > 50%
GRID_PASS_MIN = 0.70     # pre-registered: section 19.3 requirement next to IC 0.15
ENSEMBLE_PARTS = ["sma200", "mom_12_1", "breadth200_50"]

RULE_LABELS = {
    "always_up": "永遠看多（基準）",
    "sma50": "收盤 > 50 日線", "sma100": "收盤 > 100 日線", "sma150": "收盤 > 150 日線",
    "sma200": "收盤 > 200 日線", "sma250": "收盤 > 250 日線",
    "sma200_slope": "200 日線高於 21 日前",
    "mom_3m": "3 個月報酬 > 0", "mom_6m": "6 個月報酬 > 0", "mom_12m": "12 個月報酬 > 0",
    "mom_12_1": "12-1 動能 > 0", "dual_mom_12m": "12 個月報酬 > 國庫券",
    "vol21_lt_vol63": "21 日波動 < 63 日波動", "vol_spike": "無波動尖峰（10 日 < 1.5 × 63 日）",
    "vol21_lt_med252": "21 日波動 < 252 日中位數",
    "dd_10": "距 252 日高點 10% 內", "dd_15": "距 252 日高點 15% 內",
    "breadth200_50": "成份股 > 200 日線比例 > 50%", "breadth200_60": "成份股 > 200 日線比例 > 60%",
    "breadth_rising": "廣度高於 21 日前", "breadth_pos63": "63 日報酬為正的成份股 > 50%",
    "xa_ief": "指數 63 日報酬 > IEF", "xa_gld": "指數 63 日報酬 > GLD", "tbill_falling": "國庫券利率低於 63 日前",
    "ens_majority": "三取二：200 日線 / 12-1 / 廣度", "ens_all3": "三者皆多：200 日線 / 12-1 / 廣度",
    "ens_any1": "三者任一多：200 日線 / 12-1 / 廣度",
    "ens_sma_mom": "兩者皆多：200 日線 + 12-1", "ens_sma_breadth": "兩者皆多：200 日線 + 廣度",
    "ens_mom_breadth": "兩者皆多：12-1 + 廣度",
}
RULE_GROUP = {k: ("基準" if k == "always_up" else "均線" if k.startswith("sma") else "動能" if k.startswith("mom") or k.startswith("dual")
                  else "波動" if k.startswith("vol") else "回撤" if k.startswith("dd") else "廣度" if k.startswith("breadth")
                  else "跨資產" if k.startswith("xa") or k.startswith("tbill") else "組合") for k in RULE_LABELS}


# ----------------------------------------------------------------------------------------------- data

def read_series(path: str) -> pd.Series:
    return pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns").sort_index().astype(float)


def load_spy() -> pd.Series:
    """spy_long_cache (from 1998, for the 1999 SMAs) extended with spy_full_cache returns past 2024-12-31."""
    long = read_series("spy_long_cache.csv")
    full = read_series("spy_full_cache.csv")
    tail = full.loc[full.index > long.index[-1]]
    if len(tail):
        tail = tail * (long.iloc[-1] / full.loc[long.index[-1]])
        long = pd.concat([long, tail])
    return long


def tbill_index(irx: pd.Series) -> pd.Series:
    """Cumulative T-bill total return index from the daily ^IRX yield (percent, 252 trading days a year)."""
    return (1 + irx.fillna(0) / 100 / 252).cumprod()


# --------------------------------------------------------------------------------------------- signals

def breadth_series(px: pd.DataFrame, member: pd.DataFrame) -> dict[str, pd.Series]:
    """Daily breadth of the point-in-time universe: share above own 200-day SMA, share with positive 63-day return."""
    sma200 = px.rolling(200).mean()
    mask = member & px.notna() & sma200.notna()
    above = (px > sma200).astype(float).where(mask)
    cnt = mask.sum(axis=1).astype(float).replace(0.0, np.nan)
    share200 = above.sum(axis=1) / cnt
    r63 = px / px.shift(63) - 1
    mask63 = member & r63.notna()
    pos63 = (r63 > 0).astype(float).where(mask63).sum(axis=1) / mask63.sum(axis=1).astype(float).replace(0.0, np.nan)
    return {"share200": share200.where(mask.sum(axis=1) >= 100), "pos63": pos63.where(mask63.sum(axis=1) >= 100)}


def rule_signals(p: pd.Series, irx: pd.Series, tb: pd.Series, ief: pd.Series, gld: pd.Series, br: dict[str, pd.Series]) -> pd.DataFrame:
    """Boolean (nullable) daily frame: True = call next month up. Every column uses only data up to and including the row's date."""
    s = {}
    for k in (50, 100, 150, 200, 250):
        s[f"sma{k}"] = p > p.rolling(k).mean()
    sma200 = p.rolling(200).mean()
    s["sma200_slope"] = sma200 > sma200.shift(21)
    s["mom_3m"] = p / p.shift(63) > 1
    s["mom_6m"] = p / p.shift(126) > 1
    s["mom_12m"] = p / p.shift(252) > 1
    s["mom_12_1"] = p.shift(21) / p.shift(252) > 1
    s["dual_mom_12m"] = (p / p.shift(252) - 1) > (tb / tb.shift(252) - 1)
    r = np.log(p).diff()
    v10, v21, v63 = r.rolling(10).std(), r.rolling(21).std(), r.rolling(63).std()
    s["vol21_lt_vol63"] = v21 < v63
    s["vol_spike"] = ~(v10 > 1.5 * v63)  # section 13 spike rule: spike = down, otherwise up
    s["vol21_lt_med252"] = v21 < v21.rolling(252).median()
    hi252 = p.rolling(252).max()
    s["dd_10"] = p / hi252 - 1 > -0.10
    s["dd_15"] = p / hi252 - 1 > -0.15
    s["breadth200_50"] = br["share200"] > 0.5
    s["breadth200_60"] = br["share200"] > 0.6
    s["breadth_rising"] = br["share200"] > br["share200"].shift(21)
    s["breadth_pos63"] = br["pos63"] > 0.5
    s["xa_ief"] = (p / p.shift(63)) > (ief / ief.shift(63))
    s["xa_gld"] = (gld / gld.shift(63)) < (p / p.shift(63))
    s["tbill_falling"] = irx < irx.shift(63)
    # NaN inputs must stay NaN (comparisons with NaN evaluate False); rebuild the validity masks explicitly
    valid = {
        **{f"sma{k}": p.rolling(k).mean().notna() for k in (50, 100, 150, 200, 250)},
        "sma200_slope": sma200.shift(21).notna(), "mom_3m": p.shift(63).notna(), "mom_6m": p.shift(126).notna(),
        "mom_12m": p.shift(252).notna(), "mom_12_1": p.shift(252).notna(), "dual_mom_12m": p.shift(252).notna() & tb.shift(252).notna(),
        "vol21_lt_vol63": v63.notna(), "vol_spike": v63.notna(), "vol21_lt_med252": v21.rolling(252).median().notna(),
        "dd_10": hi252.notna(), "dd_15": hi252.notna(),
        "breadth200_50": br["share200"].notna(), "breadth200_60": br["share200"].notna(),
        "breadth_rising": br["share200"].notna() & br["share200"].shift(21).notna(), "breadth_pos63": br["pos63"].notna(),
        "xa_ief": (p / p.shift(63)).notna() & (ief / ief.shift(63)).notna(), "xa_gld": (p / p.shift(63)).notna() & (gld / gld.shift(63)).notna(),
        "tbill_falling": irx.notna() & irx.shift(63).notna(),
    }
    out = pd.DataFrame({k: v.astype(float).where(valid[k]) for k, v in s.items()})
    a, b, c = out["sma200"], out["mom_12_1"], out["breadth200_50"]
    votes = a + b + c
    out["ens_majority"] = (votes >= 2).astype(float).where(votes.notna())
    out["ens_all3"] = (votes == 3).astype(float).where(votes.notna())
    out["ens_any1"] = (votes >= 1).astype(float).where(votes.notna())
    out["ens_sma_mom"] = (a + b == 2).astype(float).where((a + b).notna())
    out["ens_sma_breadth"] = (a + c == 2).astype(float).where((a + c).notna())
    out["ens_mom_breadth"] = (b + c == 2).astype(float).where((b + c).notna())
    out.insert(0, "always_up", 1.0)
    return out


def next_month_returns(p: pd.Series, mes: list[pd.Timestamp]) -> pd.Series:
    """Return from the execution close (trading day after month end t) to the next month's execution close, indexed by t."""
    idx = p.index
    out = {}
    for k in range(len(mes) - 1):
        a = idx.get_loc(mes[k]) + 1
        b = idx.get_loc(mes[k + 1]) + 1
        if b >= len(idx):
            break
        out[mes[k]] = p.iloc[b] / p.iloc[a] - 1
    return pd.Series(out)


# ------------------------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def binom_two_sided(k: int, n: int, p0: float) -> float:
    """Exact two-sided binomial p-value (doubled smaller tail, capped at 1)."""
    if n == 0 or not (0 < p0 < 1):
        return np.nan
    pmf = [math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(n + 1)]
    lower = sum(pmf[: k + 1])
    upper = sum(pmf[k:])
    return min(1.0, 2 * min(lower, upper))


def months_needed(p: float) -> float:
    return np.inf if p <= 0.5 else (1.96 * 0.5 / (p - 0.5)) ** 2


def cagr_mdd(monthly: pd.Series) -> tuple[float, float]:
    nav = (1 + monthly).cumprod()
    yrs = len(monthly) / 12
    return nav.iloc[-1] ** (1 / yrs) - 1, (nav / nav.cummax() - 1).min()


def score(call: pd.Series, ret: pd.Series, cash: pd.Series) -> dict:
    """Hit statistics and switch-strategy performance for one call series (1 = up) on one target."""
    m = call.notna()
    c, r, cs = call[m].astype(bool), ret[m], cash[m]
    up = r > 0
    n = int(m.sum())
    hits = int((c == up).sum())
    lo, hi = wilson(hits, n)
    hr = hits / n if n else np.nan
    base = float(up.mean()) if n else np.nan
    sw = r.where(c, cs)
    cg, md = cagr_mdd(sw)
    bcg, bmd = cagr_mdd(r)
    switches = int((c.astype(int).diff().abs() > 0).sum())
    return {"n": n, "first_month": (m[m].index[0] + pd.offsets.MonthBegin(1)).strftime("%Y-%m") if n else "",
            "hits": hits, "hit_rate": hr, "ci_lo": lo, "ci_hi": hi,
            "hit_up_months": float((c[up]).mean()) if up.any() else np.nan,
            "hit_down_months": float((~c[~up]).mean()) if (~up).any() else np.nan,
            "share_called_up": float(c.mean()) if n else np.nan, "base_rate": base,
            "p_vs_50": binom_two_sided(hits, n, 0.5), "p_vs_base": binom_two_sided(hits, n, base) if n else np.nan,
            "months_needed_vs_50": months_needed(hr), "months_needed_vs_base": np.inf if hr <= base else (1.96 * 0.5 / (hr - base)) ** 2,
            "real_signal": bool(lo > 0.5 and hr >= REAL_SIGNAL_MIN), "grid_pass": bool(hr >= GRID_PASS_MIN),
            "switch_cagr": cg, "switch_mdd": md, "switches": switches, "bh_cagr": bcg, "bh_mdd": bmd}


# ------------------------------------------------------------------------------------------------- main

def run_null(ret: pd.Series, runs: int, seed: int) -> pd.DataFrame:
    """Coin-flip callers and noisy always-up callers scored on exactly the real target months."""
    up = (ret > 0).to_numpy()
    n = len(up)
    rng = np.random.default_rng(seed)
    coin = (rng.random((runs, n)) < 0.5)
    coin_hit = (coin == up).mean(axis=1)
    flip_p = rng.uniform(0.0, 0.5, size=runs)
    noisy = rng.random((runs, n)) >= flip_p[:, None]  # True = up unless flipped
    noisy_hit = (noisy == up).mean(axis=1)
    return pd.DataFrame({"kind": ["coin_flip"] * runs + ["always_up_noisy"] * runs, "run": list(range(runs)) * 2,
                         "flip_p": np.r_[np.full(runs, 0.5), flip_p], "hit_rate": np.r_[coin_hit, noisy_hit]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2000-01-01", help="first target month")
    ap.add_argument("--end", default="2024-12-31", help="last target month")
    ap.add_argument("--sub-start", default="2010-01-01", help="first target month of the sub-period")
    ap.add_argument("--null-runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="timing_audit")
    ap.add_argument("--chart", default="docs/timing_audit.png")
    ap.add_argument("--chart-only", action="store_true")
    args = ap.parse_args(argv)
    if args.chart_only:
        chart(pd.read_csv(f"{args.out}_rules.csv"), pd.read_csv(f"{args.out}_null.csv"), args.chart)
        return 0

    start, end, sub_start = pd.Timestamp(args.start), pd.Timestamp(args.end), pd.Timestamp(args.sub_start)
    px, member, _eligible, _spy = load_universe("1998-01-01", args.end)
    qqq = read_series("qqq_full_cache.csv")
    spy = load_spy()
    irx = read_series("irx_cache.csv")
    ief = read_series("ief_full_cache.csv")
    gld = read_series("gld_full_cache.csv")
    idx = qqq.index.union(spy.index)
    idx = idx[(idx >= spy.index[0]) & (idx <= end + pd.DateOffset(months=2))]
    targets = {"SPY": spy.reindex(idx).ffill(), "QQQ": qqq.reindex(idx).ffill()}
    irx_d = irx.reindex(idx).ffill()
    tb = tbill_index(irx_d)
    ief_d, gld_d = ief.reindex(idx).ffill(), gld.reindex(idx).ffill()
    br = {k: v.reindex(idx).ffill(limit=5) for k, v in breadth_series(px, member).items()}

    mes = list(T5.month_ends(idx))
    # signal dates: month ends whose NEXT month lies in [start, end]
    sig_dates = [d for d in mes if start.replace(day=1) <= d + pd.offsets.MonthBegin(1) <= end]
    periods = {f"{start.year}-{end.year}": sig_dates, f"{sub_start.year}-{end.year}": [d for d in sig_dates if d + pd.offsets.MonthBegin(1) >= sub_start]}
    print(f"signal month ends {sig_dates[0].date()} .. {sig_dates[-1].date()}  ({len(sig_dates)} months); sub-period {len(periods[list(periods)[1]])} months")
    cash = irx_d.reindex(sig_dates).fillna(0) / 100 / 12

    rows, null_rows = [], []
    for tname, p in targets.items():
        sig = rule_signals(p, irx_d, tb, ief_d, gld_d, br)
        ret = next_month_returns(p, mes)
        # no look-ahead: every signal row is dated t, the target return starts at the close AFTER t
        for d in sig_dates:
            a = idx[idx.get_loc(d) + 1]
            assert a > d and d in sig.index and d in ret.index, d
        # ... and recomputing the signals from inputs truncated at t reproduces the full-sample row at t
        for d in np.random.default_rng(args.seed).choice(sig_dates, 3, replace=False):
            trunc = rule_signals(p.loc[:d], irx_d.loc[:d], tb.loc[:d], ief_d.loc[:d], gld_d.loc[:d], {k: v.loc[:d] for k, v in br.items()}).loc[d]
            pd.testing.assert_series_equal(trunc, sig.loc[d], check_names=False)
        for pname, dates in periods.items():
            r = ret.reindex(dates)
            assert r.notna().all()
            null = run_null(r, args.null_runs, args.seed)
            q = null.groupby("kind")["hit_rate"].quantile([0.5, 0.95, 0.99]).unstack()
            for kind in q.index:
                null_rows.append({"target": tname, "period": pname, "n": len(dates), "kind": kind, "runs": args.null_runs,
                                  "median": q.loc[kind, 0.5], "p95": q.loc[kind, 0.95], "p99": q.loc[kind, 0.99],
                                  "max": null.loc[null["kind"] == kind, "hit_rate"].max(), "base_rate": float((r > 0).mean())})
            au_rate = score(sig["always_up"].reindex(dates), r, cash.reindex(dates))["hit_rate"]
            for rule in sig.columns:
                st = score(sig[rule].reindex(dates), r, cash.reindex(dates))
                st["beats_always_up"] = bool(st["hit_rate"] > au_rate)
                st.update({"rule": rule, "label": RULE_LABELS[rule], "group": RULE_GROUP[rule], "target": tname, "period": pname,
                           "above_coin_p95": st["hit_rate"] > q.loc["coin_flip", 0.95], "above_coin_p99": st["hit_rate"] > q.loc["coin_flip", 0.99],
                           "above_noisy_p95": st["hit_rate"] > q.loc["always_up_noisy", 0.95], "above_noisy_p99": st["hit_rate"] > q.loc["always_up_noisy", 0.99]})
                rows.append(st)
        print(f"{tname}: done", flush=True)

    cols = ["target", "period", "rule", "label", "group", "n", "first_month", "hits", "hit_rate", "ci_lo", "ci_hi", "hit_up_months", "hit_down_months",
            "share_called_up", "base_rate", "p_vs_50", "p_vs_base", "months_needed_vs_50", "months_needed_vs_base", "real_signal", "grid_pass", "beats_always_up",
            "above_coin_p95", "above_coin_p99", "above_noisy_p95", "above_noisy_p99", "switch_cagr", "switch_mdd", "switches", "bh_cagr", "bh_mdd"]
    rules = pd.DataFrame(rows)[cols]
    null_df = pd.DataFrame(null_rows)
    for (t, pn), g in rules.groupby(["target", "period"]):
        nn = null_df[(null_df["target"] == t) & (null_df["period"] == pn)].set_index("kind")
        real = g[g["rule"] != "always_up"]
        for kind in nn.index:
            null_df.loc[(null_df["target"] == t) & (null_df["period"] == pn) & (null_df["kind"] == kind), "rules_above_p95"] = int((real["hit_rate"] > nn.loc[kind, "p95"]).sum())
            null_df.loc[(null_df["target"] == t) & (null_df["period"] == pn) & (null_df["kind"] == kind), "rules_above_p99"] = int((real["hit_rate"] > nn.loc[kind, "p99"]).sum())
    summary = summarise(rules, null_df)

    rules.to_csv(f"{args.out}_rules.csv", index=False, encoding="utf-8-sig")
    null_df.to_csv(f"{args.out}_null.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {args.out}_rules.csv, {args.out}_null.csv, {args.out}_summary.csv")
    print_report(rules, null_df, summary)
    chart(rules, null_df, args.chart)
    return 0


def summarise(rules: pd.DataFrame, null_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (t, pn), g in rules.groupby(["target", "period"], sort=False):
        nn = null_df[(null_df["target"] == t) & (null_df["period"] == pn)].set_index("kind")
        real = g[g["rule"] != "always_up"].sort_values(["hit_rate", "ci_lo"], ascending=False)
        b = real.iloc[0]
        au = g[g["rule"] == "always_up"].iloc[0]
        out.append({"target": t, "period": pn, "n": int(b["n"]), "best_rule": b["rule"], "best_label": b["label"],
                    "hit_rate": b["hit_rate"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "hit_up_months": b["hit_up_months"], "hit_down_months": b["hit_down_months"],
                    "share_called_up": b["share_called_up"], "base_rate": b["base_rate"], "always_up_hit_rate": au["hit_rate"], "always_up_ci_lo": au["ci_lo"],
                    "p_vs_50": b["p_vs_50"], "p_vs_base": b["p_vs_base"], "months_needed_vs_50": b["months_needed_vs_50"], "months_needed_vs_base": b["months_needed_vs_base"],
                    "coin_p95": nn.loc["coin_flip", "p95"], "coin_p99": nn.loc["coin_flip", "p99"],
                    "noisy_always_up_p95": nn.loc["always_up_noisy", "p95"], "noisy_always_up_p99": nn.loc["always_up_noisy", "p99"],
                    "rules_total": len(real), "rules_real_signal": int(real["real_signal"].sum()), "rules_grid_pass": int(real["grid_pass"].sum()),
                    "rules_above_coin_p99": int(real["above_coin_p99"].sum()), "rules_above_noisy_p99": int(real["above_noisy_p99"].sum()),
                    "rules_beat_always_up": int((real["hit_rate"] > au["hit_rate"]).sum()),
                    "pass_60": bool(b["real_signal"]), "pass_70": bool(b["grid_pass"]),
                    "best_switch_cagr": b["switch_cagr"], "best_switch_mdd": b["switch_mdd"], "bh_cagr": b["bh_cagr"], "bh_mdd": b["bh_mdd"],
                    "best_cagr_rule": real.sort_values("switch_cagr", ascending=False).iloc[0]["rule"],
                    "best_cagr": real["switch_cagr"].max()})
    return pd.DataFrame(out)


def print_report(rules: pd.DataFrame, null_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    for _, s in summary.iterrows():
        print(f"\n[{s['target']} {s['period']}] n={s['n']}  base rate {s['base_rate']*100:.1f}%  always-up {s['always_up_hit_rate']*100:.1f}%"
              f"  coin null p95/p99 {s['coin_p95']*100:.1f}/{s['coin_p99']*100:.1f}%  noisy-always-up p95/p99 {s['noisy_always_up_p95']*100:.1f}/{s['noisy_always_up_p99']*100:.1f}%")
        print(f"  best {s['best_rule']}: {s['hit_rate']*100:.1f}% [{s['ci_lo']*100:.1f}, {s['ci_hi']*100:.1f}]  up {s['hit_up_months']*100:.0f}% / down {s['hit_down_months']*100:.0f}%"
              f"  p50={s['p_vs_50']:.3g} pbase={s['p_vs_base']:.3g}  months needed {s['months_needed_vs_50']:.0f}  pass60={s['pass_60']} pass70={s['pass_70']}"
              f"  real-signal rules {s['rules_real_signal']}/{s['rules_total']}  >=70% {s['rules_grid_pass']}  beat always-up {s['rules_beat_always_up']}")
        g = rules[(rules["target"] == s["target"]) & (rules["period"] == s["period"])].sort_values("hit_rate", ascending=False)
        print(g[["rule", "n", "hit_rate", "ci_lo", "ci_hi", "hit_up_months", "hit_down_months", "share_called_up", "p_vs_50", "p_vs_base", "switch_cagr", "switch_mdd", "bh_cagr", "bh_mdd"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ------------------------------------------------------------------------------------------------ chart

def chart(rules: pd.DataFrame, null_df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    periods = list(dict.fromkeys(rules["period"]))
    group_color = {"均線": PALETTE[0], "動能": PALETTE[1], "廣度": PALETTE[2], "波動": PALETTE[3], "回撤": PALETTE[4], "跨資產": PALETTE[5], "組合": "#555555", "基準": GREY}
    fig, axes = plt.subplots(2, 2, figsize=(17, 15.5), gridspec_kw={"height_ratios": [1.45, 1], "hspace": 0.3, "wspace": 0.42})
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.14, right=0.985)

    for ax, pn in zip(axes[0], periods):
        g = rules[(rules["target"] == "QQQ") & (rules["period"] == pn)].sort_values("hit_rate").reset_index(drop=True)
        nn = null_df[(null_df["target"] == "QQQ") & (null_df["period"] == pn)].set_index("kind")
        base = g["base_rate"].iloc[0]
        y = np.arange(len(g))
        ax.axvspan(0.5, nn.loc["coin_flip", "p99"], color=LIGHT, alpha=0.45, lw=0, zorder=0)
        ax.axvline(0.5, color=GREY, lw=1, zorder=1)
        ax.axvline(base, color=GREY, lw=1, ls="--", zorder=1)
        ax.axvline(nn.loc["always_up_noisy", "p99"], color=GREY, lw=1, ls=":", zorder=1)
        ax.axvline(REAL_SIGNAL_MIN, color=PALETTE[1], lw=1.2, ls="--", zorder=1)
        ax.axvline(GRID_PASS_MIN, color=PALETTE[1], lw=1.2, zorder=1)
        for i, r in g.iterrows():
            col = group_color[r["group"]]
            ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=col, lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot(r["hit_rate"], i, "o", color=col, ms=6, zorder=3, mec="white", mew=0.8)
            ax.text(r["ci_hi"] + 0.006, i, f"{r['hit_rate']*100:.0f}%", va="center", ha="left", fontsize=8, color="#444444")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{l}  (n={int(n)})" if n != g['n'].max() else l for l, n in zip(g["label"], g["n"])], fontsize=9)
        for lab, r in zip(ax.get_yticklabels(), g.itertuples()):
            lab.set_color(group_color[r.group])
        ax.set_xlim(0.28, 0.86)
        ax.set_ylim(-0.8, len(g) - 0.2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Patch(color=LIGHT, alpha=0.45, label=f"200 個亂猜者：50% 到第 99 百分位 {nn.loc['coin_flip','p99']*100:.0f}%"),
                   Line2D([], [], color=GREY, lw=1, ls="--", label=f"永遠看多 = 漲月比例 {base*100:.0f}%"),
                   Line2D([], [], color=GREY, lw=1, ls=":", label=f"看多加雜訊者第 99 百分位 {nn.loc['always_up_noisy','p99']*100:.0f}%"),
                   Line2D([], [], color=PALETTE[1], lw=1.2, ls="--", label="事前門檻 60%（且下界 > 50%）"),
                   Line2D([], [], color=PALETTE[1], lw=1.2, label="第 19.3 節網格 70%")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.035), frameon=False, fontsize=8, handlelength=2.2, ncol=3, columnspacing=1.2)
        ax.set_title(f"QQQ 下月方向命中率與 95% 信賴區間，目標月 {pn}（n = {int(g['n'].max())}）", loc="left", fontsize=11)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    # (3) sensitivity to up-months vs down-months, QQQ full period
    ax = axes[1][0]
    g = rules[(rules["target"] == "QQQ") & (rules["period"] == periods[0])]
    ax.plot([0, 1], [1, 0], color=LIGHT, lw=1, ls="--", zorder=0)
    ax.text(0.30, 0.995, "虛線：漲月命中 + 跌月命中 = 100%，\n不含資訊的規則（含永遠看多）都落在線上", fontsize=8, color="#666666", va="top")
    for grp, gg in g.groupby("group"):
        ax.scatter(gg["hit_up_months"], gg["hit_down_months"], s=46, color=group_color[grp], edgecolor="white", lw=0.6, zorder=3, label=grp)
    real = g[g["rule"] != "always_up"]
    pick = set(real.sort_values("hit_rate", ascending=False)["rule"].head(2)) | {"always_up", "vol_spike", "ens_any1", "sma200", "breadth_rising", "vol21_lt_med252", "tbill_falling"}
    sel = g[g["rule"].isin(pick)]
    f3 = lambda r: f"{r['label']}（{r['hit_rate']*100:.0f}%）"
    _stack_labels(ax, sel[sel["hit_up_months"] >= 0.7], "hit_up_months", "hit_down_months", fmt=f3, x_text=0.66, y_top=0.93, y_step=0.05)
    _stack_labels(ax, sel[sel["hit_up_months"] < 0.7], "hit_up_months", "hit_down_months", fmt=f3, x_text=0.04, y_top=0.78, y_step=0.05)
    ax.set_xlabel("漲月命中率（看多的比例）")
    ax.set_ylabel("跌月命中率（看空的比例）")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"QQQ {periods[0]}：對漲月與跌月的敏感度，每個規則一點", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2, title="規則家族", title_fontsize=8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # (4) CAGR vs MDD of the switch strategy on QQQ, full period
    ax = axes[1][1]
    g = rules[(rules["target"] == "QQQ") & (rules["period"] == periods[0])]
    bh = g.iloc[0]
    real = g[g["rule"] != "always_up"]
    for grp, gg in g.groupby("group"):
        ax.scatter(gg["switch_mdd"], gg["switch_cagr"], s=46, color=group_color[grp], edgecolor="white", lw=0.6, zorder=3)
    ax.scatter(bh["bh_mdd"], bh["bh_cagr"], s=110, marker="*", color="black", zorder=4)
    ax.annotate(f"QQQ 買進持有 {bh['bh_cagr']*100:.1f}% / {bh['bh_mdd']*100:.0f}%", (bh["bh_mdd"], bh["bh_cagr"]), xytext=(8, -4), textcoords="offset points", fontsize=8, va="top")
    ax.axhline(0.30, color=PALETTE[1], lw=1, ls="--")
    ax.axvline(-0.30, color=PALETTE[1], lw=1, ls="--")
    ax.text(-0.29, 0.305, "第 19 節目標：CAGR 30% 以上、MDD 30% 以內", fontsize=8, color=PALETTE[1], va="bottom")
    pick = set(real.sort_values("switch_cagr", ascending=False)["rule"].head(3)) | {"sma200", "vol_spike", "ens_majority", "breadth_rising", "breadth_pos63", "dd_10"}
    sel = g[g["rule"].isin(pick)]
    f4 = lambda r: f"{r['label']} {r['switch_cagr']*100:.1f}% / {r['switch_mdd']*100:.0f}%"
    _stack_labels(ax, sel[sel["switch_mdd"] >= -0.45], "switch_mdd", "switch_cagr", fmt=f4, x_text=-0.43, y_top=0.255, y_step=0.017)
    _stack_labels(ax, sel[sel["switch_mdd"] < -0.45], "switch_mdd", "switch_cagr", fmt=f4, x_text=-0.89, y_top=0.2, y_step=0.017)
    ax.set_xlabel("最大回撤（月底淨值）")
    ax.set_ylabel("CAGR")
    ax.set_xlim(-0.9, 0)
    ax.set_ylim(0, 0.34)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"QQQ {periods[0]}：月底開關策略（看多持 QQQ，否則國庫券）的 CAGR 與最大回撤", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("月底擇時規則的前瞻審計：對下個月方向的命中率（訊號只用 t 日以前資料，目標為 t+1 收盤起算的次月報酬）", x=0.01, ha="left", fontsize=13, y=0.995)
    fig.text(0.01, 0.005, "灰帶：200 個亂猜者的 50% 到第 99 百分位；灰虛線：永遠看多的命中率（= 漲月比例）；橘線：事前訂的 60% 與 70% 門檻。"
             "廣度規則用時點成份股，IEF / GLD 規則自 2002-11 / 2005-02 起。", fontsize=8.5, color="#555555")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def _stack_labels(ax, sel: pd.DataFrame, xcol: str, ycol: str, fmt, x_text: float, y_top: float, y_step: float) -> None:
    """Direct labels for a subset of points, stacked in an empty region of the axes with thin leader lines (ordered by the point's y so the leaders do not cross)."""
    sel = sel.sort_values(ycol, ascending=False)
    for i, (_, r) in enumerate(sel.iterrows()):
        yy = y_top - i * y_step
        ax.annotate(fmt(r), (r[xcol], r[ycol]), xytext=(x_text, yy), textcoords="data", fontsize=7.5, va="center", ha="left", color="#333333",
                    arrowprops={"arrowstyle": "-", "color": "#b0b0b0", "lw": 0.6, "shrinkA": 0, "shrinkB": 3})


if __name__ == "__main__":
    raise SystemExit(main())
