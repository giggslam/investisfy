"""Section 21: the three remaining free, non-price data sources run through the section 20 IC audit.

Same machinery as sp500_signal_audit.py (imported): monthly Spearman IC of each feature against the next month's
return (execution close after the signal month end to the execution close after the next month end), the same 200
random-feature null, the same pre-registered thresholds (mean IC >= 0.03, t >= 2, >= 8 / 12 years positive 2013-2024;
grid-passing composite OOS IC >= 0.15), and the same anchored walk-forward composites.

New features (sp500_free_signals.build_frames): 5 from XBRL balance sheet / cash flow (asset growth, net issuance,
accruals, book-to-market, cash-flow yield), 3 from Form 4 insider trades (net purchase ratio 6m, distinct open-market
buyers 3m, net value 6m / market cap), 4 from FINRA short interest (ratio, days to cover, 1m and 3m change; 2018+).

Walk-forward runs, all tested 2013-2024 (train from 2010) on the XBRL-covered universe so that they are comparable:
  technical       33 technical candidates alone (the section 20 baseline on this mask and period)
  free            the 8 XBRL + Form 4 features alone
  tech_free       33 + 8
  tech_fund_free  33 + 15 fundamental + 8 (on names also covered by the section 17 fundamentals)
  si              the 4 FINRA features alone, tested 2021-2024 (train from 2018: too short, shown for completeness)

Usage
  python sp500_free_signals_audit.py [--null-runs 200] [--seed 42] [--out free_signals_audit] [--chart docs/free_signals_audit.png]
  python sp500_free_signals_audit.py --chart-only
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from statistics import NormalDist

import numpy as np
import pandas as pd

import sp500_free_signals as S
import sp500_precursor_fundamental as Fd
import sp500_precursor_technical as Tch
import sp500_signal_audit as A
import sp500_top5_monthly as T5
from sp500_precursor_events import load_universe

PAL = A.PAL
GREYS = A.GREYS
GROUP_ZH = {"xbrl": "XBRL 資產負債與現金流（5）", "form4": "Form 4 內部人交易（3）", "finra": "FINRA 空單（4，2018 起）"}
GCOL = {"xbrl": PAL[0], "form4": PAL[2], "finra": PAL[1]}
SI_WINDOW = ("2018-01-01", "2024-12-31")
N_TESTS_TOTAL = A.N_TESTS + len(S.FEATURES)   # 57 earlier tests + 12 new ones


def window_summary(ic: pd.Series, a: str, b: str, sign: float) -> dict:
    w = (sign * ic).loc[a:b].dropna()
    s = A.ic_summary(w, "w_")
    yearly = w.groupby(w.index.year).mean()
    return {"mean_ic_2018_2024_adj": s["w_mean_ic"], "t_2018_2024_adj": s["w_t"], "years_pos_2018_2024": int((yearly > 0).sum()), "n_years_2018_2024": int(len(yearly)), "n_months_2018_2024": s["w_n_months"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--null-runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="free_signals_audit")
    p.add_argument("--chart", default="docs/free_signals_audit.png")
    p.add_argument("--chart-only", action="store_true")
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--quarterly", default="precursor_fund_quarterly.csv")
    args = p.parse_args(argv)
    if args.chart_only:
        chart(pd.read_csv(f"{args.out}_features.csv"), pd.read_csv(f"{args.out}_walkforward.csv"), pd.read_csv(f"{args.out}_walkforward_monthly.csv", parse_dates=["signal_date"]), pd.read_csv(f"{args.out}_summary.csv"), args.chart)
        print(f"wrote {args.chart}")
        return 0

    t0 = time.time()
    px, member, eligible, spy = load_universe(args.start, args.end)
    all_mes = T5.month_ends(px.index)
    mes = [d for d in all_mes if d >= pd.Timestamp(args.start) and d <= pd.Timestamp(args.end) and eligible.loc[d].sum() >= 50]
    mes_idx = pd.DatetimeIndex(mes)
    elig_mes = eligible.reindex(mes)
    cik = Fd.ticker_cik_map(list(px.columns))
    print(f"universe {px.shape[1]} tickers, {len(cik)} with a CIK; {len(mes)} signal month-ends {mes[0].date()}..{mes[-1].date()}")

    # 1. targets (identical to section 20) ------------------------------------------------------------
    fwd1, fwd3, exec_date, end1 = A.targets(px, eligible, mes)

    # 2. features -----------------------------------------------------------------------------------
    frames, groups, labels, covered = S.build_frames(px, mes, cik)
    covered = covered.reindex(index=mes_idx, columns=px.columns).fillna(False) & elig_mes
    for k in frames:
        frames[k] = frames[k].reindex(index=mes_idx, columns=px.columns)
    cov_share = (covered.sum(axis=1) / elig_mes.sum(axis=1)).loc["2010-01-01":]
    print(f"{len(frames)} free-signal features; XBRL coverage of eligible names 2010+: median {cov_share.median() * 100:.0f}%, first month with >= 50% coverage {cov_share[cov_share >= 0.5].index.min().date() if (cov_share >= 0.5).any() else None}")
    for k in frames:
        n = frames[k].where(elig_mes).notna().sum(axis=1)
        nz = n[n > 0]
        print(f"  {k:14s} {groups[k]:6s} names/month median {int(nz.median()) if len(nz) else 0:4d}, months with >= 30 names {int((n >= 30).sum()):3d}, first {nz.index.min().date() if len(nz) else None}")
    # technical and fundamental frames for the composite runs
    vol = Tch.load_volume(list(px.columns))
    F = Tch.compute_features(px, eligible, spy, vol)
    tech = {k: fr.reindex(mes) for k, fr in F.items()}
    R = T5.rank_frames(px, eligible)
    for k, fr in R.items():
        tech[f"mom_{k}"] = fr.reindex(mes)
    fund, fund_cov = A.fundamental_frames(px, eligible, spy, mes, args.quarterly)
    all_frames = {**tech, **fund, **frames}
    cands_tech = [k for k in tech if k not in A.COMPOSITE_DROP]
    cands_free = [k for k in frames if groups[k] != "finra"]
    cands_si = [k for k in frames if groups[k] == "finra"]
    print(f"composite candidates: technical {len(cands_tech)}, fundamental {len(fund)}, free {len(cands_free)}, short interest {len(cands_si)} ({time.time() - t0:.0f}s)")

    # 3. per-feature IC ------------------------------------------------------------------------------
    rows, ic_series = [], {}
    for k in frames:
        ic1, n1 = A.row_spearman(frames[k].where(elig_mes), fwd1)
        ic3, _ = A.row_spearman(frames[k].where(elig_mes), fwd3)
        ic_series[k] = ic1
        rec = {"feature": k, "label": labels[k], "group": groups[k], "expected_sign": S.EXPECTED_SIGN[k], "mean_names": float(n1[n1 > 0].mean()) if (n1 > 0).any() else 0.0,
               "first_month": ic1.dropna().index.min(), "last_month": ic1.dropna().index.max(), **A.feature_stats(ic1, ic3)}
        rec.update(window_summary(ic1, *SI_WINDOW, rec["sign"]))
        rec["sign_as_expected"] = bool(rec["sign"] == S.EXPECTED_SIGN[k])
        # the threshold evaluated in the EXPECTED direction (a significant IC of the wrong sign is not the documented effect)
        exp = S.EXPECTED_SIGN[k] * ic1
        win = exp.loc[f"{A.PREREG_YEARS[0]}-01-01":f"{A.PREREG_YEARS[1]}-12-31"].dropna()
        w = A.ic_summary(win, "e_")
        yearly = win.groupby(win.index.year).mean()
        rec.update({"mean_ic_2013_2024_expected": w["e_mean_ic"], "t_2013_2024_expected": w["e_t"], "years_pos_2013_2024_expected": int((yearly > 0).sum())})
        rec["pass_real_signal_expected"] = bool(len(win) >= 12 and w["e_mean_ic"] >= A.IC_THRESHOLD and w["e_t"] >= A.T_THRESHOLD and rec["years_pos_2013_2024_expected"] >= A.YEARS_THRESHOLD)
        rows.append(rec)
    features = pd.DataFrame(rows)

    # 4. null (same construction as section 20, plus the 2018-2024 window used for the FINRA features) --
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for i in range(args.null_runs):
        Z = pd.DataFrame(rng.standard_normal(elig_mes.shape), index=mes_idx, columns=px.columns).where(elig_mes)
        ic1, _ = A.row_spearman(Z, fwd1)
        rec = A.feature_stats(ic1, pd.Series(np.nan, index=mes_idx))
        rec = {k: v for k, v in rec.items() if not k.startswith("fwd3") and "3m" not in k}
        rec.update(window_summary(ic1, *SI_WINDOW, rec["sign"]))
        # 2010-2024 window: the XBRL / Form 4 features only exist from 2010
        w = (rec["sign"] * ic1).loc["2010-01-01":].dropna()
        rec["mean_ic_2010_2024_adj"] = w.mean()
        rec["t_2010_2024_adj"] = w.mean() / w.std(ddof=1) * math.sqrt(len(w))
        null_rows.append({"run": i, **rec})
    null = pd.DataFrame(null_rows)
    q = {"abs_ic_full_p99": null.mean_ic.abs().quantile(0.99), "abs_t_full_p99": null.t.abs().quantile(0.99),
         "adj_ic_1324_p99": null.mean_ic_2013_2024_adj.quantile(0.99), "adj_t_1324_p99": null.t_2013_2024_adj.quantile(0.99),
         "adj_ic_1024_p99": null.mean_ic_2010_2024_adj.quantile(0.99), "adj_t_1024_p99": null.t_2010_2024_adj.quantile(0.99),
         "adj_ic_1824_p99": null.mean_ic_2018_2024_adj.quantile(0.99), "adj_t_1824_p99": null.t_2018_2024_adj.quantile(0.99)}
    bonf_z = NormalDist().inv_cdf(1 - 0.05 / N_TESTS_TOTAL / 2)
    print(f"\nnull ({args.null_runs} random features): sign-adjusted mean IC p99 full {q['abs_ic_full_p99']:.4f}, 2010-24 {q['adj_ic_1024_p99']:.4f}, 2013-24 {q['adj_ic_1324_p99']:.4f}, 2018-24 {q['adj_ic_1824_p99']:.4f}; "
          f"|t| p99 full {q['abs_t_full_p99']:.2f}, 2018-24 {q['adj_t_1824_p99']:.2f}; Bonferroni |t| for {N_TESTS_TOTAL} tests: {bonf_z:.2f}")
    # the XBRL / Form 4 features live on 2010-2024, so compare them with the null over the same window
    features["null_p99_ic_same_window"] = np.where(features.group == "finra", q["adj_ic_1824_p99"], q["adj_ic_1024_p99"])
    features["null_p99_t_same_window"] = np.where(features.group == "finra", q["adj_t_1824_p99"], q["adj_t_1024_p99"])
    features["above_null_p99_ic"] = features.mean_ic_signed_abs > features.null_p99_ic_same_window
    features["above_null_p99_t"] = features.t_signed_abs > features.null_p99_t_same_window
    features["above_bonferroni_t"] = features.t_signed_abs > bonf_z
    features = features.sort_values("t_signed_abs", ascending=False).reset_index(drop=True)
    features.to_csv(f"{args.out}_features.csv", index=False, encoding="utf-8-sig")
    null.to_csv(f"{args.out}_null.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    show = ["feature", "group", "expected_sign", "sign", "mean_names", "n_months", "mean_ic", "t", "pct_years_pos", "mean_ic_2010_2019", "mean_ic_2020_2024", "mean_ic_2013_2024_adj", "t_2013_2024_adj",
            "years_pos_2013_2024", "mean_ic_2018_2024_adj", "pass_real_signal", "pass_real_signal_expected", "above_null_p99_t", "mean_ic_3m"]
    print("\n-- free-signal features sorted by sign-adjusted t")
    print(features[show].round(4).to_string(index=False))

    # 5. walk-forward composites ----------------------------------------------------------------------
    end_year = pd.Timestamp(args.end).year
    years = list(range(2013, end_year + 1))
    tr0 = pd.Timestamp("2010-01-01")
    mask = covered
    mask_fund = covered & fund_cov.reindex(index=mes_idx, columns=px.columns).fillna(False)
    runs = {}
    print(f"\nwalk-forward on the XBRL-covered universe (median {int(mask.sum(axis=1).loc['2013-01-01':].median())} names/month), test years {years[0]}-{years[-1]}")
    runs["technical"] = A.walk_forward("technical", all_frames, cands_tech, mask, fwd1, end1, mes, years, tr0, args.lam)
    runs["free"] = A.walk_forward("free", all_frames, cands_free, mask, fwd1, end1, mes, years, tr0, args.lam)
    runs["tech_free"] = A.walk_forward("tech_free", all_frames, cands_tech + cands_free, mask, fwd1, end1, mes, years, tr0, args.lam)
    runs["tech_fund_free"] = A.walk_forward("tech_fund_free", all_frames, cands_tech + list(fund) + cands_free, mask_fund, fwd1, end1, mes, years, tr0, args.lam)
    si_mask = mask & frames["si_ratio"].notna()
    si_years = list(range(2021, end_year + 1))
    if (si_mask.sum(axis=1) >= A.MIN_NAMES).sum() >= 36:
        runs["si"] = A.walk_forward("si", all_frames, cands_si, si_mask, fwd1, end1, mes, si_years, pd.Timestamp(SI_WINDOW[0]), args.lam)
        runs["tech_free_si"] = A.walk_forward("tech_free_si", all_frames, cands_tech + cands_free + cands_si, si_mask, fwd1, end1, mes, si_years, pd.Timestamp(SI_WINDOW[0]), args.lam)
    wf = pd.concat([r[0] for r in runs.values()], ignore_index=True)
    monthly = pd.concat([r[1] for r in runs.values()], ignore_index=True)
    wf.to_csv(f"{args.out}_walkforward.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(f"{args.out}_walkforward_monthly.csv", index=False, encoding="utf-8-sig")
    comp = {}
    print("\n-- out-of-sample composite IC")
    for run, (w, mo, sc) in runs.items():
        for col in ("ic_ew", "ic_ridge", "ic_ridge_all", "ic_best_single"):
            comp[(run, col)] = A.oos_summary(mo, col)
            s = comp[(run, col)]
            print(f"  {run:15s} {col:15s} mean {s['mean']:+.4f}  t {s['t']:+.2f}  years>0 {s['years_pos']}/{s['n_years']}  months {s['n_months']}")
    # paired comparison technical vs tech_free on the same months
    a = runs["technical"][1].set_index("signal_date"); b = runs["tech_free"][1].set_index("signal_date")
    paired = {}
    for col in ("ic_ew", "ic_ridge", "ic_ridge_all"):
        d = (b[col] - a[col]).dropna()
        paired[col] = {"mean_diff": d.mean(), "t": d.mean() / d.std(ddof=1) * math.sqrt(len(d)) if len(d) > 1 else np.nan, "n": int(len(d))}
        print(f"  tech_free minus technical, {col}: {d.mean():+.4f} (t {paired[col]['t']:+.2f}, {len(d)} months)")
    # scores of the tech_free run for a possible forward search (only used if the thresholds are met)
    sc = runs["tech_free"][2]
    sc.insert(1, "exec_date", exec_date.reindex(sc.signal_date).to_numpy())
    for c in ("score_ew", "score_ridge"):
        sc[f"rank_{c[6:]}"] = sc.groupby("signal_date")[c].rank(ascending=False, method="first")
    sc.sort_values(["signal_date", "rank_ridge"]).to_csv(f"{args.out}_scores.csv", index=False, encoding="utf-8-sig")

    # 6. summary ------------------------------------------------------------------------------------
    best = features.iloc[0]
    best_exp = features.sort_values("t_2013_2024_expected", ascending=False).iloc[0]
    rows = [("signal_months", len(mes)), ("n_new_features", len(frames)), ("n_tests_total", N_TESTS_TOTAL), ("bonferroni_abs_t", bonf_z),
            ("xbrl_coverage_median_2010", cov_share.median()), ("n_form4_tickers", int(frames["ins_npr_6m"].ne(0).any().sum()) if "ins_npr_6m" in frames else 0),
            ("best_feature_by_t", best.feature), ("best_feature_label", best.label), ("best_feature_mean_ic", best.mean_ic), ("best_feature_t", best.t), ("best_feature_sign", best.sign),
            ("best_feature_expected_sign", best.expected_sign), ("best_feature_mean_ic_2013_2024_adj", best.mean_ic_2013_2024_adj), ("best_feature_years_pos_2013_2024", best.years_pos_2013_2024),
            ("best_expected_direction_feature", best_exp.feature), ("best_expected_direction_mean_ic_2013_2024", best_exp.mean_ic_2013_2024_expected), ("best_expected_direction_t_2013_2024", best_exp.t_2013_2024_expected),
            ("max_abs_mean_ic", features.mean_ic_signed_abs.max()),
            ("n_sign_as_expected", int(features.sign_as_expected.sum())), ("n_pass_real_signal", int(features.pass_real_signal.sum())), ("features_pass_real_signal", "|".join(features[features.pass_real_signal].feature)),
            ("n_pass_real_signal_expected", int(features.pass_real_signal_expected.sum())), ("n_above_null_p99_t", int(features.above_null_p99_t.sum())), ("n_above_bonferroni", int(features.above_bonferroni_t.sum())),
            ("null_runs", args.null_runs)] + [(f"null_{k}", v) for k, v in q.items()]
    for (run, col), s in comp.items():
        key = f"oos_{run}_{col[3:]}"
        rows += [(f"{key}_mean_ic", s["mean"]), (f"{key}_t", s["t"]), (f"{key}_years_pos", f"{s['years_pos']}/{s['n_years']}"), (f"{key}_months", s["n_months"])]
    for col, d in paired.items():
        rows += [(f"paired_tech_free_minus_technical_{col[3:]}_mean", d["mean_diff"]), (f"paired_tech_free_minus_technical_{col[3:]}_t", d["t"])]
    rows += [("threshold_real_signal", f"mean IC >= {A.IC_THRESHOLD}, t >= {A.T_THRESHOLD}, >= {A.YEARS_THRESHOLD}/12 years positive 2013-2024"), ("threshold_grid_passing", f"OOS composite IC >= {A.GRID_THRESHOLD}"),
             ("lam", args.lam), ("seed", args.seed), ("runtime_s", round(time.time() - t0))]
    summary = pd.DataFrame(rows, columns=["key", "value"])
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    print("\n-- walk-forward by year")
    print(wf[["run", "year", "n_train_months", "n_selected", "selected", "best_single_feature", "ic_ew", "ic_ridge", "ic_ridge_all", "ic_best_single"]].round(3).to_string(index=False))
    chart(features, wf, monthly, summary, args.chart)
    print(f"\nwrote {args.out}_features.csv, _null.csv, _walkforward.csv, _walkforward_monthly.csv, _scores.csv, _summary.csv, {args.chart} ({time.time() - t0:.0f}s)")
    return 0


# ----------------------------------------------------------------------------- chart
def chart(features: pd.DataFrame, wf: pd.DataFrame, monthly: pd.DataFrame, summary: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    sv = dict(zip(summary.key, summary.value))
    fig = plt.figure(figsize=(17, 12.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.38, wspace=0.22, left=0.17, right=0.985, top=0.93, bottom=0.07)

    # panel 1: sign-adjusted mean IC per feature, 2013-2024 (2018-2024 for FINRA), with the null band and the threshold
    ax1 = fig.add_subplot(gs[0, :])
    f = features.copy()
    f["win_ic"] = np.where(f.group == "finra", f.mean_ic_2018_2024_adj, f.mean_ic_2013_2024_adj)
    f["win_t"] = np.where(f.group == "finra", f.t_2018_2024_adj, f.t_2013_2024_adj)
    f = f.sort_values("win_ic", ascending=True).reset_index(drop=True)
    y = np.arange(len(f))
    p99_1324 = float(sv["null_adj_ic_1324_p99"]); p99_1824 = float(sv["null_adj_ic_1824_p99"])
    ax1.axvspan(0, p99_1324, color="#e4e4e4", zorder=0)
    ax1.axvline(A.IC_THRESHOLD, color=GREYS[0], ls="--", lw=1)
    ax1.axvline(0, color=GREYS[2], lw=0.8)
    ax1.barh(y, f.win_ic, color=[GCOL[g] for g in f.group], height=0.62, zorder=2)
    for yi, (v, t, s, e, lab) in enumerate(zip(f.win_ic, f.win_t, f.sign, f.expected_sign, f.label)):
        note = "" if s == e else "（方向與文獻相反）"
        ax1.annotate(f"{v:+.3f}（t {t:+.1f}）{note}", (max(v, 0), yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.2, color="#333333", zorder=3)
    ax1.set_yticks(y); ax1.set_yticklabels(f.label, fontsize=9)
    ax1.set_xlim(min(-0.005, f.win_ic.min() - 0.003), max(0.045, f.win_ic.max() + 0.02))
    ax1.set_xlabel("月度 Spearman IC 平均（已按全樣本方向調正；XBRL 與 Form 4 為 2013 至 2024 年，FINRA 為 2018 至 2024 年）", fontsize=9.5)
    ax1.set_title("12 個免費非價格訊號對下月報酬的 IC：灰帶 = 200 個亂數特徵的 99 百分位，虛線 = 事前門檻 0.03", loc="left", fontsize=12, fontweight="bold")
    ax1.legend(handles=[Patch(color=GCOL[g], label=GROUP_ZH[g]) for g in ("xbrl", "form4", "finra")] + [Patch(color="#e4e4e4", label=f"亂數 99% 帶（2013-24 {p99_1324:.3f}，2018-24 {p99_1824:.3f}）"),
                        Line2D([], [], color=GREYS[0], ls="--", lw=1, label="門檻 0.03")], loc="lower right", fontsize=8.5, frameon=False)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # panel 2: yearly OOS composite IC, technical vs tech_free vs tech_fund_free
    ax2 = fig.add_subplot(gs[1, 0])
    names = {"technical": ("技術 33 個（基準）", GREYS[0]), "free": ("免費新訊號 8 個", PAL[2]), "tech_free": ("技術 + 新訊號 8 個", PAL[0]), "tech_fund_free": ("技術 + 財報 15 + 新訊號 8", PAL[1])}
    for run, (lab, col) in names.items():
        w = wf[wf.run == run]
        if w.empty:
            continue
        # the all-feature ridge is defined every year; the |t| >= 2 selection picks nothing in most years from a 2010 start
        ax2.plot(w.year, w.ic_ridge_all, marker="o", ms=4, lw=1.6, color=col, label=f"{lab}：平均 {float(sv[f'oos_{run}_ridge_all_mean_ic']):+.3f}（t {float(sv[f'oos_{run}_ridge_all_t']):+.1f}）")
    ax2.axhline(0, color=GREYS[2], lw=0.8)
    ax2.axhline(A.IC_THRESHOLD, color=GREYS[0], ls="--", lw=1)
    ax2.axhline(A.GRID_THRESHOLD, color=PAL[3], ls=":", lw=1.2)
    ax2.annotate("達標線 0.15", (ax2.get_xlim()[1] - 0.3, A.GRID_THRESHOLD), xytext=(0, -9), textcoords="offset points", fontsize=8, color=PAL[3], ha="right")
    ax2.set_ylim(min(-0.08, ax2.get_ylim()[0]), 0.17)
    ax2.set_title("走動式合成（Ridge，全部候選特徵）逐年樣本外 IC，2013 至 2024 年", loc="left", fontsize=11, fontweight="bold")
    ax2.set_ylabel("年平均月度 IC", fontsize=9.5)
    ax2.legend(fontsize=8.3, frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # panel 3: paired monthly difference tech_free - technical, 12-month rolling mean
    ax3 = fig.add_subplot(gs[1, 1])
    a = monthly[monthly.run == "technical"].set_index("signal_date"); b = monthly[monthly.run == "tech_free"].set_index("signal_date")
    for col, lab, colr in (("ic_ridge_all", "Ridge（全部特徵）", PAL[1]),):
        d = (b[col] - a[col]).dropna().sort_index()
        if d.empty:
            continue
        ax3.plot(d.index, d.rolling(12).mean(), lw=1.6, color=colr, label=f"{lab}：平均 {float(sv[f'paired_tech_free_minus_technical_{col[3:]}_mean']):+.4f}（t {float(sv[f'paired_tech_free_minus_technical_{col[3:]}_t']):+.1f}）")
    ax3.axhline(0, color=GREYS[2], lw=0.8)
    ax3.set_title("加入 8 個新訊號後合成 IC 的增量（同月配對差，12 個月滾動平均）", loc="left", fontsize=11, fontweight="bold")
    ax3.set_ylabel("IC 差", fontsize=9.5)
    ax3.legend(fontsize=8.3, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    fig.suptitle("第 21 節：三種免費非價格資料（XBRL 資產負債、Form 4 內部人、FINRA 空單）的訊號稽核", fontsize=14, fontweight="bold", x=0.17, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
