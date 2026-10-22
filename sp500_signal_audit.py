#!/usr/bin/env python3
"""
Forward signal IC audit on the point-in-time S&P 500 universe (section 19.4 / 19.8, step 1).

Question. Before any strategy is built from a stock-selection feature, does the feature have a
measurable, stable cross-sectional rank correlation with NEXT-month returns? Section 19.4 says the
right order is: fix the label, use only data dated <= the signal date, score with the monthly IC
(not a back-test curve), compare with a null distribution of random signals, and judge against a
threshold fixed BEFORE looking. This script does exactly that for every candidate feature the repo
already computes, and then builds anchored walk-forward composites so the orchestrator can trade the
resulting score through the section-15 engine.

Universe and dates. load_universe() from sp500_precursor_events: point-in-time members with a price
and 252 days of history (eligible), prices 1998-06..2024-12, zero closes already NaN. Signal dates are
month ends t with >= 50 eligible names, 2000-01 .. 2024-12. Execution close = first trading day after t.

Targets. fwd1 = close at the execution day after the NEXT month end / close at this month's execution
day - 1 (the same convention as sp500_skill_needed.forward_returns); fwd3 = the same over three month
ends. Only stocks eligible at t carry a target. Both are strictly after the execution date (asserted).

Features (57 candidate rows: 37 technical + 15 fundamental + 5 momentum definitions as reference rows).
  technical    sp500_precursor_technical.compute_features(px, eligible, spy, volume): returns, ranks,
               rank changes, distance from 252-day low/high, relative strength, SMA ratios, trend age,
               volatility, drawdown, up-day share, volume ratios, spikes, dollar-volume rank. Every
               value at row t uses closes/volumes dated <= t (rolling windows, shifts, same-date ranks);
               re-asserted here by recomputing the whole set on a history truncated at 2012-06-29.
  fundamental  sp500_precursor_fundamental.build_features on precursor_fund_quarterly.csv (SEC EDGAR
               XBRL, first-filed values dated by `filed`): the latest quarter with filed <= t for every
               eligible stock each month end (asserted), 2010-2024 only (XBRL era). The 15th row is
               eps_ttm (trailing-4-quarter EPS, the numerator of ey_ttm) added here.
  momentum     sp500_top5_monthly.rank_frames: 21 / 63 / 252-day, 12-1 and the three-stage composite.

IC. Monthly Spearman rank IC = Pearson correlation of the within-month average ranks of feature and
target over the stocks that have both (>= 30 names, else the month is skipped). Per feature: mean,
std, t = mean / std * sqrt(months), IR = mean / std, share of months > 0, share of calendar years whose
mean IC > 0, mean IC by sub-period (2000-2009, 2010-2019, 2020-2024), the sign, and the sign-adjusted
(flipped if negative) mean and t. The same against fwd3 (overlapping windows: that t-stat is inflated by
roughly sqrt(3) and is labelled as such).

Pre-registered thresholds (fixed in docs/ANALYSIS.md 17.4 / 19.4 before this run; reported, not tuned):
  "real signal"   sign-adjusted mean monthly IC >= 0.03, t >= 2 and >= 8 of the 12 years 2013-2024 with a
                  positive yearly mean IC. `pass_real_signal` evaluates mean and t INSIDE 2013-2024;
                  `pass_real_signal_fullsample` uses the full-sample mean and t with the same year count.
  "grid-passing"  out-of-sample composite IC >= 0.15 (the 19.3 grid cell that reaches 30% / -30%).

Null distribution. --null-runs (200) random features, independent N(0,1) per eligible stock-month, seed
--seed, through the same IC pipeline: distribution of mean IC and t over the full sample and over
2013-2024; 95th / 99th percentiles of |mean IC| and |t|; and the Bonferroni line for 57 two-sided tests
at 5% (|t| >= 3.04).

Walk-forward composites (anchored, expanding, re-fitted every 1 January using only rows whose 1-month
target window ENDED by 31 December of the previous year):
  * candidates: technical features minus the within-month duplicates (r21/r63/r126/r252/rs21/rs63 rank
    identically to pr21/pr63/pr126/pr252 among eligible names) plus the 12-1 and composite momentum
    definitions (33); the "tech+fund" run adds the 15 fundamental rows (48) and runs on the covered
    universe (stocks with a filed quarter at t) with training from 2010-01.
  * selection: features whose training-sample |t| >= 2, sign = sign of the training mean IC.
  * ew     equal-weight mean of sign-adjusted within-month rank-normalised features ((rank-0.5)/n - 0.5,
           missing -> 0).
  * ridge  closed-form ridge (lambda --lam = 10, intercept unpenalised) of the cross-sectionally demeaned
           next-month return on the selected rank-normalised features; ridge_all = same on every candidate.
  * best_single  the feature with the largest training |t|, re-selected yearly ("pick the winner").
  Every eligible stock is scored each month of the test year; the out-of-sample monthly IC of each score
  is reported per year, overall (mean, t over months, years positive).
  Technical run: test years 2005-2024; tech+fund run: 2013-2024.

Outputs (utf-8-sig CSV):
  <out>_features.csv            one row per feature with every statistic and the pass flags
  <out>_null.csv                one row per random feature
  <out>_walkforward.csv         year x run with the OOS IC of ew / ridge / ridge_all / best_single
  <out>_walkforward_monthly.csv the monthly OOS IC series behind it
  <out>_summary.csv             key numbers (best feature, null percentiles, composite OOS IC, pass flags)
  <out>_scores.csv              OOS composite scores per eligible stock-month 2005-2024 (technical run;
                                tech+fund columns 2013-2024) with ranks, for the trading step
  <chart>                       docs/signal_audit.png

Usage: python sp500_signal_audit.py [--start 2000-01-01] [--end 2024-12-31] [--null-runs 200] [--seed 42]
                                    [--out signal_audit] [--chart docs/signal_audit.png] [--chart-only] [--lam 10]
"""

from __future__ import annotations

import argparse
import math
import os
import time
from statistics import NormalDist

import numpy as np
import pandas as pd

import sp500_precursor_fundamental as Fd
import sp500_precursor_technical as Tch
import sp500_top5_monthly as T5
from sp500_precursor_events import load_universe

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREYS = ["#555555", "#8a8a8a", "#b5b5b5", "#2b2b2b"]
MIN_NAMES = 30            # a month needs this many stocks with feature and target to count
IC_THRESHOLD = 0.03       # pre-registered "real signal" mean monthly IC
T_THRESHOLD = 2.0
YEARS_THRESHOLD = 8       # of the 12 years 2013-2024
GRID_THRESHOLD = 0.15     # pre-registered "grid-passing" OOS composite IC
PREREG_YEARS = (2013, 2024)
N_TESTS = 57
SUBPERIODS = [("2000_2009", "2000-01-01", "2009-12-31"), ("2010_2019", "2010-01-01", "2019-12-31"), ("2020_2024", "2020-01-01", "2024-12-31")]
MOM_LABELS = {"mom_short": "動能：短期 21 日", "mom_medium": "動能：中期 63 日", "mom_long": "動能：長期 252 日", "mom_12_1": "動能：12-1", "mom_composite": "動能：三階段合計"}
FUND_FEATURES = Fd.FEATURES + ["eps_ttm"]
FUND_LABELS = {**Fd.FEATURE_ZH, "eps_ttm": "近四季 EPS"}
# within-month duplicates of the percentile-rank features (identical Spearman IC); kept in the table, dropped from composites
COMPOSITE_DROP = set(Tch.MODEL_DROP) | {"mom_short", "mom_medium", "mom_long"}


# ----------------------------------------------------------------------------- IC machinery
def row_spearman(F: pd.DataFrame, T: pd.DataFrame, min_names: int = MIN_NAMES) -> tuple[pd.Series, pd.Series]:
    """Spearman rank correlation of each row of F with the same row of T (Pearson on average ranks, no scipy).
    Only cells where both are finite enter the ranks. Returns (ic, n) indexed like F."""
    valid = F.notna() & T.notna()
    n = valid.sum(axis=1)
    rf = F.where(valid).rank(axis=1)
    rt = T.where(valid).rank(axis=1)
    rf = rf.sub(rf.mean(axis=1), axis=0)
    rt = rt.sub(rt.mean(axis=1), axis=0)
    num = (rf * rt).sum(axis=1)
    den = np.sqrt((rf ** 2).sum(axis=1) * (rt ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / den
    ic[n < min_names] = np.nan
    return ic, n


def ic_summary(ic: pd.Series, prefix: str = "") -> dict:
    ic = ic.dropna()
    out = {f"{prefix}n_months": int(len(ic))}
    if len(ic) < 12:
        out.update({f"{prefix}mean_ic": np.nan, f"{prefix}std_ic": np.nan, f"{prefix}t": np.nan, f"{prefix}ir": np.nan, f"{prefix}pct_months_pos": np.nan, f"{prefix}pct_years_pos": np.nan, f"{prefix}n_years": 0})
        return out
    m, s = ic.mean(), ic.std(ddof=1)
    yearly = ic.groupby(ic.index.year).mean()
    out.update({f"{prefix}mean_ic": m, f"{prefix}std_ic": s, f"{prefix}t": m / s * math.sqrt(len(ic)), f"{prefix}ir": m / s,
                f"{prefix}pct_months_pos": (ic > 0).mean() * 100, f"{prefix}pct_years_pos": (yearly > 0).mean() * 100, f"{prefix}n_years": int(len(yearly))})
    return out


def feature_stats(ic1: pd.Series, ic3: pd.Series) -> dict:
    rec = ic_summary(ic1)
    m0 = rec["mean_ic"]
    sign = -1.0 if (np.isfinite(m0) and m0 < 0) else 1.0
    rec["sign"] = int(sign)
    rec["mean_ic_signed_abs"] = sign * rec["mean_ic"]
    rec["t_signed_abs"] = sign * rec["t"]
    for name, a, b in SUBPERIODS:
        sub = ic1.loc[a:b].dropna()
        rec[f"mean_ic_{name}"] = sub.mean() if len(sub) >= 12 else np.nan
        rec[f"n_months_{name}"] = int(len(sub))
    # pre-registered window 2013-2024, in the direction given by the full-sample sign
    win = (sign * ic1).loc[f"{PREREG_YEARS[0]}-01-01":f"{PREREG_YEARS[1]}-12-31"].dropna()
    w = ic_summary(win, "p1324_")
    yearly = win.groupby(win.index.year).mean()
    rec.update({"mean_ic_2013_2024_adj": w["p1324_mean_ic"], "t_2013_2024_adj": w["p1324_t"], "years_pos_2013_2024": int((yearly > 0).sum()), "n_years_2013_2024": int(len(yearly)),
                "n_months_2013_2024": w["p1324_n_months"]})
    rec["pass_real_signal"] = bool(len(win) >= 12 and w["p1324_mean_ic"] >= IC_THRESHOLD and w["p1324_t"] >= T_THRESHOLD and rec["years_pos_2013_2024"] >= YEARS_THRESHOLD)
    rec["pass_real_signal_fullsample"] = bool(rec["n_months"] >= 12 and rec["mean_ic_signed_abs"] >= IC_THRESHOLD and rec["t_signed_abs"] >= T_THRESHOLD and rec["years_pos_2013_2024"] >= YEARS_THRESHOLD)
    r3 = ic_summary(ic3, "fwd3_")
    rec.update({"mean_ic_3m": r3["fwd3_mean_ic"], "t_3m_overlapping": r3["fwd3_t"], "pct_months_pos_3m": r3["fwd3_pct_months_pos"], "n_months_3m": r3["fwd3_n_months"]})
    return rec


def rank_norm(fr: pd.DataFrame, mask: pd.DataFrame) -> np.ndarray:
    """Within-row rank mapped to (-0.5, 0.5) over the masked cells, missing -> 0 (median)."""
    v = fr.where(mask)
    r = v.rank(axis=1)
    n = v.notna().sum(axis=1).replace(0, np.nan)
    z = r.sub(0.5).div(n, axis=0) - 0.5
    return z.fillna(0.0).to_numpy()


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    R = lam * np.eye(p + 1)
    R[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + R, Xb.T @ y)


def spearman_arrays(a: np.ndarray, b: np.ndarray, min_names: int = MIN_NAMES) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_names:
        return np.nan
    ra, rb = pd.Series(a[m]).rank().to_numpy(), pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------------- data assembly
def targets(px: pd.DataFrame, eligible: pd.DataFrame, mes: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """fwd1 / fwd3 (signal date x ticker) from the execution close after t to the execution close after the
    next (third) month end, eligible names only; plus the execution date and the 1-month target end date per signal."""
    idx = px.index
    pos = np.array([idx.get_loc(d) for d in mes])
    exec_pos = pos + 1
    f1 = pd.DataFrame(np.nan, index=mes, columns=px.columns)
    f3 = f1.copy()
    exec_date = pd.Series(pd.NaT, index=mes, dtype="datetime64[ns]")
    end1 = pd.Series(pd.NaT, index=mes, dtype="datetime64[ns]")
    for k, d in enumerate(mes):
        a = exec_pos[k]
        if a >= len(idx):
            continue
        exec_date.iloc[k] = idx[a]
        for h, frame, store_end in ((1, f1, True), (3, f3, False)):
            if k + h >= len(mes):
                continue
            b = exec_pos[k + h]
            if b >= len(idx):
                continue
            assert a > pos[k] and b > a, "target window must start after the signal and end after execution"
            frame.iloc[k] = (px.iloc[b] / px.iloc[a] - 1).where(eligible.iloc[pos[k]]).to_numpy()
            if store_end:
                end1.iloc[k] = idx[b]
    return f1, f3, exec_date, end1


def fundamental_frames(px: pd.DataFrame, eligible: pd.DataFrame, spy: pd.Series, mes: list[pd.Timestamp], path: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    q = pd.read_csv(path, parse_dates=["filed", "period_end"])
    fe = Fd.build_features(q, px, eligible, spy, [d for d in mes if d >= pd.Timestamp("2009-12-01")])
    assert (fe["filed"] <= fe["signal_date"]).all(), "look-ahead: a quarter filed after the signal date was used"
    fe["eps_ttm"] = fe["ey_ttm"] * px.to_numpy()[px.index.get_indexer(fe.signal_date), px.columns.get_indexer(fe.ticker)]
    frames = {}
    for f in FUND_FEATURES:
        frames[f] = fe.pivot(index="signal_date", columns="ticker", values=f).reindex(index=mes, columns=px.columns)
    covered = fe.pivot(index="signal_date", columns="ticker", values="days_since_filing").reindex(index=mes, columns=px.columns).notna()
    print(f"fundamentals: {len(fe)} (month end, ticker) rows, {fe.ticker.nunique()} tickers, {fe.signal_date.min().date()}..{fe.signal_date.max().date()}; "
          f"median coverage of eligible names {np.nanmedian((covered.sum(axis=1) / eligible.reindex(mes).sum(axis=1)).loc['2010-01-01':]) * 100:.0f}%")
    return frames, covered


def truncation_check(px, eligible, spy, vol, F: dict[str, pd.DataFrame], T: pd.Timestamp) -> None:
    Ft = Tch.compute_features(px.loc[:T], eligible.loc[:T], spy.loc[:T], vol.loc[:T])
    bad = []
    for k in F:
        a, b = F[k].loc[T].to_numpy(), Ft[k].loc[T].to_numpy()
        same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9)
        if not same.all():
            bad.append((k, int((~same).sum())))
    assert not bad, f"technical features differ when the history is truncated at {T.date()}: {bad}"
    print(f"truncation check at {T.date()}: all {len(F)} technical features identical with and without later data (no look-ahead)")


# ----------------------------------------------------------------------------- walk-forward
def walk_forward(run: str, frames: dict[str, pd.DataFrame], cands: list[str], mask: pd.DataFrame, fwd1: pd.DataFrame, end1: pd.Series,
                 mes: list[pd.Timestamp], test_years: list[int], train_start: pd.Timestamp, lam: float):
    """Anchored yearly re-fit; returns (yearly rows, monthly OOS IC frame, long score table)."""
    mes_idx = pd.DatetimeIndex(mes)
    M = mask.to_numpy()
    rows_i, cols_i = np.where(M)
    X = np.empty((len(rows_i), len(cands)))
    for j, name in enumerate(cands):
        X[:, j] = rank_norm(frames[name], mask)[rows_i, cols_i]
    y = fwd1.where(mask).to_numpy()[rows_i, cols_i]
    sig = mes_idx[rows_i]
    y_dm = y - pd.Series(y).groupby(rows_i).transform("mean").to_numpy()
    # per-feature monthly IC inside this run's universe (for selection and the best-single baseline)
    ic_mat = pd.DataFrame({name: row_spearman(frames[name].where(mask), fwd1.where(mask))[0] for name in cands}, index=mes_idx)
    has_target = fwd1.where(mask).notna().sum(axis=1) >= MIN_NAMES
    yearly_rows, monthly = [], []
    score_ew = np.full(len(rows_i), np.nan)
    score_ridge = np.full(len(rows_i), np.nan)
    score_ridge_all = np.full(len(rows_i), np.nan)
    n_sel_row = np.zeros(len(rows_i), dtype=int)
    for Y in test_years:
        cutoff = pd.Timestamp(f"{Y - 1}-12-31")
        train_months = mes_idx[(end1.reindex(mes_idx) <= cutoff).to_numpy() & (mes_idx >= train_start) & has_target.to_numpy()]
        test_months = mes_idx[mes_idx.year == Y]
        assert train_months.max() < test_months.min() and end1.loc[train_months].max() <= cutoff, "training window leaks into the test year"
        tr_ic = ic_mat.loc[train_months]
        mean, std, n = tr_ic.mean(), tr_ic.std(ddof=1), tr_ic.notna().sum()
        tstat = (mean / std * np.sqrt(n)).replace([np.inf, -np.inf], np.nan)
        selected = [c for c in cands if abs(tstat[c]) >= T_THRESHOLD]
        signs = np.sign(mean.reindex(selected)).to_numpy()
        best = tstat.abs().idxmax()
        best_sign = 1.0 if mean[best] >= 0 else -1.0
        tr_rows = np.isin(sig.to_numpy(), train_months.to_numpy()) & np.isfinite(y)
        te_rows = np.isin(sig.to_numpy(), test_months.to_numpy())
        n_sel_row[te_rows] = len(selected)
        beta_all = fit_ridge(X[tr_rows], y_dm[tr_rows], lam)
        score_ridge_all[te_rows] = beta_all[0] + X[te_rows] @ beta_all[1:]
        beta = None
        if selected:
            cols = [cands.index(c) for c in selected]
            score_ew[te_rows] = (X[te_rows][:, cols] * signs).mean(axis=1)
            beta = fit_ridge(X[tr_rows][:, cols], y_dm[tr_rows], lam)
            score_ridge[te_rows] = beta[0] + X[te_rows][:, cols] @ beta[1:]
        for d in test_months:
            m = (sig == d) & te_rows
            if not has_target.loc[d]:
                continue
            monthly.append({"run": run, "signal_date": d, "year": Y, "ic_ew": spearman_arrays(score_ew[m], y[m]), "ic_ridge": spearman_arrays(score_ridge[m], y[m]),
                            "ic_ridge_all": spearman_arrays(score_ridge_all[m], y[m]), "ic_best_single": best_sign * ic_mat.loc[d, best], "n_names": int(np.isfinite(y[m]).sum())})
        ym = pd.DataFrame([r for r in monthly if r["year"] == Y])
        yearly_rows.append({"run": run, "year": Y, "n_train_months": len(train_months), "n_train_rows": int(tr_rows.sum()), "train_last_signal": train_months.max().date(),
                            "n_selected": len(selected), "selected": "|".join(f"{'+' if s > 0 else '-'}{c}" for c, s in zip(selected, signs)),
                            "best_single_feature": best, "best_single_t_insample": float(tstat[best]), "best_single_ic_insample": float(mean[best]),
                            "ic_ew": ym.ic_ew.mean() if len(ym) else np.nan, "ic_ridge": ym.ic_ridge.mean() if len(ym) else np.nan,
                            "ic_ridge_all": ym.ic_ridge_all.mean() if len(ym) else np.nan, "ic_best_single": ym.ic_best_single.mean() if len(ym) else np.nan, "n_test_months": int(len(ym)),
                            "ridge_top_coef": "|".join(f"{c}:{b:+.3f}" for c, b in sorted(zip(selected, beta[1:]), key=lambda t: -abs(t[1]))[:5]) if beta is not None else ""})
        print(f"  [{run}] {Y}: train {len(train_months)} months to {train_months.max().date()}, {len(selected)} selected, best {best} (t {tstat[best]:+.1f}) | OOS IC ew {yearly_rows[-1]['ic_ew']:+.3f}  ridge {yearly_rows[-1]['ic_ridge']:+.3f}  ridge_all {yearly_rows[-1]['ic_ridge_all']:+.3f}  best_single {yearly_rows[-1]['ic_best_single']:+.3f}", flush=True)
    scored = np.isin(sig.to_numpy(), mes_idx[mes_idx.year.isin(test_years)].to_numpy())
    scores = pd.DataFrame({"signal_date": sig[scored], "ticker": np.asarray(mask.columns)[cols_i[scored]], "score_ew": score_ew[scored], "score_ridge": score_ridge[scored],
                           "score_ridge_all": score_ridge_all[scored], "n_selected": n_sel_row[scored]})
    return pd.DataFrame(yearly_rows), pd.DataFrame(monthly), scores


def oos_summary(monthly: pd.DataFrame, col: str) -> dict:
    s = monthly.set_index("signal_date")[col].dropna()
    if len(s) < 12:
        return {"mean": np.nan, "t": np.nan, "years_pos": 0, "n_years": 0, "n_months": int(len(s))}
    yearly = s.groupby(s.index.year).mean()
    return {"mean": s.mean(), "t": s.mean() / s.std(ddof=1) * math.sqrt(len(s)), "years_pos": int((yearly > 0).sum()), "n_years": int(len(yearly)), "n_months": int(len(s))}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--null-runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="signal_audit")
    p.add_argument("--chart", default="docs/signal_audit.png")
    p.add_argument("--chart-only", action="store_true", help="redraw the chart from the saved CSVs")
    p.add_argument("--lam", type=float, default=10.0, help="ridge penalty on rank-normalised features")
    p.add_argument("--quarterly", default="precursor_fund_quarterly.csv")
    args = p.parse_args(argv)
    if args.chart_only:
        chart(pd.read_csv(f"{args.out}_features.csv"), pd.read_csv(f"{args.out}_null.csv"), pd.read_csv(f"{args.out}_walkforward.csv"), pd.read_csv(f"{args.out}_summary.csv"), args.chart)
        print(f"wrote {args.chart}")
        return 0

    t0 = time.time()
    px, member, eligible, spy = load_universe(args.start, args.end)
    all_mes = T5.month_ends(px.index)
    mes = [d for d in all_mes if d >= pd.Timestamp(args.start) and d <= pd.Timestamp(args.end) and eligible.loc[d].sum() >= 50]
    mes_idx = pd.DatetimeIndex(mes)
    elig_mes = eligible.reindex(mes)
    print(f"universe {px.shape[1]} tickers {px.index[0].date()}..{px.index[-1].date()}; {len(mes)} signal month-ends {mes[0].date()}..{mes[-1].date()}, "
          f"median {int(elig_mes.sum(axis=1).median())} eligible names per month")

    # 1. targets -----------------------------------------------------------------------------------
    fwd1, fwd3, exec_date, end1 = targets(px, eligible, mes)
    e1, ex = end1.dropna(), exec_date.dropna()
    assert (e1.to_numpy() > e1.index.to_numpy()).all() and (ex.to_numpy() > ex.index.to_numpy()).all(), "targets must lie strictly after the signal date"
    assert (e1.to_numpy() > ex.reindex(e1.index).to_numpy()).all(), "target end must lie strictly after the execution date"
    n_t1 = int(fwd1.notna().any(axis=1).sum())
    print(f"targets: {n_t1} months with a 1-month target (last signal {fwd1.dropna(how='all').index[-1].date()}), {int(fwd3.notna().any(axis=1).sum())} with a 3-month target")

    # 2. features ----------------------------------------------------------------------------------
    vol = Tch.load_volume(list(px.columns))
    F = Tch.compute_features(px, eligible, spy, vol)
    truncation_check(px, eligible, spy, vol, F, pd.Timestamp("2012-06-29"))
    frames: dict[str, pd.DataFrame] = {k: fr.reindex(mes) for k, fr in F.items()}
    groups = {k: "technical" for k in F}
    labels = {k: Tch.LABELS.get(k, k) for k in F}
    R = T5.rank_frames(px, eligible)
    for k, fr in R.items():
        frames[f"mom_{k}"] = fr.reindex(mes); groups[f"mom_{k}"] = "momentum"; labels[f"mom_{k}"] = MOM_LABELS[f"mom_{k}"]
    fund, covered = fundamental_frames(px, eligible, spy, mes, args.quarterly)
    for k, fr in fund.items():
        frames[k] = fr; groups[k] = "fundamental"; labels[k] = FUND_LABELS.get(k, k)
    feats = list(frames)
    print(f"{len(feats)} candidate features ({sum(g == 'technical' for g in groups.values())} technical, {sum(g == 'fundamental' for g in groups.values())} fundamental, "
          f"{sum(g == 'momentum' for g in groups.values())} momentum) in {time.time() - t0:.0f}s")

    # 3. per-feature IC ----------------------------------------------------------------------------
    ic_rows, ic_series = [], {}
    for k in feats:
        ic1, n1 = row_spearman(frames[k].where(elig_mes), fwd1)
        ic3, _ = row_spearman(frames[k].where(elig_mes), fwd3)
        ic_series[k] = ic1
        rec = {"feature": k, "label": labels[k], "group": groups[k], "mean_names": float(n1[n1 > 0].mean()) if (n1 > 0).any() else 0.0,
               "first_month": ic1.dropna().index.min(), "last_month": ic1.dropna().index.max(), **feature_stats(ic1, ic3)}
        ic_rows.append(rec)
    features = pd.DataFrame(ic_rows)
    features["in_composite_pool"] = ~features.feature.isin(COMPOSITE_DROP)

    # 4. null distribution -------------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for i in range(args.null_runs):
        Z = pd.DataFrame(rng.standard_normal(elig_mes.shape), index=mes_idx, columns=px.columns).where(elig_mes)
        ic1, _ = row_spearman(Z, fwd1)
        rec = feature_stats(ic1, pd.Series(np.nan, index=mes_idx))
        rec = {k: v for k, v in rec.items() if not k.startswith("fwd3") and "3m" not in k}
        null_rows.append({"run": i, **rec})
    null = pd.DataFrame(null_rows)
    q_abs_ic = null.mean_ic.abs().quantile([0.95, 0.99])
    q_abs_t = null.t.abs().quantile([0.95, 0.99])
    # window statistics carry the full-sample sign (same procedure as for the real features), so they are sign-adjusted, not absolute
    q_abs_ic_1324 = null.mean_ic_2013_2024_adj.quantile([0.95, 0.99])
    q_abs_t_1324 = null.t_2013_2024_adj.quantile([0.95, 0.99])
    bonf_z = NormalDist().inv_cdf(1 - 0.05 / N_TESTS / 2)
    null_pass = int(null.pass_real_signal.sum())
    print(f"\nnull ({args.null_runs} random features, seed {args.seed}): |mean IC| p95 {q_abs_ic[0.95]:.4f} p99 {q_abs_ic[0.99]:.4f} max {null.mean_ic.abs().max():.4f}; "
          f"|t| p95 {q_abs_t[0.95]:.2f} p99 {q_abs_t[0.99]:.2f} max {null.t.abs().max():.2f}; Bonferroni |t| for {N_TESTS} tests at 5%: {bonf_z:.2f}; "
          f"2013-2024 (sign-adjusted): mean IC p99 {q_abs_ic_1324[0.99]:.4f}, t p99 {q_abs_t_1324[0.99]:.2f}; null features passing the real-signal threshold: {null_pass}")
    features["above_null_p99_abs_ic"] = features.mean_ic_signed_abs > q_abs_ic[0.99]
    features["above_null_p99_abs_t"] = features.t_signed_abs > q_abs_t[0.99]
    features["above_bonferroni_t"] = features.t_signed_abs > bonf_z
    features = features.sort_values("t_signed_abs", ascending=False).reset_index(drop=True)
    features.to_csv(f"{args.out}_features.csv", index=False, encoding="utf-8-sig")
    null.to_csv(f"{args.out}_null.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    show = ["feature", "group", "n_months", "mean_ic", "t", "pct_months_pos", "pct_years_pos", "mean_ic_2000_2009", "mean_ic_2010_2019", "mean_ic_2020_2024",
            "mean_ic_2013_2024_adj", "t_2013_2024_adj", "years_pos_2013_2024", "pass_real_signal", "mean_ic_3m"]
    print("\n-- features sorted by sign-adjusted t (full sample)")
    print(features[show].round(4).to_string(index=False))

    # 5. walk-forward composites -------------------------------------------------------------------
    end_year = pd.Timestamp(args.end).year
    cands_tech = [k for k in feats if groups[k] != "fundamental" and k not in COMPOSITE_DROP]
    cands_all = cands_tech + [k for k in feats if groups[k] == "fundamental"]
    print(f"\nwalk-forward: technical run {len(cands_tech)} candidates, test years 2005-{end_year}; tech+fund run {len(cands_all)} candidates on the covered universe, test years 2013-{end_year}")
    wf_tech, mo_tech, sc_tech = walk_forward("technical", frames, cands_tech, elig_mes, fwd1, end1, mes, list(range(2005, end_year + 1)), mes_idx[0], args.lam)
    cov_mask = elig_mes & covered
    wf_all, mo_all, sc_all = walk_forward("tech_fund", frames, cands_all, cov_mask, fwd1, end1, mes, list(range(2013, end_year + 1)), pd.Timestamp("2010-01-01"), args.lam)
    wf = pd.concat([wf_tech, wf_all], ignore_index=True)
    monthly = pd.concat([mo_tech, mo_all], ignore_index=True)
    wf.to_csv(f"{args.out}_walkforward.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(f"{args.out}_walkforward_monthly.csv", index=False, encoding="utf-8-sig")
    comp_stats = {}
    for run, mo in (("technical", mo_tech), ("tech_fund", mo_all)):
        for col in ("ic_ew", "ic_ridge", "ic_ridge_all", "ic_best_single"):
            comp_stats[(run, col)] = oos_summary(mo, col)
    print("\n-- out-of-sample composite IC (monthly, over all test months)")
    for (run, col), s in comp_stats.items():
        print(f"  {run:10s} {col:15s} mean {s['mean']:+.4f}  t {s['t']:+.2f}  years>0 {s['years_pos']}/{s['n_years']}  months {s['n_months']}")

    # 6. scores panel ------------------------------------------------------------------------------
    sc = sc_tech.merge(sc_all.rename(columns={"score_ew": "score_ew_fund", "score_ridge": "score_ridge_fund", "score_ridge_all": "score_ridge_all_fund", "n_selected": "n_selected_fund"}),
                       on=["signal_date", "ticker"], how="left")
    sc.insert(1, "exec_date", exec_date.reindex(sc.signal_date).to_numpy())
    for c in ("score_ew", "score_ridge"):
        sc[f"rank_{c[6:]}"] = sc.groupby("signal_date")[c].rank(ascending=False, method="first")
    sc = sc.sort_values(["signal_date", "rank_ridge"]).reset_index(drop=True)
    sc.to_csv(f"{args.out}_scores.csv", index=False, encoding="utf-8-sig")
    size_mb = os.path.getsize(f"{args.out}_scores.csv") / 1e6
    if size_mb > 50:
        keep = sc[(sc.rank_ew <= 50) | (sc.rank_ridge <= 50) | (sc.rank_ew > sc.groupby("signal_date").rank_ew.transform("max") - 50) | (sc.rank_ridge > sc.groupby("signal_date").rank_ridge.transform("max") - 50)]
        keep.to_csv(f"{args.out}_scores.csv", index=False, encoding="utf-8-sig")
        print(f"scores panel {size_mb:.0f} MB > 50 MB: kept only the top/bottom 50 per month ({len(keep)} rows)")
    else:
        print(f"scores panel: {len(sc)} eligible stock-months {sc.signal_date.min().date()}..{sc.signal_date.max().date()}, {size_mb:.1f} MB")

    # 7. summary -----------------------------------------------------------------------------------
    best = features.iloc[0]
    best_f = features[features.group == "fundamental"].iloc[0]
    best_1324 = features.sort_values("t_2013_2024_adj", ascending=False).iloc[0]
    rows = [("signal_months", len(mes)), ("months_with_1m_target", n_t1), ("n_features", len(feats)), ("n_tests_bonferroni", N_TESTS),
            ("best_feature_by_t", best.feature), ("best_feature_label", best.label), ("best_feature_mean_ic", best.mean_ic), ("best_feature_t", best.t),
            ("best_feature_sign", best.sign), ("best_feature_pct_years_pos", best.pct_years_pos), ("best_feature_mean_ic_2013_2024_adj", best.mean_ic_2013_2024_adj),
            ("best_feature_years_pos_2013_2024", best.years_pos_2013_2024),
            ("best_2013_2024_feature", best_1324.feature), ("best_2013_2024_mean_ic_adj", best_1324.mean_ic_2013_2024_adj), ("best_2013_2024_t_adj", best_1324.t_2013_2024_adj), ("best_2013_2024_years_pos", best_1324.years_pos_2013_2024),
            ("best_fundamental_feature", best_f.feature), ("best_fundamental_mean_ic", best_f.mean_ic), ("best_fundamental_t", best_f.t),
            ("max_abs_mean_ic_any_feature", features.mean_ic_signed_abs.max()),
            ("null_runs", args.null_runs), ("null_abs_mean_ic_p95", q_abs_ic[0.95]), ("null_abs_mean_ic_p99", q_abs_ic[0.99]), ("null_abs_mean_ic_max", null.mean_ic.abs().max()),
            ("null_abs_t_p95", q_abs_t[0.95]), ("null_abs_t_p99", q_abs_t[0.99]), ("null_abs_t_max", null.t.abs().max()),
            ("null_2013_2024_adj_mean_ic_p99", q_abs_ic_1324[0.99]), ("null_2013_2024_adj_t_p99", q_abs_t_1324[0.99]), ("null_features_passing_real_signal", null_pass),
            ("bonferroni_abs_t_57_tests", bonf_z), ("n_features_above_null_p99_abs_t", int(features.above_null_p99_abs_t.sum())), ("n_features_above_bonferroni", int(features.above_bonferroni_t.sum())),
            ("n_features_pass_real_signal_2013_2024", int(features.pass_real_signal.sum())), ("features_pass_real_signal_2013_2024", "|".join(features[features.pass_real_signal].feature)),
            ("n_features_pass_real_signal_fullsample", int(features.pass_real_signal_fullsample.sum())), ("features_pass_real_signal_fullsample", "|".join(features[features.pass_real_signal_fullsample].feature)),
            ("threshold_real_signal", f"mean IC >= {IC_THRESHOLD}, t >= {T_THRESHOLD}, >= {YEARS_THRESHOLD}/12 years positive 2013-2024"), ("threshold_grid_passing", f"OOS composite IC >= {GRID_THRESHOLD}")]
    for (run, col), s in comp_stats.items():
        key = f"oos_{run}_{col[3:]}"
        rows += [(f"{key}_mean_ic", s["mean"]), (f"{key}_t", s["t"]), (f"{key}_years_pos", f"{s['years_pos']}/{s['n_years']}"),
                 (f"{key}_pass_0.03", bool(s["mean"] >= IC_THRESHOLD) if np.isfinite(s["mean"]) else False), (f"{key}_pass_0.15", bool(s["mean"] >= GRID_THRESHOLD) if np.isfinite(s["mean"]) else False)]
    rows += [("oos_technical_years_with_no_selection", int((wf_tech.n_selected == 0).sum())), ("oos_tech_fund_years_with_no_selection", int((wf_all.n_selected == 0).sum())),
             ("lam", args.lam), ("seed", args.seed), ("runtime_s", round(time.time() - t0))]
    summary = pd.DataFrame(rows, columns=["key", "value"])
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    print("\n-- walk-forward by year")
    print(wf[["run", "year", "n_train_months", "n_selected", "best_single_feature", "best_single_t_insample", "ic_ew", "ic_ridge", "ic_ridge_all", "ic_best_single"]].round(3).to_string(index=False))
    chart(features, null, wf, summary, args.chart)
    print(f"\nwrote {args.out}_features.csv, _null.csv, _walkforward.csv, _walkforward_monthly.csv, _summary.csv, _scores.csv, {args.chart} ({time.time() - t0:.0f}s)")
    return 0


# ----------------------------------------------------------------------------- chart
def chart(features: pd.DataFrame, null: pd.DataFrame, wf: pd.DataFrame, summary: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    sv = dict(zip(summary.key, summary.value))
    gcol = {"technical": PAL[0], "fundamental": PAL[1], "momentum": PAL[2]}
    gname = {"technical": "技術面（37）", "fundamental": "基本面（15，2010 起）", "momentum": "動能定義（5，參考）"}
    p99_ic = float(sv["null_abs_mean_ic_p99"]); p99_t = float(sv["null_abs_t_p99"]); bonf = float(sv["bonferroni_abs_t_57_tests"])
    fig = plt.figure(figsize=(19, 15.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.0, 1.0], hspace=0.42, wspace=0.16, left=0.13, right=0.985, top=0.93, bottom=0.05)
    ax1 = fig.add_subplot(gs[:, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 1]); ax4 = fig.add_subplot(gs[2, 1])

    # (1) mean monthly IC, every feature, sorted
    f = features.sort_values("mean_ic").reset_index(drop=True)
    y = np.arange(len(f))
    ax1.axvspan(-p99_ic, p99_ic, color="#dddddd", alpha=0.7, lw=0, zorder=0)
    for v, ls, lab in ((IC_THRESHOLD, "--", f"預設門檻 ±{IC_THRESHOLD}"), (-IC_THRESHOLD, "--", None)):
        ax1.axvline(v, color=GREYS[0], ls=ls, lw=1, zorder=1)
    ax1.axvline(0, color=GREYS[1], lw=0.8, zorder=1)
    ax1.barh(y, f.mean_ic, color=[gcol[g] for g in f.group], height=0.7, zorder=2)
    for yi, (v, t, ok) in enumerate(zip(f.mean_ic, f.t, f.pass_real_signal)):
        ax1.annotate(f"{v:+.3f} (t {t:+.1f}){' ＊通過' if ok else ''}", (v, yi), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points", ha="left" if v >= 0 else "right", va="center", fontsize=6.8, color="#333333", zorder=3)
    ax1.set_yticks(y); ax1.set_yticklabels(f.label, fontsize=7.6)
    ax1.tick_params(axis="y", length=0)
    lim = max(abs(f.mean_ic.min()), abs(f.mean_ic.max()), IC_THRESHOLD) * 1.75
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-0.8, len(f) - 0.2)
    ax1.set_xlabel("月度 Spearman IC 平均（對下月報酬，全樣本）", fontsize=9.5)
    ax1.set_title(f"(1) 57 個候選特徵對下月報酬的月均 IC（2000 至 2024；基本面 2010 起）\n灰帶 = 200 個亂數特徵 |IC| 的第 99 百分位 ±{p99_ic:.3f}；虛線 = 預設門檻 ±0.03；＊ = 通過預設「真訊號」門檻", loc="left", fontsize=10.5)
    ax1.legend(handles=[Patch(color=gcol[g], label=gname[g]) for g in ("technical", "fundamental", "momentum")] + [Patch(color="#dddddd", label="亂數特徵 99% 帶"), Line2D([], [], color=GREYS[0], ls="--", lw=1, label="±0.03 預設門檻")],
               frameon=False, fontsize=8.2, loc="lower right")
    ax1.grid(axis="x", alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)

    # (2) t vs mean IC, null cloud
    ax2.scatter(null.mean_ic, null.t, s=14, color="#bbbbbb", alpha=0.7, label="200 個亂數特徵", zorder=1)
    for g in ("technical", "fundamental", "momentum"):
        d = features[features.group == g]
        ax2.scatter(d.mean_ic, d.t, s=34, color=gcol[g], edgecolor="white", lw=0.5, label=gname[g], zorder=3)
    for v in (p99_t, -p99_t):
        ax2.axhline(v, color=GREYS[1], ls=":", lw=1)
    for v in (bonf, -bonf):
        ax2.axhline(v, color=GREYS[0], ls="--", lw=1)
    for v in (IC_THRESHOLD, -IC_THRESHOLD):
        ax2.axvline(v, color=GREYS[0], ls="--", lw=1)
    xl = ax2.get_xlim(); yl = ax2.get_ylim()
    ax2.set_ylim(yl[0] - 0.4, yl[1] + 0.4)
    ax2.annotate(f"亂數 |t| 第 99 百分位 {p99_t:.2f}", (xl[1], p99_t), xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=7.8, color=GREYS[1])
    ax2.annotate(f"Bonferroni 57 檢定 |t| {bonf:.2f}", (xl[1], bonf), xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=7.8, color=GREYS[0])
    # label the 7 strongest features; labels are stacked to the right with a minimum vertical gap and a thin connector
    # the strongest features all sit in the lower-left (negative IC); their labels are stacked in the empty upper-left corner
    top = features.sort_values("t_signed_abs", ascending=False).head(7).sort_values("t")
    gap = 0.4
    x_lab = xl[0] + (xl[1] - xl[0]) * 0.02
    y0 = -0.9 if top.t.mean() < 0 else 0.9
    for i, (_, r) in enumerate(top.iterrows()):
        y = y0 + i * gap if top.t.mean() < 0 else y0 - i * gap
        ax2.annotate(r.label, (r.mean_ic, r.t), xytext=(x_lab, y), textcoords="data", fontsize=7.6, color=gcol[r.group], va="center", ha="left",
                     arrowprops={"arrowstyle": "-", "color": gcol[r.group], "lw": 0.5, "alpha": 0.5, "shrinkB": 2})
    ax2.set_xlabel("月均 IC"); ax2.set_ylabel("t 統計量（月均 / 標準差 × √月數）")
    ax2.set_title("(2) t 統計量對月均 IC：真實特徵對亂數雲（點 = 一個特徵）", loc="left", fontsize=10.5)
    ax2.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=8, loc="lower right")
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)

    # (3) walk-forward OOS IC by year
    wt = wf[wf.run == "technical"].sort_values("year")
    wa = wf[wf.run == "tech_fund"].sort_values("year")
    x = np.arange(len(wt)); w = 0.27
    series = [("ic_ew", "等權合成（|t|≥2 的特徵）", PAL[0]), ("ic_ridge", "Ridge 合成（λ=10）", PAL[1]), ("ic_best_single", "每年挑樣本內最佳單一特徵", PAL[3])]
    for j, (col, lab, c) in enumerate(series):
        ax3.bar(x + (j - 1) * w, wt[col].fillna(0), w, color=c, label=lab)
    if len(wa):
        xa = np.array([list(wt.year).index(yv) for yv in wa.year])
        ax3.plot(xa, wa.ic_ridge, "D", color=GREYS[3], ms=4.5, label="技術 + 基本面 Ridge（2013 起，覆蓋股池）", zorder=4)
    ax3.axhline(0, color=GREYS[1], lw=0.8)
    ax3.axhline(IC_THRESHOLD, color=GREYS[0], ls="--", lw=1); ax3.axhline(GRID_THRESHOLD, color=PAL[5], ls="--", lw=1)
    ax3.annotate("預設門檻 0.03", (len(wt) - 0.5, IC_THRESHOLD), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8, color=GREYS[0])
    ax3.annotate("網格達標線 0.15（30% / -30% 需要的 IC）", (len(wt) - 0.5, GRID_THRESHOLD), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8, color=PAL[5])
    ax3.set_xticks(x); ax3.set_xticklabels(wt.year.astype(int), fontsize=8, rotation=0)
    ymin = min(-0.06, float(np.nanmin(wt[[c for c, _, _ in series]].to_numpy())) * 1.25)
    ax3.set_ylim(ymin, max(0.24, float(np.nanmax(wt[[c for c, _, _ in series]].to_numpy())) * 1.3))
    txt = "\n".join(f"{lab}：{float(sv[f'oos_technical_{col[3:]}_mean_ic']):+.3f}（t {float(sv[f'oos_technical_{col[3:]}_t']):+.1f}，{sv[f'oos_technical_{col[3:]}_years_pos']} 年為正）" for col, lab, _ in series)
    if len(wa):
        txt += f"\n技術 + 基本面 Ridge：{float(sv['oos_tech_fund_ridge_mean_ic']):+.3f}（t {float(sv['oos_tech_fund_ridge_t']):+.1f}，{sv['oos_tech_fund_ridge_years_pos']} 年為正）"
    ax3.text(0.01, 0.97, txt, transform=ax3.transAxes, va="top", ha="left", fontsize=8, color="#333333")
    ax3.set_ylabel("樣本外月均 IC")
    ax3.set_title("(3) 錨定走動式合成訊號：每年 1 月只用前一年底前已知的報酬重選、重估，當年逐月樣本外 IC", loc="left", fontsize=10.5)
    ax3.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=4)
    ax3.grid(axis="y", alpha=0.25); ax3.spines[["top", "right"]].set_visible(False)

    # (4) sub-period IC of the top 8 features (sign-adjusted)
    top8 = features.sort_values("t_signed_abs", ascending=False).head(8)
    x = np.arange(len(top8)); w = 0.26
    subs = [("mean_ic_2000_2009", "2000 至 2009", PAL[0]), ("mean_ic_2010_2019", "2010 至 2019", PAL[1]), ("mean_ic_2020_2024", "2020 至 2024", PAL[2])]
    for j, (col, lab, c) in enumerate(subs):
        vals = (top8[col] * top8.sign).to_numpy()
        bars = ax4.bar(x + (j - 1) * w, np.nan_to_num(vals), w, color=c, label=lab)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                ax4.annotate("無資料", (b.get_x() + b.get_width() / 2, 0), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.5, color=GREYS[1], rotation=90)
            else:
                ax4.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -9), textcoords="offset points", ha="center", fontsize=6.8)
    ax4.axhline(0, color=GREYS[1], lw=0.8); ax4.axhline(IC_THRESHOLD, color=GREYS[0], ls="--", lw=1)
    ax4.set_xticks(x); ax4.set_xticklabels([f"{l}\n（{'＋' if s > 0 else '－'}）" for l, s in zip(top8.label, top8.sign)], fontsize=8)
    ax4.set_ylabel("月均 IC（已依全樣本符號調整）")
    ax4.set_title("(4) 全樣本 |t| 最高的 8 個特徵，分三個子期間的月均 IC（括號 = 使用方向；虛線 = 0.03）", loc="left", fontsize=10.5)
    ax4.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    vals_all = (top8[[c for c, _, _ in subs]].mul(top8.sign, axis=0)).to_numpy()
    ax4.set_ylim(min(-0.02, np.nanmin(vals_all) * 1.4), max(0.06, np.nanmax(vals_all) * 1.45))
    ax4.grid(axis="y", alpha=0.25); ax4.spines[["top", "right"]].set_visible(False)

    n_pass = int(sv["n_features_pass_real_signal_2013_2024"])
    fig.suptitle(f"訊號 IC 稽核：{int(sv['n_features'])} 個候選選股特徵對下月報酬的月度等級 IC、亂數對照與走動式合成（時點 S&P 500 成份股，2000 至 2024）\n"
                 f"最強單一特徵 {sv['best_feature_label']}：月均 IC {float(sv['best_feature_mean_ic']):+.3f}（t {float(sv['best_feature_t']):+.1f}）；通過預設「真訊號」門檻（IC ≥ 0.03、t ≥ 2、2013 至 2024 有 8 年為正）的特徵：{n_pass} 個；"
                 f"走動式合成樣本外 IC：等權 {float(sv['oos_technical_ew_mean_ic']):+.3f}、Ridge {float(sv['oos_technical_ridge_mean_ic']):+.3f}（網格達標需 0.15）",
                 x=0.01, ha="left", fontsize=11.5, y=0.985)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
