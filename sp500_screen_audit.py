"""Section 22: the two Finviz-style screens (value, high growth) rebuilt point-in-time and audited.

Value screen      P/E < 15 (loose: < 20), P/B < 1 (loose: < 3), ROE > 15%, long-term debt / equity < 0.5, operating margin > 10%
High-growth screen 5-year sales CAGR > 15%, gross margin > 30%; "small" variant adds market cap in the bottom third of the
                  eligible universe that month.  (Finviz's "EPS growth next years" and PEG need analyst estimates, which
                  are not in free data; they are left out and said so.)

Every input is a trailing-twelve-month or balance-sheet value from SEC filings available on the signal date (filing
date <= month end), prices are the Yahoo adjusted close on that date, and the universe is the point-in-time S&P 500
membership, so the screens are what Finviz would have shown on each historical month end if it had existed then.

Audit: (1) monthly Spearman IC of each ingredient, of the "criteria met" counts and of the pass flags against next-month
return, with the section 21 random-feature percentiles; (2) an equal-weight portfolio of all passers, rebalanced at
month end, executed next close, 0.1% cost, cash in T-bills when nothing passes, against SPY, QQQ and 50 random
same-count portfolios, with and without the QQQ 200-day switch.

Usage
  python sp500_screen_audit.py [--random-runs 50] [--seed 42] [--out screen_audit] [--chart docs/screen_audit.png]
  python sp500_screen_audit.py --chart-only
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

import sp500_forward_search as F
import sp500_free_signals as S
import sp500_precursor_fundamental as Fd
import sp500_signal_audit as A
import sp500_top5_monthly as T5
from sp500_precursor_events import load_universe

PAL = A.PAL
GREYS = A.GREYS
DEBT_PATH = "free_signals_xbrl_debt.csv"
COGS_PATH = "free_signals_xbrl_cogs.csv"      # trailing-twelve-month cost of revenue for filers without a GrossProfit tag
LABELS = {
    "ey_ttm": "近四季盈餘殖利率（1 / P/E）", "bm": "帳面市值比（1 / P/B）", "roe": "ROE（近四季淨利 / 股東權益）", "ltde": "長期負債 / 股東權益",
    "om_ttm": "近四季營益率", "gm_ttm": "近四季毛利率", "sales_cagr5": "過去 5 年營收年複合成長率", "mcap": "市值",
    "value_score_loose": "價值篩選達成條件數（寬鬆，0 到 5）", "value_score_strict": "價值篩選達成條件數（嚴格，0 到 5）", "growth_score": "高成長篩選達成條件數（0 到 2）",
    "value_strict": "通過價值篩選（P/E<15、P/B<1）", "value_loose": "通過價值篩選（P/E<20、P/B<3）", "growth": "通過高成長篩選", "growth_small": "通過高成長篩選且市值後三分之一",
}
EXPECTED = {"ey_ttm": 1, "bm": 1, "roe": 1, "ltde": -1, "om_ttm": 1, "gm_ttm": 1, "sales_cagr5": 1, "mcap": -1, "value_score_loose": 1, "value_score_strict": 1, "growth_score": 1,
            "value_strict": 1, "value_loose": 1, "growth": 1, "growth_small": 1}
SCREENS = ["value_strict", "value_loose", "growth", "growth_small"]
SCREEN_ZH = {"value_strict": "價值篩選（嚴格）", "value_loose": "價值篩選（寬鬆）", "growth": "高成長篩選", "growth_small": "高成長篩選 + 小市值"}


def ttm_table(g: pd.DataFrame, col: str) -> pd.DataFrame:
    """(end, filed, val) trailing-four-quarter sums for one ticker from the quarterly table; a sum needs four consecutive
    quarters present and is usable once the last of them is filed."""
    g = g.sort_values("period").drop_duplicates("period")
    per = pd.PeriodIndex(g.period, freq="Q")
    full = pd.period_range(per.min(), per.max(), freq="Q")
    v = pd.Series(g[col].to_numpy(), index=per).reindex(full)
    filed = pd.Series(g.filed.to_numpy(), index=per).reindex(full)
    end = pd.Series(g.period_end.to_numpy(), index=per).reindex(full)
    s = v.rolling(4).sum()
    ok = v.notna().rolling(4).sum() == 4
    fn = filed.astype("datetime64[ns]")
    fmax = pd.to_datetime(fn.astype("int64").where(fn.notna()).rolling(4).max())
    out = pd.DataFrame({"end": end, "filed": fmax, "val": s, "period": full})[ok.to_numpy()]
    return out.dropna(subset=["end", "filed"]).reset_index(drop=True)


def five_year_cagr(ttm: pd.DataFrame) -> pd.DataFrame:
    """Sales CAGR between the TTM ending at period p and the TTM ending 20 quarters earlier; usable once both are filed."""
    t = ttm.set_index("period")
    rows = []
    for p, r in t.iterrows():
        p0 = p - 20
        if p0 in t.index and t.loc[p0, "val"] > 0 and r.val > 0:
            rows.append((r.end, max(r.filed, t.loc[p0, "filed"]), (r.val / t.loc[p0, "val"]) ** 0.2 - 1))
    return pd.DataFrame(rows, columns=["end", "filed", "val"])


def build_features(px: pd.DataFrame, eligible: pd.DataFrame, mes: list[pd.Timestamp], quarterly: str) -> tuple[dict[str, pd.DataFrame], dict]:
    mes_idx = pd.DatetimeIndex(mes)
    pxm = px.reindex(mes_idx)
    xb = pd.read_csv(S.XBRL_PATH, parse_dates=["end", "filed"])
    debt = pd.read_csv(DEBT_PATH, parse_dates=["end", "filed"])
    cogs = pd.read_csv(COGS_PATH, parse_dates=["end", "filed"]) if os.path.exists(COGS_PATH) else pd.DataFrame(columns=["ticker", "end", "filed", "val"])
    q = pd.read_csv(quarterly, parse_dates=["filed", "period_end"])
    shares = S.shares_frame(xb, px, mes)
    mcap = shares * pxm
    cols = ["ey_ttm", "bm", "roe", "ltde", "om_ttm", "gm_ttm", "sales_cagr5", "mcap"]
    fr = {k: pd.DataFrame(np.nan, index=mes_idx, columns=px.columns) for k in cols}
    fr["mcap"] = mcap
    xtab = {t: {m: gg[["end", "filed", "val"]] for m, gg in g.groupby("metric")} for t, g in xb.groupby("ticker")}
    dtab = {t: g[["end", "filed", "val"]] for t, g in debt.groupby("ticker")}
    ctab = {t: g[["end", "filed", "val"]] for t, g in cogs.groupby("ticker")}
    n_debt_missing, n_gm_from_cogs = 0, 0
    for t, g in q.groupby("ticker"):
        if t not in px.columns:
            continue
        rev = ttm_table(g, "revenue")
        eps = ttm_table(g, "eps")
        gp = ttm_table(g, "gross_profit")
        oi = ttm_table(g, "operating_income")
        if not eps.empty:
            e = S._latest_at(eps, mes_idx)
            fr["ey_ttm"][t] = (e.val.to_numpy() / pxm[t].to_numpy())
        if not rev.empty:
            r = S._latest_at(rev, mes_idx)
            if not oi.empty:
                o = S._latest_at(oi, mes_idx)
                fr["om_ttm"][t] = np.where((o.end == r.end) & (r.val > 0), o.val / r.val, np.nan)
            if len(gp) >= 8:
                gpl = S._latest_at(gp, mes_idx)
                fr["gm_ttm"][t] = np.where((gpl.end == r.end) & (r.val > 0), gpl.val / r.val, np.nan)
            elif t in ctab:
                # no GrossProfit tag: gross margin = 1 - cost of revenue / revenue, both trailing twelve months on the same period end
                c = S._latest_at(ctab[t], mes_idx)
                same = (c.end - r.end).abs() <= pd.Timedelta(days=12)
                fr["gm_ttm"][t] = np.where(same & (r.val > 0), 1 - c.val / r.val, np.nan)
                n_gm_from_cogs += 1
            c5 = five_year_cagr(rev)
            if not c5.empty:
                fr["sales_cagr5"][t] = S._latest_at(c5, mes_idx).val.to_numpy()
        xt = xtab.get(t, {})
        if "equity" in xt:
            eq = S._latest_at(xt["equity"], mes_idx).val.to_numpy()
            fr["bm"][t] = np.where(mcap[t].to_numpy() > 0, eq / mcap[t].to_numpy(), np.nan)
            if "ni" in xt:
                ni = S._latest_at(xt["ni"], mes_idx).val.to_numpy()
                fr["roe"][t] = np.where(eq > 0, ni / eq, np.nan)
            if t in dtab:
                d = S._latest_at(dtab[t], mes_idx).val.to_numpy()
                fr["ltde"][t] = np.where(eq > 0, d / eq, np.nan)
            else:
                # no long-term-debt tag in any filing: the filer has no long-term debt (counted and reported)
                fr["ltde"][t] = np.where(eq > 0, 0.0, np.nan)
                n_debt_missing += 1
    # screens (a criterion with a missing input fails, as a blank field does on Finviz)
    pe = 1 / fr["ey_ttm"]
    pb = 1 / fr["bm"]
    pos_pe = (fr["ey_ttm"] > 0)
    pos_pb = (fr["bm"] > 0)
    c_roe = fr["roe"] > 0.15
    c_ltde = fr["ltde"] < 0.5
    c_om = fr["om_ttm"] > 0.10
    c_pe15, c_pe20 = pos_pe & (pe < 15), pos_pe & (pe < 20)
    c_pb1, c_pb3 = pos_pb & (pb < 1), pos_pb & (pb < 3)
    c_sales = fr["sales_cagr5"] > 0.15
    c_gm = fr["gm_ttm"] > 0.30
    small_cut = mcap.where(eligible.reindex(mes_idx)).quantile(1 / 3, axis=1)
    c_small = mcap.le(small_cut, axis=0)
    as_num = lambda b: b.astype(float)
    fr["value_score_strict"] = as_num(c_pe15) + as_num(c_pb1) + as_num(c_roe) + as_num(c_ltde) + as_num(c_om)
    fr["value_score_loose"] = as_num(c_pe20) + as_num(c_pb3) + as_num(c_roe) + as_num(c_ltde) + as_num(c_om)
    fr["growth_score"] = as_num(c_sales) + as_num(c_gm)
    fr["value_strict"] = as_num(c_pe15 & c_pb1 & c_roe & c_ltde & c_om)
    fr["value_loose"] = as_num(c_pe20 & c_pb3 & c_roe & c_ltde & c_om)
    fr["growth"] = as_num(c_sales & c_gm)
    fr["growth_small"] = as_num(c_sales & c_gm & c_small)
    # scores and flags are meaningless where the ingredients are entirely missing: keep them only where P/E or sales data exists
    have = fr["ey_ttm"].notna() | fr["sales_cagr5"].notna()
    for k in ("value_score_strict", "value_score_loose", "value_strict", "value_loose"):
        fr[k] = fr[k].where(fr["ey_ttm"].notna())
    for k in ("growth_score", "growth", "growth_small"):
        fr[k] = fr[k].where(fr["sales_cagr5"].notna())
    info = {"n_debt_missing_tickers": n_debt_missing, "n_gm_from_cogs": n_gm_from_cogs, "coverage": {k: int(fr[k].where(eligible.reindex(mes_idx)).notna().sum(axis=1).loc["2015-01-01":].median()) for k in cols}}
    return fr, info


def portfolio_stats(nav: pd.Series) -> dict:
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    dd = nav / nav.cummax() - 1
    daily = nav.pct_change().dropna()
    return {"cagr": nav.iloc[-1] ** (1 / yrs) - 1, "mdd": dd.min(), "vol": daily.std() * math.sqrt(252), "start": nav.index[0].date(), "end": nav.index[-1].date(), "years": yrs}


def turnover(picks: dict) -> float:
    ds = sorted(picks)
    ch = []
    for a, b in zip(ds[:-1], ds[1:]):
        sa, sb = set(picks[a]), set(picks[b])
        if sa or sb:
            ch.append(1 - len(sa & sb) / max(len(sa), len(sb)))
    return float(np.mean(ch)) if ch else np.nan


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--random-runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cost", type=float, default=0.001)
    p.add_argument("--out", default="screen_audit")
    p.add_argument("--chart", default="docs/screen_audit.png")
    p.add_argument("--chart-only", action="store_true")
    p.add_argument("--quarterly", default="precursor_fund_quarterly.csv")
    p.add_argument("--null", default="free_signals_audit_null.csv")
    args = p.parse_args(argv)
    if args.chart_only:
        chart(pd.read_csv(f"{args.out}_features.csv"), pd.read_csv(f"{args.out}_portfolios.csv"), pd.read_csv(f"{args.out}_equity.csv", index_col=0, parse_dates=True),
              pd.read_csv(f"{args.out}_counts.csv", index_col=0, parse_dates=True), pd.read_csv(f"{args.out}_summary.csv"), args.chart)
        print(f"wrote {args.chart}")
        return 0
    t0 = time.time()
    px, member, eligible, spy = load_universe(args.start, args.end)
    all_mes = T5.month_ends(px.index)
    mes = [d for d in all_mes if d >= pd.Timestamp(args.start) and d <= pd.Timestamp(args.end) and eligible.loc[d].sum() >= 50]
    mes_idx = pd.DatetimeIndex(mes)
    elig = eligible.reindex(mes_idx)
    qqq = F.load_series("qqq_full_cache.csv", px.index)
    irx = F.load_series("irx_cache.csv", px.index)
    cash = (irx / 100 / 252).to_numpy()
    fwd1, fwd3, exec_date, end1 = A.targets(px, eligible, mes)

    # 1. features and screens ------------------------------------------------------------------------
    fr, info = build_features(px, eligible, mes, args.quarterly)
    for k in fr:
        fr[k] = fr[k].reindex(index=mes_idx, columns=px.columns)
    print(f"features built in {time.time() - t0:.0f}s; eligible names with data (median per month, 2015+): {info['coverage']}; tickers without a long-term-debt tag (treated as debt-free): {info['n_debt_missing_tickers']}; gross margin from cost of revenue: {info['n_gm_from_cogs']}")
    counts = pd.DataFrame({k: fr[k].where(elig).eq(1).sum(axis=1) for k in SCREENS}, index=mes_idx)
    counts["eligible"] = elig.sum(axis=1)
    counts["value_data"] = fr["ey_ttm"].where(elig).notna().sum(axis=1)
    counts["growth_data"] = fr["sales_cagr5"].where(elig).notna().sum(axis=1)
    counts.to_csv(f"{args.out}_counts.csv", encoding="utf-8-sig")
    print("passers per month (median 2015+):", counts.loc["2015-01-01":].median().astype(int).to_dict())

    # 2. IC ------------------------------------------------------------------------------------------
    null = pd.read_csv(args.null)
    p99_ic, p99_t = null.mean_ic_2010_2024_adj.quantile(0.99), null.t_2010_2024_adj.quantile(0.99)
    rows = []
    for k in fr:
        x = fr[k].where(elig)
        if k in SCREENS or k.endswith("_score"):
            x = x.where(x.notna())
        ic1, n1 = A.row_spearman(x, fwd1)
        ic3, _ = A.row_spearman(x, fwd3)
        rec = {"feature": k, "label": LABELS[k], "expected_sign": EXPECTED[k], "mean_names": float(n1[n1 > 0].mean()) if (n1 > 0).any() else 0.0,
               "first_month": ic1.dropna().index.min(), **A.feature_stats(ic1, ic3)}
        exp = EXPECTED[k] * ic1
        win = exp.loc["2013-01-01":"2024-12-31"].dropna()
        w = A.ic_summary(win, "e_")
        rec.update({"mean_ic_2013_2024_expected": w["e_mean_ic"], "t_2013_2024_expected": w["e_t"], "years_pos_2013_2024_expected": int((win.groupby(win.index.year).mean() > 0).sum())})
        rec["pass_real_signal_expected"] = bool(len(win) >= 12 and w["e_mean_ic"] >= A.IC_THRESHOLD and w["e_t"] >= A.T_THRESHOLD and rec["years_pos_2013_2024_expected"] >= A.YEARS_THRESHOLD)
        rec["sign_as_expected"] = bool(rec["sign"] == EXPECTED[k])
        rec["above_null_p99_ic"] = bool(rec["mean_ic_signed_abs"] > p99_ic)
        rec["above_null_p99_t"] = bool(rec["t_signed_abs"] > p99_t)
        rows.append(rec)
    features = pd.DataFrame(rows).sort_values("t_signed_abs", ascending=False).reset_index(drop=True)
    features.to_csv(f"{args.out}_features.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(f"\nnull (section 21, 2010-2024 window): sign-adjusted mean IC p99 {p99_ic:.4f}, |t| p99 {p99_t:.2f}")
    print(features[["feature", "expected_sign", "sign", "mean_names", "n_months", "mean_ic", "t", "mean_ic_2013_2024_adj", "t_2013_2024_adj", "years_pos_2013_2024", "mean_ic_2010_2019", "mean_ic_2020_2024",
                    "pass_real_signal", "pass_real_signal_expected", "above_null_p99_t"]].round(4).to_string(index=False))

    # 3. portfolios ----------------------------------------------------------------------------------
    calls = F.switch_calls("qqq_sma200", px, member, qqq, spy, mes)
    rng = np.random.default_rng(args.seed)
    prow, eq = [], {}
    for sc in SCREENS:
        flag = fr[sc].where(elig)
        months = [d for d in mes if flag.loc[d].notna().sum() >= 30]
        first = next((d for d in months if flag.loc[d].eq(1).sum() >= 1), None)
        if first is None:
            continue
        months = [d for d in months if d >= first]
        picks = {d: list(flag.columns[flag.loc[d].eq(1).to_numpy()]) for d in months}
        n_names = pd.Series({d: len(v) for d, v in picks.items()})
        for sw, label in (("none", "無開關"), ("qqq_sma200", "QQQ 200 日線開關")):
            pk = picks if sw == "none" else {d: (v if calls[d] else []) for d, v in picks.items()}
            nav = T5.run_portfolio(px, pk, cost=args.cost, cash_daily=cash)
            st = portfolio_stats(nav)
            eq[f"{sc}_{sw}"] = nav
            # random same-count control
            rnd = []
            for s in range(args.random_runs):
                rp = {}
                for d in months:
                    pool = list(elig.columns[elig.loc[d].to_numpy()])
                    k = min(len(pool), n_names[d])
                    rp[d] = list(rng.choice(pool, size=k, replace=False)) if k else []
                    if sw != "none" and not calls[d]:
                        rp[d] = []
                rnd.append(portfolio_stats(T5.run_portfolio(px, rp, cost=args.cost, cash_daily=cash)))
            rc = pd.DataFrame(rnd)
            yr = nav.dropna().resample("YE").last().pct_change().dropna()
            prow.append({"screen": sc, "switch": sw, "label": f"{SCREEN_ZH[sc]}，{label}", "start": st["start"], "end": st["end"], "years": round(st["years"], 1), "cagr": st["cagr"], "mdd": st["mdd"], "vol": st["vol"],
                         "mean_names": float(n_names.mean()), "median_names": float(n_names.median()), "months_empty": int((n_names == 0).sum()), "turnover": turnover(picks),
                         "random_cagr_median": rc.cagr.median(), "random_cagr_p90": rc.cagr.quantile(0.9), "random_cagr_p10": rc.cagr.quantile(0.1), "random_mdd_median": rc.mdd.median(),
                         "pct_random_beaten": float((rc.cagr < st["cagr"]).mean()),
                         "spy_cagr": portfolio_stats(spy.loc[nav.dropna().index[0]:nav.dropna().index[-1]] / spy.loc[nav.dropna().index[0]])["cagr"],
                         "spy_mdd": portfolio_stats(spy.loc[nav.dropna().index[0]:nav.dropna().index[-1]] / spy.loc[nav.dropna().index[0]])["mdd"],
                         "qqq_cagr": portfolio_stats(qqq.loc[nav.dropna().index[0]:nav.dropna().index[-1]] / qqq.loc[nav.dropna().index[0]])["cagr"],
                         "qqq_mdd": portfolio_stats(qqq.loc[nav.dropna().index[0]:nav.dropna().index[-1]] / qqq.loc[nav.dropna().index[0]])["mdd"],
                         "best_year": yr.max(), "best_year_label": int(yr.idxmax().year), "worst_year": yr.min(), "worst_year_label": int(yr.idxmin().year),
                         "yearly": "|".join(f"{d.year}:{v:+.1%}" for d, v in yr.items())})
            print(f"  {sc:13s} {sw:11s} {st['start']}..{st['end']} CAGR {st['cagr']:+.1%} MDD {st['mdd']:+.0%} names {n_names.mean():.0f} turnover {turnover(picks):.0%} | random median {rc.cagr.median():+.1%} p90 {rc.cagr.quantile(0.9):+.1%} | SPY {prow[-1]['spy_cagr']:+.1%} / {prow[-1]['spy_mdd']:+.0%}", flush=True)
    portfolios = pd.DataFrame(prow)
    portfolios.to_csv(f"{args.out}_portfolios.csv", index=False, encoding="utf-8-sig")
    first_exec = min(v.dropna().index[0] for v in eq.values())
    eq["SPY"] = spy.loc[first_exec:] / spy.loc[first_exec]
    eq["QQQ"] = qqq.loc[first_exec:] / qqq.loc[first_exec]
    equity = pd.DataFrame(eq).loc[first_exec:]
    equity.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")

    # 4. summary -------------------------------------------------------------------------------------
    best = features.iloc[0]
    rows = [("signal_months", len(mes)), ("n_features", len(fr)), ("null_p99_ic_2010_2024", p99_ic), ("null_p99_t_2010_2024", p99_t),
            ("best_feature", best.feature), ("best_feature_mean_ic", best.mean_ic), ("best_feature_t", best.t), ("best_feature_sign", best.sign), ("best_feature_expected_sign", best.expected_sign),
            ("n_pass_real_signal", int(features.pass_real_signal.sum())), ("n_pass_real_signal_expected", int(features.pass_real_signal_expected.sum())), ("n_above_null_p99_t", int(features.above_null_p99_t.sum())),
            ("n_sign_as_expected", int(features.sign_as_expected.sum())), ("tickers_without_ltdebt_tag_treated_debt_free", info["n_debt_missing_tickers"]), ("tickers_gross_margin_from_cost_of_revenue", info["n_gm_from_cogs"])]
    for k, v in info["coverage"].items():
        rows.append((f"coverage_median_names_{k}", v))
    for r in prow:
        key = f"{r['screen']}_{r['switch']}"
        rows += [(f"{key}_cagr", r["cagr"]), (f"{key}_mdd", r["mdd"]), (f"{key}_mean_names", r["mean_names"]), (f"{key}_turnover", r["turnover"]), (f"{key}_random_cagr_median", r["random_cagr_median"]),
                 (f"{key}_random_cagr_p90", r["random_cagr_p90"]), (f"{key}_pct_random_beaten", r["pct_random_beaten"]), (f"{key}_spy_cagr", r["spy_cagr"]), (f"{key}_spy_mdd", r["spy_mdd"]), (f"{key}_start", r["start"])]
    rows += [("cost", args.cost), ("random_runs", args.random_runs), ("seed", args.seed), ("runtime_s", round(time.time() - t0))]
    summary = pd.DataFrame(rows, columns=["key", "value"])
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    print("\n-- portfolios")
    print(portfolios[["label", "start", "end", "cagr", "mdd", "mean_names", "turnover", "random_cagr_median", "random_cagr_p90", "pct_random_beaten", "spy_cagr", "spy_mdd", "qqq_cagr", "best_year", "best_year_label", "worst_year", "worst_year_label"]].round(3).to_string(index=False))
    chart(features, portfolios, equity, counts, summary, args.chart)
    print(f"\nwrote {args.out}_features.csv, _counts.csv, _portfolios.csv, _equity.csv, _summary.csv, {args.chart} ({time.time() - t0:.0f}s)")
    return 0


# ----------------------------------------------------------------------------- chart
def chart(features: pd.DataFrame, portfolios: pd.DataFrame, equity: pd.DataFrame, counts: pd.DataFrame, summary: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    sv = dict(zip(summary.key, summary.value))
    p99 = float(sv["null_p99_ic_2010_2024"])
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.36, wspace=0.2, left=0.2, right=0.985, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, :])
    f = features.copy()
    f["kind"] = np.where(f.feature.isin(SCREENS), "flag", np.where(f.feature.str.endswith("_score"), "score", "ingredient"))
    f = f.sort_values("mean_ic_2013_2024_adj").reset_index(drop=True)
    col = {"ingredient": PAL[0], "score": PAL[2], "flag": PAL[1]}
    y = np.arange(len(f))
    ax1.axvspan(0, p99, color="#e4e4e4", zorder=0)
    ax1.axvline(A.IC_THRESHOLD, color=GREYS[0], ls="--", lw=1)
    ax1.axvline(0, color=GREYS[2], lw=0.8)
    ax1.barh(y, f.mean_ic_2013_2024_adj, color=[col[k] for k in f.kind], height=0.62, zorder=2)
    for yi, r in f.iterrows():
        note = "" if r.sign == r.expected_sign else "（方向與篩選假設相反）"
        ax1.annotate(f"{r.mean_ic_2013_2024_adj:+.3f}（t {r.t_2013_2024_adj:+.1f}）{note}", (max(r.mean_ic_2013_2024_adj, 0), yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.2, color="#333333", zorder=3)
    ax1.set_yticks(y); ax1.set_yticklabels(f.label, fontsize=8.8)
    ax1.set_xlim(min(-0.005, f.mean_ic_2013_2024_adj.min() - 0.003), max(0.045, f.mean_ic_2013_2024_adj.max() + 0.022))
    ax1.set_xlabel("月度 Spearman IC 平均（對下月報酬，2013 至 2024 年，已按全樣本方向調正）", fontsize=9.5)
    ax1.set_title("篩選條件的成分、達成條件數與通過旗標的 IC：灰帶 = 第 21 節 200 個亂數特徵的 99 百分位，虛線 = 門檻 0.03", loc="left", fontsize=11.5, fontweight="bold")
    ax1.legend(handles=[Patch(color=col["ingredient"], label="成分（連續值）"), Patch(color=col["score"], label="達成條件數"), Patch(color=col["flag"], label="通過旗標"), Patch(color="#e4e4e4", label=f"亂數 99% 帶 {p99:.3f}"),
                        Line2D([], [], color=GREYS[0], ls="--", lw=1, label="門檻 0.03")], loc="lower right", fontsize=8.5, frameon=False)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0])
    cols = {"value_strict_none": (SCREEN_ZH["value_strict"], PAL[0]), "value_loose_none": (SCREEN_ZH["value_loose"], PAL[4]), "growth_none": (SCREEN_ZH["growth"], PAL[2]), "growth_small_none": (SCREEN_ZH["growth_small"], PAL[3]),
            "SPY": ("SPY", GREYS[0]), "QQQ": ("QQQ", GREYS[2])}
    pst = portfolios.set_index(portfolios.screen + "_" + portfolios.switch)
    for k, (lab, c) in cols.items():
        if k not in equity.columns:
            continue
        s = equity[k].dropna()
        s = s / s.iloc[0]
        extra = f"：{float(pst.loc[k, 'cagr']):+.1%} / {float(pst.loc[k, 'mdd']):+.0%}" if k in pst.index else f"：{(s.iloc[-1] ** (365.25 / (s.index[-1] - s.index[0]).days) - 1):+.1%} / {(s / s.cummax() - 1).min():+.0%}"
        ax2.plot(s.index, s, lw=1.5 if not k.startswith(("SPY", "QQQ")) else 1.2, color=c, ls="-" if not k.startswith(("SPY", "QQQ")) else "--", label=lab + extra)
    ax2.set_yscale("log")
    ax2.set_title("通過者等權組合（每月底換股、次日收盤成交、成本 0.1%）對 SPY、QQQ", loc="left", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("淨值（起點 = 1，對數）", fontsize=9.5)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    for k, (lab, c) in {"value_strict": (SCREEN_ZH["value_strict"], PAL[0]), "value_loose": (SCREEN_ZH["value_loose"], PAL[4]), "growth": (SCREEN_ZH["growth"], PAL[2]), "growth_small": (SCREEN_ZH["growth_small"], PAL[3])}.items():
        s = counts[k].loc["2010-01-01":]
        ax3.plot(s.index, s, lw=1.5, color=c, label=f"{lab}：中位數 {int(s.loc['2015-01-01':].median())} 檔")
    ax3.plot(counts["value_data"].loc["2010-01-01":].index, counts["value_data"].loc["2010-01-01":], lw=1, color=GREYS[2], ls=":", label="有 P/E 資料的可交易名單")
    ax3.set_title("每月通過篩選的檔數（時點成份股）", loc="left", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("檔數", fontsize=9.5)
    ax3.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    fig.suptitle("第 22 節：Finviz 價值篩選與高成長篩選，按時點資料重現 2010 至 2024 年", fontsize=14, fontweight="bold", x=0.2, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
