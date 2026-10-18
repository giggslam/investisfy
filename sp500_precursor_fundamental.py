#!/usr/bin/env python3
"""
Fundamental precursors of a momentum surge (section 17b).

Question. Section 15 showed that "hold the 5 strongest 63-day-momentum S&P 500 stocks each month" has
no edge. When a stock is about to ENTER that top-5 (a momentum surge), is there anything in its
fundamentals, visible beforehand, that (a) distinguishes next month's entrants from the rest of the
universe and (b) separates entrants whose surge continues (stock beats SPY over the next 63 trading
days, label `cont63` from precursor_events.csv) from those that fade? And does a score built from
those precursors beat the plain momentum ranking in the same monthly top-N engine, out of sample?

Data source. SEC EDGAR XBRL "company concept" API (data.sec.gov/api/xbrl/companyconcept), one JSON
per (company, us-gaap tag): every reported value of Revenues (3 tag variants merged),
EarningsPerShareDiluted (Basic as fallback), NetIncomeLoss, GrossProfit and OperatingIncomeLoss, with
the fiscal period (start/end), the form and the `filed` date. Tickers are mapped to CIKs with
www.sec.gov/files/company_tickers.json (current tickers only, so delisted names do not map; coverage is
reported). Every response is cached under edgar_cache/ (gitignored); requests use a contact-style
User-Agent and are throttled to <= 8 per second.

Point-in-time rule. A quarter's numbers become usable only from the `filed` date of the FIRST filing
that reported that (start, end) period (the original 10-Q or 10-K, not a later restated comparative).
Quarterly periods are those with a 75-100 day duration; Q4 is derived as the 10-K annual value minus
the three 10-Q quarters of the same fiscal year when no explicit Q4 period exists, and is dated by the
10-K's `filed`. Periods are bucketed into calendar quarters by the quarter that contains (end - 15 days),
so 52/53-week and off-calendar fiscal years line up; YoY compares with the bucket four quarters earlier.
Features at a signal date t use the latest quarter whose `filed` <= t (asserted for every row); the
"filing reaction" is the stock-minus-SPY return from the close before `filed` to two trading days
after, truncated at t. Labels (is_event, cont63, ex63) come from precursor_events.csv.

Coverage limits. XBRL company facts start in 2009 (large accelerated filers) and 2011 for the rest, so
the study runs 2010-2024 and the model/portfolio tests 2013-2024. Only tickers in the current SEC
ticker file map to a CIK; the point-in-time universe's delisted names are therefore dropped (the
coverage numbers are printed and saved). Tag gaps (companies that use non-standard revenue or
margin tags, banks without GrossProfit) appear as missing features, which the model treats as the
median rank (0.5). Fiscal-year quirks (changes of fiscal year end, 53-week years) can leave a bucket
empty, which breaks the YoY chain for that quarter only.

Features (per signal date, from the latest filed quarter): revenue YoY, EPS YoY (sign-guarded,
capped +-3), net-income YoY, gross and operating margin level and YoY change (points), revenue-growth
acceleration (this YoY minus previous quarter's), EPS acceleration, consecutive quarters of positive
revenue YoY, trailing-4-quarter earnings yield, days since the filing, the market reaction to the last
two filings.

Analyses. (1) Single-feature AUC for entry and for continuation, monthly cross-sectional quintile lifts.
(2) Anchored expanding-window logistic regression (numpy, L2, Newton steps) on within-month rank-
normalised features: T1 = is_event, T2 = is_event & cont63 (events + sampled controls), T3 = ridge
regression of rank-normalised ex63 on every covered universe row; trained through the previous year
(label embargo: T2/T3 training rows end three months earlier so all 63-day labels are known),
tested on each year 2013-2024. (3) Walk-forward monthly top-N portfolios scored with those models,
compared with the 63-day-momentum top-5 on the SAME covered universe, SPY, QQQ, random portfolios and
two one-feature rules, each with and without the QQQ 200-day month-end filter.

Outputs: precursor_fund_quarterly.csv, precursor_fund_features.csv, precursor_fund_lift.csv,
precursor_fund_model.csv, precursor_fund_equity.csv, precursor_fund_summary.csv,
docs/precursors_fundamental.png

Usage: python sp500_precursor_fundamental.py [--n 5] [--seed 42] [--start-test 2013] [--out precursor_fund]
                                             [--refresh] [--chart-only] [--max-tickers K] [--fetch-only]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import sp500_beat_qqq as B
import sp500_top5_monthly as T5
from sp500_precursor_events import load_universe

CACHE = "edgar_cache"
HEADERS = {"User-Agent": "Mozilla/5.0 investisfy-research admin@investisfy.local", "Accept-Encoding": "gzip, deflate"}
MIN_GAP = 0.13  # seconds between EDGAR requests (<= 8/s)
TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
}
# extra tags, requested only when the primary tags yield fewer than MIN_QUARTERS quarterly periods
FALLBACK_TAGS = {
    "revenue": ["SalesRevenueGoodsNet", "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense"],
    "net_income": ["ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
}
MIN_QUARTERS = 20
SPLIT_RATIOS = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 20, 50, 1 / 1.5, 0.5, 1 / 3, 0.25, 0.2, 0.1, 0.05]
UNITS = {"eps": "USD/shares"}
METRICS = list(TAGS)
# Yahoo symbol -> current SEC ticker for names that were renamed but kept their CIK (only used when unmapped)
ALIASES = {"FB": "META", "ANTM": "ELV", "ABC": "COR", "BLL": "BALL", "FISV": "FI", "WLTW": "WTW", "ADS": "BFH", "RE": "EG",
           "PKI": "RVTY", "FLT": "CPAY", "FBHS": "FBIN", "PEAK": "DOC", "DISCA": "WBD", "VIAC": "PARA"}
FEATURES = ["rev_yoy", "eps_yoy", "ni_yoy", "gm", "om", "gm_chg", "om_chg", "rev_accel", "eps_accel", "rev_streak", "ey_ttm",
            "days_since_filing", "react_last", "react_prev"]
FEATURE_ZH = {"rev_yoy": "營收年增率", "eps_yoy": "EPS年增率", "ni_yoy": "淨利年增率", "gm": "毛利率", "om": "營益率", "gm_chg": "毛利率年變化",
              "om_chg": "營益率年變化", "rev_accel": "營收加速度", "eps_accel": "EPS加速度", "rev_streak": "連續營收正成長季數", "ey_ttm": "近四季盈餘殖利率",
              "days_since_filing": "距上次申報天數", "react_last": "上次申報市場反應", "react_prev": "前次申報市場反應"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7f7f7f", "#b0b0b0"]

EDGAR_STATS = {"requests": 0, "cached": 0, "404": 0, "403": 0, "other_errors": 0}
_last_call = [0.0]


# ----------------------------------------------------------------------------------------------------- EDGAR
def _cache_path(url: str) -> str:
    return os.path.join(CACHE, url.split("sec.gov/", 1)[1].replace("/", "__"))


def edgar_get(url: str, refresh: bool = False) -> str:
    """Return the body as text ('' for a 404), served from edgar_cache/ when present."""
    os.makedirs(CACHE, exist_ok=True)
    fn = _cache_path(url)
    if not refresh and os.path.exists(fn):
        EDGAR_STATS["cached"] += 1
        with open(fn, encoding="utf-8") as f:
            return f.read()
    for attempt in range(4):
        wait = MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
        except requests.RequestException:
            _last_call[0] = time.time()
            EDGAR_STATS["other_errors"] += 1
            time.sleep(2 + 3 * attempt)
            continue
        _last_call[0] = time.time()
        EDGAR_STATS["requests"] += 1
        if r.status_code == 200:
            text = r.text
        elif r.status_code == 404:
            EDGAR_STATS["404"] += 1
            text = ""
        elif r.status_code in (403, 429, 503):
            EDGAR_STATS["403"] += 1
            time.sleep(5 + 10 * attempt)
            continue
        else:
            EDGAR_STATS["other_errors"] += 1
            time.sleep(2)
            continue
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    print(f"  ! giving up on {url}", file=sys.stderr)
    return ""


def ticker_cik_map(tickers: list[str], refresh: bool = False) -> dict[str, int]:
    raw = edgar_get("https://www.sec.gov/files/company_tickers.json", refresh)
    d = json.loads(raw)
    sec = {v["ticker"].upper(): int(v["cik_str"]) for v in d.values()}
    out = {}
    for t in tickers:
        key = t.upper()
        if key in sec:
            out[t] = sec[key]
        elif key.replace("-", ".") in sec:
            out[t] = sec[key.replace("-", ".")]
        elif ALIASES.get(key) and ALIASES[key] in sec:
            out[t] = sec[ALIASES[key]]
    return out


def concept_entries(cik: int, tag: str, unit: str, refresh: bool = False) -> list[dict]:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"
    raw = edgar_get(url, refresh)
    if not raw:
        return []
    try:
        return json.loads(raw).get("units", {}).get(unit, [])
    except json.JSONDecodeError:
        return []


def periods_from_entries(entries: list[dict], dedup: bool = True) -> pd.DataFrame:
    """(start, end, kind, filed, val) per reported duration period, keeping the EARLIEST filing of each period
    (dedup=False keeps every re-report, used to detect stock splits in per-share data)."""
    rows = []
    for e in entries:
        form = str(e.get("form", ""))
        if not (form.startswith("10-Q") or form.startswith("10-K")) or "start" not in e or e.get("val") is None:
            continue
        s, en = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        dur = (en - s).days
        if 75 <= dur <= 100:
            kind = "Q"
        elif 340 <= dur <= 380:
            kind = "A"
        else:
            continue
        rows.append((s, en, kind, pd.Timestamp(e["filed"]), float(e["val"])))
    if not rows:
        return pd.DataFrame(columns=["start", "end", "kind", "filed", "val"])
    df = pd.DataFrame(rows, columns=["start", "end", "kind", "filed", "val"]).sort_values("filed")
    if not dedup:
        return df.reset_index(drop=True)
    return df.groupby(["start", "end"], as_index=False).first()


def split_events(per_all: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    """Stock splits inferred from per-share data: when a later filing re-reports a period at 1/k of the first value
    (k a usual split ratio) a split of k became visible at that later filing date. Returns [(discovery date, k)]."""
    found = []
    for _, g in per_all.groupby(["start", "end"]):
        g = g.sort_values("filed")
        vals, filed = g["val"].to_numpy(), g["filed"].to_numpy()
        for i in range(1, len(vals)):
            v0, v1 = vals[i - 1], vals[i]
            if v0 == 0 or v1 == 0 or v0 * v1 < 0:
                continue
            r = v0 / v1
            for k in SPLIT_RATIOS:
                if abs(r / k - 1) < 0.01:
                    found.append((pd.Timestamp(filed[i]), k))
    events: list[tuple[pd.Timestamp, float]] = []
    for d, k in sorted(found):
        if any(abs(k2 - k) < 1e-9 and abs((d - d2).days) <= 120 for d2, k2 in events):
            continue
        events.append((d, k))
    return events


def apply_splits(qm: pd.DataFrame, events: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame:
    """Put first-reported per-share values on the latest share basis (divide by every split disclosed after the filing)."""
    if qm.empty or not events:
        return qm
    qm = qm.copy()
    factor = np.ones(len(qm))
    for d, k in events:
        factor[(qm["filed"] < d).to_numpy()] *= k
    qm["val"] = qm["val"] / factor
    return qm


def quarter_bucket(end: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex((pd.DatetimeIndex(end) - pd.Timedelta(days=15)), freq="Q")


def quarterly_metric(per: pd.DataFrame) -> pd.DataFrame:
    """Quarterly values (explicit quarters + Q4 derived from 10-K minus three 10-Qs), one row per calendar-quarter bucket."""
    if per.empty:
        return pd.DataFrame(columns=["bucket", "end", "filed", "val"])
    q = per[per.kind == "Q"].copy()
    a = per[per.kind == "A"].copy()
    q["bucket"] = quarter_bucket(q["end"])
    derived = []
    for _, r in a.iterrows():
        inside = q[(q["start"] >= r["start"] - pd.Timedelta(days=7)) & (q["end"] <= r["end"] - pd.Timedelta(days=60))]
        inside = inside.sort_values("end").drop_duplicates("bucket", keep="first")
        if len(inside) != 3:
            continue
        derived.append({"start": inside["end"].max() + pd.Timedelta(days=1), "end": r["end"], "kind": "Q4d", "filed": r["filed"],
                        "val": r["val"] - inside["val"].sum(), "bucket": quarter_bucket(pd.Series([r["end"]]))[0]})
    allq = pd.concat([q, pd.DataFrame(derived)], ignore_index=True) if derived else q
    allq = allq.sort_values(["filed", "end"]).drop_duplicates("bucket", keep="first")
    return allq[["bucket", "end", "filed", "val"]].sort_values("bucket").reset_index(drop=True)


def fetch_company(ticker: str, cik: int, refresh: bool = False) -> pd.DataFrame:
    """Long quarterly table for one company: ticker, period, period_end, filed, <metrics>."""
    tables = {}
    for metric, tags in TAGS.items():
        unit = UNITS.get(metric, "USD")
        raw = [periods_from_entries(concept_entries(cik, tag, unit, refresh), dedup=False) for tag in tags]
        if sum((p.kind == "Q").sum() for p in raw if not p.empty) < MIN_QUARTERS:
            raw += [periods_from_entries(concept_entries(cik, tag, unit, refresh), dedup=False) for tag in FALLBACK_TAGS.get(metric, [])]
        raw = [p for p in raw if not p.empty]
        if not raw:
            continue
        per_all = pd.concat(raw, ignore_index=True).sort_values("filed")
        per = per_all.groupby(["start", "end"], as_index=False).first()
        qm = quarterly_metric(per)
        if metric == "eps" and not qm.empty:
            qm = apply_splits(qm, split_events(per_all))
        if not qm.empty:
            tables[metric] = qm.set_index("bucket")
    if "revenue" not in tables and "net_income" not in tables:
        return pd.DataFrame()
    buckets = sorted(set().union(*[set(t.index) for t in tables.values()]))
    out = pd.DataFrame(index=pd.PeriodIndex(buckets, freq="Q"))
    for metric, t in tables.items():
        out[metric] = t["val"].reindex(out.index)
        out[f"{metric}_filed"] = t["filed"].reindex(out.index)
        out[f"{metric}_end"] = t["end"].reindex(out.index)
    # availability date of the row: anchor on revenue / net income; metrics first reported much later are dropped (stay NaN)
    anchor = pd.concat([out.get(f"{m}_filed", pd.Series(pd.NaT, index=out.index)) for m in ("revenue", "net_income")], axis=1).min(axis=1)
    out = out[anchor.notna()].copy()
    anchor = anchor[out.index]
    filed = anchor.copy()
    ends = pd.Series(pd.NaT, index=out.index)
    for metric in METRICS:
        if metric not in tables:
            out[metric] = np.nan
            continue
        late = out[f"{metric}_filed"] > anchor + pd.Timedelta(days=120)
        out.loc[late, metric] = np.nan
        ok = out[f"{metric}_filed"].notna() & ~late
        filed = pd.concat([filed, out[f"{metric}_filed"].where(ok)], axis=1).max(axis=1)
        ends = ends.fillna(out[f"{metric}_end"].where(ok))
    res = pd.DataFrame({"ticker": ticker, "period": [str(p) for p in out.index], "period_end": ends.to_numpy(), "filed": filed.to_numpy()})
    for metric in METRICS:
        res[metric] = out[metric].to_numpy()
    return res.reset_index(drop=True)


def build_quarterly(tickers_first: list[str], tickers_rest: list[str], cik: dict[str, int], refresh: bool, path: str) -> pd.DataFrame:
    frames = []
    order = [t for t in tickers_first if t in cik] + [t for t in tickers_rest if t in cik]
    t0 = time.time()
    for i, t in enumerate(order, 1):
        df = fetch_company(t, cik[t], refresh)
        if not df.empty:
            frames.append(df)
        if i % 25 == 0 or i == len(order):
            print(f"  fetched {i}/{len(order)} companies, {EDGAR_STATS['requests']} live requests, {EDGAR_STATS['cached']} cached, "
                  f"{EDGAR_STATS['404']} missing tags, {EDGAR_STATS['403']} 403/429, {time.time() - t0:.0f}s", flush=True)
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    qdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    qdf.to_csv(path, index=False, encoding="utf-8-sig")
    return qdf


# -------------------------------------------------------------------------------------------------- features
def quarter_features(q: pd.DataFrame) -> pd.DataFrame:
    """Per-quarter derived numbers for one ticker, on a complete consecutive quarterly index (missing quarters = NaN)."""
    q = q.copy()
    q["bucket"] = pd.PeriodIndex(q["period"], freq="Q")
    q = q.set_index("bucket").sort_index()
    full = pd.period_range(q.index.min(), q.index.max(), freq="Q")
    q = q.reindex(full)
    rev, eps, ni, gp, oi = (q[c].astype(float) for c in METRICS)
    f = pd.DataFrame(index=full)
    f["filed"] = pd.to_datetime(q["filed"])
    f["period_end"] = pd.to_datetime(q["period_end"])
    f["revenue"] = rev
    lag = rev.shift(4)
    f["rev_yoy"] = np.where(lag > 0, rev / lag - 1, np.nan)
    f["eps_yoy"] = ((eps - eps.shift(4)) / eps.shift(4).abs()).clip(-3, 3)
    f["ni_yoy"] = ((ni - ni.shift(4)) / ni.shift(4).abs()).clip(-3, 3)
    gm = np.where(rev > 0, gp / rev, np.nan)
    om = np.where(rev > 0, oi / rev, np.nan)
    f["gm"] = gm
    f["om"] = om
    f["gm_chg"] = (f["gm"] - f["gm"].shift(4)) * 100
    f["om_chg"] = (f["om"] - f["om"].shift(4)) * 100
    f["rev_accel"] = f["rev_yoy"] - f["rev_yoy"].shift(1)
    f["eps_accel"] = f["eps_yoy"] - f["eps_yoy"].shift(1)
    pos = (f["rev_yoy"] > 0).to_numpy()
    known = f["rev_yoy"].notna().to_numpy()
    streak = np.zeros(len(f))
    for i in range(len(f)):
        if not known[i]:
            streak[i] = np.nan
        elif pos[i]:
            streak[i] = (streak[i - 1] if i > 0 and not np.isnan(streak[i - 1]) else 0) + 1
    f["rev_streak"] = streak
    f["eps_ttm"] = eps.rolling(4).sum()
    f = f[f["filed"].notna()]
    return f


def build_features(qdf: pd.DataFrame, px: pd.DataFrame, eligible: pd.DataFrame, spy: pd.Series, mes: list[pd.Timestamp]) -> pd.DataFrame:
    """Point-in-time features for every (month end, covered ticker eligible at that month end)."""
    idx = px.index
    pxv = px.to_numpy(dtype=float)
    spyv = spy.to_numpy(dtype=float)
    col = {c: i for i, c in enumerate(px.columns)}
    me_pos = np.array([idx.get_loc(d) for d in mes])
    elig = eligible.to_numpy()
    rows = []
    for ticker, q in qdf.groupby("ticker", sort=True):
        if ticker not in col:
            continue
        f = quarter_features(q)
        if f.empty:
            continue
        f = f.reset_index().rename(columns={"index": "bucket"})
        f["ord"] = [p.ordinal for p in f["bucket"]]
        f = f.sort_values(["filed", "ord"]).reset_index(drop=True)
        filed = f["filed"].to_numpy()
        ords = f["ord"].to_numpy()
        # row with the most recent quarter among everything filed so far (a late comparative must not replace a newer quarter)
        best = np.zeros(len(f), dtype=int)
        for i in range(1, len(f)):
            best[i] = i if ords[i] >= ords[best[i - 1]] else best[i - 1]
        row_of_ord = {o: i for i, o in enumerate(ords)}
        fpos = idx.searchsorted(filed)  # first trading day on/after the filing date
        base_ok = fpos >= 1
        c = col[ticker]
        vals = {k: f[k].to_numpy(dtype=float) for k in ["rev_yoy", "eps_yoy", "ni_yoy", "gm", "om", "gm_chg", "om_chg", "rev_accel", "eps_accel", "rev_streak", "eps_ttm"]}

        def react(k: int, tpos: int) -> float:
            if not base_ok[k]:
                return np.nan
            b = fpos[k] - 1
            e = min(fpos[k] + 2, tpos)
            if e <= b:
                return np.nan
            p0, p1, s0, s1 = pxv[b, c], pxv[e, c], spyv[b], spyv[e]
            if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and np.isfinite(s0) and np.isfinite(s1)):
                return np.nan
            return (p1 / p0 - 1) - (s1 / s0 - 1)

        ks = np.searchsorted(filed, np.array(mes, dtype="datetime64[ns]"), side="right") - 1
        for j, d in enumerate(mes):
            tpos = me_pos[j]
            if ks[j] < 0 or not elig[tpos, col[ticker]]:
                continue
            k = best[ks[j]]
            assert filed[k] <= d.to_datetime64(), (ticker, d, filed[k])
            kp = row_of_ord.get(ords[k] - 1, -1)
            if kp >= 0 and filed[kp] > d.to_datetime64():
                kp = -1
            price = pxv[tpos, c]
            row = {"signal_date": d, "ticker": ticker, "period": str(f["bucket"].iloc[k]), "filed": pd.Timestamp(filed[k]), "period_end": f["period_end"].iloc[k]}
            for key, arr in vals.items():
                if key != "eps_ttm":
                    row[key] = arr[k]
            row["ey_ttm"] = vals["eps_ttm"][k] / price if np.isfinite(price) and price > 0 else np.nan
            row["days_since_filing"] = (d - pd.Timestamp(filed[k])).days
            row["react_last"] = react(k, tpos)
            row["react_prev"] = react(kp, tpos) if kp >= 0 else np.nan
            rows.append(row)
    feats = pd.DataFrame(rows)
    return feats


# ------------------------------------------------------------------------------------------------------ stats
def auc(score: pd.Series, y: pd.Series) -> float:
    m = score.notna() & y.notna()
    s, t = score[m], y[m].astype(int)
    n1, n0 = int(t.sum()), int((1 - t).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = s.rank(method="average")
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def month_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Within-month percentile rank in (0,1), centred so the median is 0.5; missing -> 0.5."""
    g = df.groupby("signal_date")
    out = pd.DataFrame(index=df.index)
    for c in cols:
        r = g[c].rank(method="average", pct=True)
        n = g[c].transform("count")
        out[c] = (r - 0.5 / n).fillna(0.5)
    return out


def lift_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURES:
        d = ev[["signal_date", "is_event", "cont63", "complete63", c]].dropna(subset=[c])
        if d.empty:
            continue
        q = (d.groupby("signal_date")[c].rank(pct=True, method="first") * 5 - 1e-9).astype(int).clip(0, 4) + 1
        d = d.assign(q=q)
        base = d.is_event.mean()
        pq = d.groupby("q").is_event.mean() / base
        e = d[(d.is_event == 1) & (d.complete63 == 1)]
        cq = e.groupby("q").cont63.mean()
        ctrl = d[d.is_event == 0]
        rows.append({"feature": c, "n": len(d), "n_events": int(d.is_event.sum()),
                     "auc_entry": auc(d[c], d.is_event), "lift_q5": pq.get(5, np.nan), "lift_q1": pq.get(1, np.nan),
                     "auc_cont63": auc(e[c], e.cont63), "cont_q5": cq.get(5, np.nan), "cont_q1": cq.get(1, np.nan),
                     "mean_event": e[c].mean() if len(e) else d[d.is_event == 1][c].mean(), "mean_control": ctrl[c].mean(),
                     "median_event": d[d.is_event == 1][c].median(), "median_control": ctrl[c].median(),
                     "median_cont1": e[e.cont63 == 1][c].median(), "median_cont0": e[e.cont63 == 0][c].median()})
    lt = pd.DataFrame(rows)
    lt["abs_auc_entry"] = (lt.auc_entry - 0.5).abs()
    lt["abs_auc_cont63"] = (lt.auc_cont63 - 0.5).abs()
    return lt.sort_values("abs_auc_entry", ascending=False).reset_index(drop=True)


def logit_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-penalised logistic regression (intercept unpenalised), Newton steps with step halving."""
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    R = np.eye(p + 1) * l2
    R[0, 0] = 0.0
    w = np.zeros(p + 1)
    w[0] = math.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))

    def loss(w):
        z = Xb @ w
        return float(np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)) + 0.5 * w @ R @ w)

    cur = loss(w)
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(Xb @ w)))
        g = Xb.T @ (pr - y) + R @ w
        H = Xb.T @ (Xb * (pr * (1 - pr))[:, None]) + R + 1e-8 * np.eye(p + 1)
        step = np.linalg.solve(H, g)
        for _ in range(20):
            w2 = w - step
            new = loss(w2)
            if new <= cur + 1e-9:
                break
            step *= 0.5
        if abs(cur - new) < 1e-8:
            w = w2
            break
        w, cur = w2, new
    return w


def ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    R = np.eye(p + 1) * l2
    R[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + R, Xb.T @ y)


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return w[0] + X @ w[1:]


def spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return np.nan
    return float(a[m].rank().corr(b[m].rank()))


# ------------------------------------------------------------------------------------------------- portfolio
def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1
    yearly = nav.groupby(nav.index.year).last() / nav.groupby(nav.index.year).first() - 1
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    return {"final_x": round(nav.iloc[-1] / nav.iloc[0], 2), "cagr_pct": round(c * 100, 2), "ann_vol_pct": round(r.std() * math.sqrt(252) * 100, 1),
            "sharpe": round(r.mean() / r.std() * math.sqrt(252), 2) if r.std() > 0 else np.nan, "max_dd_pct": round(dd.min() * 100, 1),
            "worst_year_pct": round(yearly.min() * 100, 1), "cagr_2013_2019_pct": round(T5.period_cagr(nav, "2013-01-01", "2019-12-31") * 100, 2),
            "cagr_2020_2024_pct": round(T5.period_cagr(nav, "2020-01-01", "2024-12-31") * 100, 2)}


# ---------------------------------------------------------------------------------------------------- chart
def chart(lift: pd.DataFrame, model: pd.DataFrame, eq: pd.DataFrame, n: int, path: str) -> None:
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), dpi=150, gridspec_kw={"width_ratios": [1.05, 0.9, 1.25]})
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # (a) feature bars
    ax = axes[0]
    lt = lift.sort_values("abs_auc_entry", ascending=True)
    y = np.arange(len(lt))
    h = 0.38
    ax.barh(y + h / 2, (lt.auc_entry - 0.5) * 100, height=h, color=PALETTE[0], label="下月進入前五 (進場)")
    ax.barh(y - h / 2, (lt.auc_cont63 - 0.5) * 100, height=h, color=PALETTE[1], label="進場後 63 日續強 (cont63)")
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_ZH.get(f, f) for f in lt.feature], fontsize=9)
    ax.axvline(0, color="#7f7f7f", lw=0.8)
    ax.set_xlabel("單一特徵 AUC - 0.5（百分點；正 = 數值越高機率越高）")
    ax.set_title("(a) 基本面前兆：單一特徵區分力 2010-2024", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    # (b) OOS by year
    ax = axes[1]
    yrs = model["year"].to_numpy()
    w = 0.27
    ax.bar(yrs - w, (model["auc_T1_entry"] - 0.5) * 100, width=w, color=PALETTE[0], label="T1 進場 AUC - 0.5")
    ax.bar(yrs, (model["auc_T2_entry_cont"] - 0.5) * 100, width=w, color=PALETTE[1], label="T2 進場且續強 AUC - 0.5")
    ax.bar(yrs + w, model["ic_T3_monthly_mean"] * 100, width=w, color=PALETTE[2], label="T3 ex63 月均 Spearman IC")
    ax.axhline(0, color="#7f7f7f", lw=0.8)
    ax.set_xticks(yrs)
    ax.set_xticklabels([str(int(v)) for v in yrs], rotation=45, fontsize=8)
    ax.set_ylabel("百分點")
    ax.set_title("(b) 樣本外模型表現（前一年止訓練，逐年測試）", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    # (c) NAV
    ax = axes[2]
    series = [(f"fund_T3_top{n}", f"基本面分數 前{n}", PALETTE[0]), (f"fund_T3_top{n}_r200", f"基本面分數 前{n} + QQQ 200日濾網", PALETTE[1]),
              (f"mom63_top{n}", f"63日動能 前{n}（同一覆蓋股池）", PALETTE[3]), ("SPY", "SPY", PALETTE[6]), ("QQQ", "QQQ", PALETTE[2])]
    ends = []
    for col, label, color in series:
        if col not in eq.columns:
            continue
        s = eq[col].dropna()
        ax.plot(s.index, s.values, color=color, lw=1.6)
        ends.append((s.index[-1], s.iloc[-1], label, color))
    ax.set_yscale("log")
    ymax_v = max(eq[c].dropna().max() for c, _, _ in series if c in eq.columns)
    ymin_v = min(eq[c].dropna().min() for c, _, _ in series if c in eq.columns)
    ticks = [t for t in (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30) if ymin_v * 0.9 <= t <= ymax_v * 1.3]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title("(c) 走勢驗證 2013-2024：淨值（對數座標，起點 1）", loc="left", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6, which="both")
    # direct labels, spread vertically in log space
    ends.sort(key=lambda e: e[1])
    ymin, ymax = ax.get_ylim()
    logs = [math.log(e[1]) for e in ends]
    gap = 0.075 * (math.log(ymax) - math.log(ymin))
    for i in range(1, len(logs)):
        if logs[i] - logs[i - 1] < gap:
            logs[i] = logs[i - 1] + gap
    x_end = ends[-1][0]
    for (d, v, label, color), ly in zip(ends, logs):
        ax.annotate(f"{label}  {v:.2f}x", xy=(d, v), xytext=(x_end + pd.Timedelta(days=60), math.exp(ly)), fontsize=8.5, color=color, va="center",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6, "alpha": 0.6})
    ax.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=1100))
    fig.suptitle("S&P 500 動能突進的基本面前兆（SEC EDGAR XBRL，依申報日 point-in-time）", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-test", type=int, default=2013)
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out", default="precursor_fund")
    p.add_argument("--events", default="precursor_events.csv")
    p.add_argument("--refresh", action="store_true", help="re-download EDGAR data instead of using edgar_cache/")
    p.add_argument("--chart-only", action="store_true")
    p.add_argument("--fetch-only", action="store_true", help="stop after writing the quarterly table")
    p.add_argument("--max-tickers", type=int, default=0, help="quick test: only fetch this many tickers")
    p.add_argument("--random-runs", type=int, default=200)
    p.add_argument("--sma", type=int, default=200)
    p.add_argument("--chart", default="docs/precursors_fundamental.png")
    args = p.parse_args(argv)
    n = args.n

    if args.chart_only:
        lift = pd.read_csv(f"{args.out}_lift.csv")
        model = pd.read_csv(f"{args.out}_model.csv")
        model = model[model["year"].notna()] if "year" in model else model
        eq = pd.read_csv(f"{args.out}_equity.csv", index_col=0, parse_dates=True)
        chart(lift, model[model.kind == "year"] if "kind" in model else model, eq, n, args.chart)
        print(f"wrote {args.chart}")
        return 0

    px, member, eligible, spy = load_universe()
    ev_all = pd.read_csv(args.events, parse_dates=["signal_date", "exec_date"])
    ev_all = ev_all[ev_all.signal_date >= "2010-01-01"].copy()
    tickers = list(px.columns)
    cik = ticker_cik_map(tickers, args.refresh)
    pri = sorted(set(ev_all[(ev_all.is_event == 1) | (ev_all.control_sample == 1)].ticker))
    rest = sorted(set(tickers) - set(pri))
    if args.max_tickers:
        pri, rest = pri[: args.max_tickers], []
    print(f"universe {len(tickers)} tickers, {len(cik)} mapped to a CIK ({len(tickers) - len(cik)} unmapped, mostly delisted); "
          f"{sum(t in cik for t in pri)}/{len(pri)} event/control tickers mapped")

    qpath = f"{args.out}_quarterly.csv"
    qdf = build_quarterly(pri, rest, cik, args.refresh, qpath)
    if qdf.empty:
        print("no quarterly data"); return 1
    qdf["filed"] = pd.to_datetime(qdf["filed"])
    qdf["period_end"] = pd.to_datetime(qdf["period_end"])
    covered = sorted(set(qdf.ticker))
    print(f"quarterly table: {len(qdf)} rows, {len(covered)} tickers with usable data; EDGAR stats {EDGAR_STATS}")
    if args.fetch_only:
        return 0

    # ---------------------------------------------------------------- features for every covered eligible (month end, ticker)
    mes = [d for d in T5.month_ends(px.index) if d >= pd.Timestamp("2009-12-01") and d <= pd.Timestamp(args.end)]
    feats = build_features(qdf, px, eligible, spy, mes)
    assert (feats["filed"] <= feats["signal_date"]).all(), "look-ahead: a quarter filed after the signal date was used"
    print(f"features: {len(feats)} (month end, ticker) rows, {feats.ticker.nunique()} tickers, {feats.signal_date.nunique()} month ends")

    ev = ev_all.merge(feats, on=["signal_date", "ticker"], how="inner")
    ev_events = ev_all[ev_all.is_event == 1]
    cov_events = ev[ev.is_event == 1]
    print(f"events 2010-2024: {len(ev_events)} total, {len(cov_events)} with fundamentals ({len(cov_events) / len(ev_events) * 100:.1f}%); "
          f"control rows {int(ev_all.control_sample.sum())} -> {int(ev.control_sample.sum())} covered; all universe rows {len(ev_all)} -> {len(ev)}")
    # spot check
    sc = cov_events.sort_values("signal_date").iloc[len(cov_events) // 2]
    print(f"spot check: {sc.ticker} signal {sc.signal_date.date()} uses quarter {sc.period} (period end {pd.Timestamp(sc.period_end).date()}) "
          f"filed {sc.filed.date()} ({sc.days_since_filing:.0f} days before the signal); rev_yoy={sc.rev_yoy:.3f} eps_yoy={sc.eps_yoy:.3f} react_last={sc.react_last:.4f}")
    ec = ev[(ev.is_event == 1) | (ev.control_sample == 1)].copy()
    ec.to_csv(f"{args.out}_features.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- lift analysis
    lift = lift_table(ec)
    lift.to_csv(f"{args.out}_lift.csv", index=False, encoding="utf-8-sig")
    print("\nsingle-feature lift (events vs sampled controls, 2010-2024):")
    print(lift[["feature", "n", "auc_entry", "lift_q5", "lift_q1", "auc_cont63", "cont_q5", "cont_q1", "median_event", "median_control"]].round(3).to_string(index=False))

    e_acc = ec[(ec.is_event == 1) & (ec.complete63 == 1) & ec.rev_accel.notna()]
    for name, m in (("rev_accel > 0", e_acc.rev_accel > 0), ("rev_accel <= 0", e_acc.rev_accel <= 0)):
        print(f"  entrants with {name}: n={int(m.sum())}, cont63 rate {e_acc[m].cont63.mean() * 100:.1f}%, median ex63 {e_acc[m].ex63.median() * 100:.2f}%, mean ex63 {e_acc[m].ex63.mean() * 100:.2f}%")

    # ---------------------------------------------------------------- time-split models
    ev["exec_year"] = ev["exec_date"].dt.year
    R_all = month_rank(ev, FEATURES)
    X_all = (R_all[FEATURES] - 0.5).to_numpy()
    y1 = ev.is_event.to_numpy(dtype=float)
    y2 = ((ev.is_event == 1) & (ev.cont63 == 1)).to_numpy(dtype=float)
    ex_rank = ev.groupby("signal_date")["ex63"].rank(pct=True)
    ex_rank = (ex_rank - 0.5 / ev.groupby("signal_date")["ex63"].transform("count")).to_numpy()
    is_ec = ((ev.is_event == 1) | (ev.control_sample == 1)).to_numpy()
    complete = (ev.complete63 == 1).to_numpy()
    sig = ev.signal_date.to_numpy()
    years = list(range(args.start_test, pd.Timestamp(args.end).year + 1))
    model_rows, coef_rows = [], []
    models: dict[int, dict[str, np.ndarray]] = {}
    for Y in years:
        cut_T1 = np.datetime64(f"{Y - 1}-11-30")
        cut_T23 = np.datetime64(f"{Y - 1}-08-31")  # 63-trading-day labels of these rows are known by the end of Y-1
        tr1 = is_ec & (sig <= cut_T1)
        tr2 = is_ec & (sig <= cut_T23) & complete
        tr3 = (sig <= cut_T23) & complete
        w1 = logit_fit(X_all[tr1], y1[tr1])
        w2 = logit_fit(X_all[tr2], y2[tr2])
        w3 = ridge_fit(X_all[tr3], ex_rank[tr3])
        models[Y] = {"T1": w1, "T2": w2, "T3": w3}
        te = (ev.exec_year == Y).to_numpy()
        te_ec = te & is_ec
        s1 = pd.Series(predict(w1, X_all[te_ec]), index=ev.index[te_ec])
        s2 = pd.Series(predict(w2, X_all[te_ec]), index=ev.index[te_ec])
        s3 = pd.Series(predict(w3, X_all[te]), index=ev.index[te])
        ics = [spearman(s3[g.index], g["ex63"]) for _, g in ev[te & complete].groupby("signal_date")]
        model_rows.append({"kind": "year", "year": Y, "n_train_T1": int(tr1.sum()), "n_train_T3": int(tr3.sum()), "n_test_ec": int(te_ec.sum()), "n_test_all": int(te.sum()),
                           "auc_T1_entry": auc(s1, ev.is_event[te_ec]), "auc_T2_entry_cont": auc(s2, pd.Series(y2[te_ec], index=ev.index[te_ec])),
                           "auc_T3_entry": auc(s3.loc[ev.index[te_ec]], ev.is_event[te_ec]),
                           "ic_T3_monthly_mean": float(np.nanmean(ics)) if ics else np.nan, "ic_T3_pooled": spearman(s3, pd.Series(ex_rank[te], index=ev.index[te])),
                           "ic_T3_months_pos_pct": float(np.mean([v > 0 for v in ics if not np.isnan(v)]) * 100) if ics else np.nan})
        for name, w in models[Y].items():
            for f, b in zip(FEATURES, w[1:]):
                coef_rows.append({"model": name, "year": Y, "feature": f, "coef": b})
    model = pd.DataFrame(model_rows)
    coefs = pd.DataFrame(coef_rows)
    coef_summary = coefs.groupby(["model", "feature"]).coef.agg(mean="mean", std="std").reset_index()
    sign_stab = coefs.merge(coef_summary[["model", "feature", "mean"]], on=["model", "feature"])
    sign_stab["same"] = np.sign(sign_stab.coef) == np.sign(sign_stab["mean"])
    coef_summary["sign_stability"] = sign_stab.groupby(["model", "feature"]).same.mean().to_numpy()
    coef_rows_out = [{"kind": f"coef_{r.model}", "feature": r.feature, "coef_mean": r["mean"], "coef_std": r["std"], "sign_stability": r.sign_stability} for _, r in coef_summary.iterrows()]
    pd.concat([model, pd.DataFrame(coef_rows_out)], ignore_index=True).to_csv(f"{args.out}_model.csv", index=False, encoding="utf-8-sig")
    print("\nout-of-sample by year:")
    print(model[["year", "n_train_T1", "n_test_ec", "auc_T1_entry", "auc_T2_entry_cont", "ic_T3_monthly_mean", "ic_T3_pooled", "ic_T3_months_pos_pct"]].round(3).to_string(index=False))
    print(f"mean AUC T1 {model.auc_T1_entry.mean():.3f} (range {model.auc_T1_entry.min():.3f}..{model.auc_T1_entry.max():.3f}, {int((model.auc_T1_entry > 0.5).sum())}/{len(model)} years > 0.5); "
          f"T2 {model.auc_T2_entry_cont.mean():.3f} ({int((model.auc_T2_entry_cont > 0.5).sum())}/{len(model)} > 0.5); "
          f"T3 monthly IC {model.ic_T3_monthly_mean.mean():.4f} ({int((model.ic_T3_monthly_mean > 0).sum())}/{len(model)} years > 0)")
    print("\ncoefficients (mean over years, sign stability):")
    cs = coef_summary.pivot(index="feature", columns="model", values=["mean", "sign_stability"]).round(3)
    print(cs.to_string())

    # ---------------------------------------------------------------- walk-forward portfolios on the covered universe
    cov = [t for t in px.columns if t in set(covered)]
    pxc, elc = px[cov], eligible[cov]
    feats["exec_year"] = [d.year + (1 if d.month == 12 else 0) for d in feats.signal_date]
    test_mes = [d for d in mes if (d.year + (1 if d.month == 12 else 0)) in years and d < pd.Timestamp(args.end)]
    fe = feats[feats.signal_date.isin(test_mes)].copy()
    Rf = month_rank(fe, FEATURES)
    Xf = (Rf[FEATURES] - 0.5).to_numpy()
    for name in ("T1", "T2", "T3"):
        fe[f"score_{name}"] = np.nan
    for Y in years:
        m = (fe.exec_year == Y).to_numpy()
        for name in ("T1", "T2", "T3"):
            fe.loc[fe.index[m], f"score_{name}"] = predict(models[Y][name], Xf[m])
    qqq = B.load_series("QQQ")
    irx = B.load_series("IRX")
    qqq_on = qqq > qqq.rolling(args.sma).mean()
    on = {d: bool(qqq_on.reindex([d], method="ffill").iloc[0]) for d in test_mes}
    cash = (irx.reindex(px.index).ffill() / 100 / 252).to_numpy()
    gate = lambda picks: {d: (v if on[d] else []) for d, v in picks.items()}
    groups = {d: g for d, g in fe.groupby("signal_date")}

    def top_by(col: str, k: int) -> dict:
        return {d: groups[d].nlargest(k, col).ticker.tolist() if d in groups else [] for d in test_mes}

    curves: dict[str, pd.Series] = {}
    picks_sets = {f"fund_T3_top{n}": top_by("score_T3", n), f"fund_T3_top{2 * n}": top_by("score_T3", 2 * n), f"fund_T2_top{n}": top_by("score_T2", n),
                  f"fund_T1_top{n}": top_by("score_T1", n), f"rule_rev_accel_top{n}": top_by("rev_accel", n), f"rule_react_last_top{n}": top_by("react_last", n)}
    mom = T5.rank_frames(pxc, elc)["medium"]
    picks_sets[f"mom63_top{n}"] = {d: mom.loc[d].dropna().nlargest(n).index.tolist() for d in test_mes}
    # does the fundamental information ADD to momentum? pre-select the 4N strongest momentum names, then pick N by the fundamental score
    for col_name, label in (("score_T3", "T3"), ("score_T2", "T2"), ("rev_accel", "rev_accel")):
        pk = {}
        for d in test_mes:
            cand = mom.loc[d].dropna().nlargest(4 * n).index
            g = groups[d] if d in groups else None
            pk[d] = g[g.ticker.isin(cand)].nlargest(n, col_name).ticker.tolist() if g is not None else []
        picks_sets[f"mom63_top{4 * n}_then_{label}_top{n}"] = pk
    for name, pk in picks_sets.items():
        curves[name] = T5.run_portfolio(px, pk, cash_daily=cash)
        curves[f"{name}_r{args.sma}"] = T5.run_portfolio(px, gate(pk), cash_daily=cash)
    first = curves[f"fund_T3_top{n}"].index[0]
    for nm, s in (("SPY", spy), ("QQQ", qqq)):
        s = s.reindex(px.index).ffill().loc[first:args.end]
        curves[nm] = s / s.iloc[0]
    qf = pd.DataFrame({"QQQ": qqq.reindex(px.index).ffill()})
    curves[f"QQQ_r{args.sma}"] = T5.run_portfolio(qf, gate({d: ["QQQ"] for d in test_mes}), cash_daily=cash)
    eq = pd.DataFrame(curves).loc[f"{args.start_test}-01-01":args.end]
    eq = eq / eq.iloc[0]
    eq.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")
    summary = pd.DataFrame({k: metrics(eq[k]) for k in eq.columns}).T
    summary.index.name = "portfolio"
    # random control: N covered eligible names at random each month
    rng = np.random.default_rng(args.seed)
    pool = {d: elc.loc[d][elc.loc[d]].index.tolist() for d in test_mes}
    rand = {"none": [], f"qqq_sma{args.sma}": []}
    for _ in range(args.random_runs):
        pk = {d: list(rng.choice(pool[d], size=min(n, len(pool[d])), replace=False)) for d in test_mes}
        for regime, pp in (("none", pk), (f"qqq_sma{args.sma}", gate(pk))):
            nav = T5.run_portfolio(px, pp, cash_daily=cash).loc[f"{args.start_test}-01-01":args.end]
            rand[regime].append(metrics(nav)["cagr_pct"])
    for regime, vals in rand.items():
        v = np.array(vals)
        suffix = "" if regime == "none" else f"_r{args.sma}"
        summary.loc[f"random_top{n}{suffix}_median", "cagr_pct"] = float(np.median(v))
        summary.loc[f"random_top{n}{suffix}_p90", "cagr_pct"] = float(np.percentile(v, 90))
        summary.loc[f"random_top{n}{suffix}_median", "note"] = f"{len(v)} random portfolios, seed {args.seed}"
    summary["time_in_market_pct"] = np.nan
    summary.loc[[k for k in eq.columns if k.endswith(f"_r{args.sma}")], "time_in_market_pct"] = round(np.mean([on[d] for d in test_mes]) * 100, 1)
    summary.to_csv(f"{args.out}_summary.csv", encoding="utf-8-sig")
    print(f"\nwalk-forward portfolios {args.start_test}-01..{args.end[:7]} on the covered universe ({len(cov)} tickers):")
    print(summary.to_string())
    print(f"random top-{n}: median CAGR {np.median(rand['none']):.2f}%, p90 {np.percentile(rand['none'], 90):.2f}% (filtered: median {np.median(rand[f'qqq_sma{args.sma}']):.2f}%, p90 {np.percentile(rand[f'qqq_sma{args.sma}'], 90):.2f}%)")

    chart(lift, model, eq, n, args.chart)
    print(f"\nwrote {qpath}, {args.out}_features.csv, {args.out}_lift.csv, {args.out}_model.csv, {args.out}_equity.csv, {args.out}_summary.csv, {args.chart}")
    print(f"EDGAR: {EDGAR_STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
