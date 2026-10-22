"""Three free, non-price data sources for the S&P 500 point-in-time universe, fetched and turned into
month-end (signal date x ticker) feature frames for the section 20 IC audit (section 21).

Sources
  xbrl    SEC companyconcept API (same cache/rate limit as sp500_precursor_fundamental): total assets, stockholders'
          equity, common shares outstanding, operating cash flow and net income.  Features: asset growth, net share
          issuance (split-adjusted), accruals (NI - CFO) / assets, book-to-market, cash-flow yield.
  form4   SEC "Insider Transactions Data Sets" quarterly zips (2006Q1 ..): open-market purchases (code P) and sales
          (code S) by officers, directors and 10% owners.  Features: net purchase ratio over 6 months, distinct
          open-market buyers over 3 months, net traded value over 6 months / market cap.
  finra   FINRA consolidated short interest (API, bi-monthly, history starts 2017-12): short interest / shares
          outstanding, days to cover, 1- and 3-month change in short interest.

Point-in-time rules
  A value is usable at month end t only if its SEC filing date (or FINRA publication date = settlement date + 12
  calendar days) is <= t.  Year-ago values are the filing whose period end lies 330-400 days before the latest one.
  Names with no insider trade in the window get 0 (neutral), so the insider features cover the whole universe.

Usage
  python sp500_free_signals.py fetch-xbrl    [--refresh]        -> free_signals_xbrl.csv
  python sp500_free_signals.py fetch-form4   [--from 2006q1 --to 2024q4]  -> free_signals_form4.csv
  python sp500_free_signals.py fetch-finra                      -> free_signals_finra.csv
  (the audit script imports build_frames())
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

import sp500_precursor_fundamental as Fd

XBRL_PATH = "free_signals_xbrl.csv"
FORM4_PATH = "free_signals_form4.csv"
FINRA_PATH = "free_signals_finra.csv"
FINRA_LAG_DAYS = 12          # FINRA publishes about 8 business days after the settlement date
STALE_DAYS = 400             # a balance-sheet value older than this at the signal date is not used
INSIDER_ZIP_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"

INSTANT_TAGS = {
    "assets": [("us-gaap", "Assets")],
    "equity": [("us-gaap", "StockholdersEquity"), ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "shares": [("us-gaap", "CommonStockSharesOutstanding"), ("dei", "EntityCommonStockSharesOutstanding")],
}
DURATION_TAGS = {
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"), ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "ni": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"), ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic")],
}
SHARES_FALLBACK = ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")

FEATURES = {
    "asset_growth": ("xbrl", "總資產年增率"), "net_issuance": ("xbrl", "流通股數年變化（淨發行）"), "accruals": ("xbrl", "應計項目 / 總資產"),
    "btm": ("xbrl", "帳面市值比"), "cfo_yield": ("xbrl", "近四季營業現金流殖利率"),
    "ins_npr_6m": ("form4", "內部人淨買進比（6 個月）"), "ins_buyers_3m": ("form4", "內部人公開市場買進人數（3 個月）"), "ins_netval_6m": ("form4", "內部人淨買進金額 / 市值（6 個月）"),
    "si_ratio": ("finra", "空單 / 流通股數"), "si_dtc": ("finra", "空單回補天數"), "si_chg_1m": ("finra", "空單 1 個月變化"), "si_chg_3m": ("finra", "空單 3 個月變化"),
}
EXPECTED_SIGN = {"asset_growth": -1, "net_issuance": -1, "accruals": -1, "btm": +1, "cfo_yield": +1, "ins_npr_6m": +1, "ins_buyers_3m": +1, "ins_netval_6m": +1,
                 "si_ratio": -1, "si_dtc": -1, "si_chg_1m": -1, "si_chg_3m": -1}


# ------------------------------------------------------------------------------------------------ XBRL
def concept(cik: int, ns: str, tag: str, refresh: bool = False) -> dict:
    raw = Fd.edgar_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{ns}/{tag}.json", refresh)
    if not raw:
        return {}
    try:
        return json.loads(raw).get("units", {})
    except json.JSONDecodeError:
        return {}


def _ok_form(e: dict) -> bool:
    f = str(e.get("form", ""))
    return (f.startswith("10-Q") or f.startswith("10-K")) and e.get("val") is not None


def instant_rows(units: dict, unit: str) -> pd.DataFrame:
    """(end, filed, val): earliest filing of each balance-sheet date."""
    rows = [(pd.Timestamp(e["end"]), pd.Timestamp(e["filed"]), float(e["val"])) for e in units.get(unit, []) if _ok_form(e) and "start" not in e]
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val"])
    df = pd.DataFrame(rows, columns=["end", "filed", "val"]).sort_values("filed")
    return df.groupby("end", as_index=False).first().sort_values("end").reset_index(drop=True)


def duration_rows(units: dict, unit: str = "USD") -> pd.DataFrame:
    """(start, end, filed, val, dur): earliest filing of each reported period (quarter, half, nine months or year)."""
    rows = []
    for e in units.get(unit, []):
        if not _ok_form(e) or "start" not in e:
            continue
        s, en = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        rows.append((s, en, pd.Timestamp(e["filed"]), float(e["val"]), (en - s).days))
    if not rows:
        return pd.DataFrame(columns=["start", "end", "filed", "val", "dur"])
    df = pd.DataFrame(rows, columns=["start", "end", "filed", "val", "dur"]).sort_values("filed")
    return df.groupby(["start", "end"], as_index=False).first().sort_values("end").reset_index(drop=True)


def ttm_rows(per: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month value available at each filing: annual periods as reported; a year-to-date period plus the
    previous fiscal year minus the same year-to-date period one year earlier.  (end, filed, val)."""
    if per.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    annual = per[(per.dur >= 340) & (per.dur <= 380)]
    out = [(r.end, r.filed, r.val) for r in annual.itertuples()]
    ytd = per[(per.dur >= 75) & (per.dur <= 290)]
    for r in ytd.itertuples():
        fy = annual[(annual.end - r.start).abs() <= pd.Timedelta(days=12)]
        prev = ytd[((ytd.end - (r.end - pd.DateOffset(years=1))).abs() <= pd.Timedelta(days=12)) & ((ytd.dur - r.dur).abs() <= 12)]
        if fy.empty or prev.empty:
            continue
        filed = max(r.filed, fy.filed.iloc[0], prev.filed.iloc[0])
        out.append((r.end, filed, r.val + fy.val.iloc[0] - prev.val.iloc[0]))
    df = pd.DataFrame(out, columns=["end", "filed", "val"]).sort_values(["end", "filed"])
    return df.groupby("end", as_index=False).first().sort_values("end").reset_index(drop=True)


def fetch_xbrl_company(ticker: str, cik: int, refresh: bool) -> pd.DataFrame:
    frames = []
    for metric, tags in INSTANT_TAGS.items():
        unit = "shares" if metric == "shares" else "USD"
        rows = pd.DataFrame()
        for ns, tag in tags:
            rows = instant_rows(concept(cik, ns, tag, refresh), unit)
            if len(rows) >= 8:
                break
        if metric == "shares" and len(rows) < 8:
            per = duration_rows(concept(cik, *SHARES_FALLBACK, refresh), "shares")
            per = per[(per.dur >= 75) & (per.dur <= 100)]
            rows = per[["end", "filed", "val"]].reset_index(drop=True)
        if not rows.empty:
            frames.append(rows.assign(metric=metric))
    for metric, tags in DURATION_TAGS.items():
        rows = pd.DataFrame()
        for ns, tag in tags:
            rows = ttm_rows(duration_rows(concept(cik, ns, tag, refresh)))
            if len(rows) >= 8:
                break
        if not rows.empty:
            frames.append(rows.assign(metric=metric))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).assign(ticker=ticker, cik=cik)[["ticker", "cik", "metric", "end", "filed", "val"]]


def fetch_xbrl(tickers: list[str], cik: dict[str, int], refresh: bool, path: str = XBRL_PATH) -> pd.DataFrame:
    frames, t0 = [], time.time()
    order = [t for t in tickers if t in cik]
    for i, t in enumerate(order, 1):
        df = fetch_xbrl_company(t, cik[t], refresh)
        if not df.empty:
            frames.append(df)
        if i % 25 == 0 or i == len(order):
            print(f"  xbrl {i}/{len(order)} companies, {Fd.EDGAR_STATS['requests']} live, {Fd.EDGAR_STATS['cached']} cached, {Fd.EDGAR_STATS['404']} missing tags, {time.time() - t0:.0f}s", flush=True)
            pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


# ------------------------------------------------------------------------------------------------ Form 4
def quarters(q_from: str, q_to: str) -> list[str]:
    y0, q0 = int(q_from[:4]), int(q_from[-1]); y1, q1 = int(q_to[:4]), int(q_to[-1])
    out = []
    y, q = y0, q0
    while (y, q) <= (y1, q1):
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def _read_tsv(z: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
    member = next((n for n in z.namelist() if n.upper().endswith(name.upper())), None)
    if member is None:
        raise KeyError(f"{name} not in zip ({z.namelist()[:8]})")
    with z.open(member) as f:
        raw = f.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw), sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=3, on_bad_lines="skip", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    keep = [c for c in usecols if c in df.columns]
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        print(f"    ! {name}: missing columns {missing}", file=sys.stderr)
    return df[keep]


def _to_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    bad = d.isna() & s.notna()
    if bad.any():
        d[bad] = pd.to_datetime(s[bad], errors="coerce")
    return d


def fetch_form4(ciks: set[int], symbols: set[str], q_from: str, q_to: str, path: str = FORM4_PATH, tmp_dir: str = "form4_tmp") -> pd.DataFrame:
    os.makedirs(tmp_dir, exist_ok=True)
    frames, t0 = [], time.time()
    for q in quarters(q_from, q_to):
        fn = os.path.join(tmp_dir, f"{q}.zip")
        if not os.path.exists(fn):
            for attempt in range(4):
                try:
                    r = requests.get(INSIDER_ZIP_URL.format(q=q), headers=Fd.HEADERS, timeout=300)
                    if r.status_code == 200:
                        with open(fn, "wb") as f:
                            f.write(r.content)
                        break
                    print(f"    {q}: HTTP {r.status_code}", file=sys.stderr)
                    time.sleep(5 + 10 * attempt)
                except requests.RequestException as e:
                    print(f"    {q}: {e}", file=sys.stderr)
                    time.sleep(5 + 10 * attempt)
            else:
                print(f"  ! giving up on {q}", file=sys.stderr)
                continue
        try:
            with zipfile.ZipFile(fn) as z:
                sub = _read_tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
                tr = _read_tsv(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
                ow = _read_tsv(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
        except (zipfile.BadZipFile, KeyError) as e:
            print(f"  ! {q}: {e}", file=sys.stderr)
            os.remove(fn)
            continue
        sub["ISSUERCIK"] = pd.to_numeric(sub.ISSUERCIK, errors="coerce")
        sub["ISSUERTRADINGSYMBOL"] = sub.ISSUERTRADINGSYMBOL.astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
        sub = sub[sub.ISSUERCIK.isin(ciks) | sub.ISSUERTRADINGSYMBOL.isin(symbols)]
        tr = tr[tr.TRANS_CODE.isin(["P", "S"]) & tr.ACCESSION_NUMBER.isin(sub.ACCESSION_NUMBER)]
        ow = ow[ow.ACCESSION_NUMBER.isin(tr.ACCESSION_NUMBER)].copy()
        ow["rel"] = ow.RPTOWNER_RELATIONSHIP.astype(str).str.lower()
        ow = ow.groupby("ACCESSION_NUMBER").agg(owner_cik=("RPTOWNERCIK", "first"), n_owners=("RPTOWNERCIK", "nunique"),
                                                is_officer=("rel", lambda s: s.str.contains("officer").any()), is_director=("rel", lambda s: s.str.contains("director").any()),
                                                is_tenpct=("rel", lambda s: s.str.contains("tenpercent|10%|ten percent", regex=True).any())).reset_index()
        df = tr.merge(sub, on="ACCESSION_NUMBER", how="inner").merge(ow, on="ACCESSION_NUMBER", how="left")
        df["filing_date"] = _to_date(df.FILING_DATE)
        df["trans_date"] = _to_date(df.TRANS_DATE)
        df["shares"] = pd.to_numeric(df.TRANS_SHARES, errors="coerce")
        df["price"] = pd.to_numeric(df.TRANS_PRICEPERSHARE, errors="coerce")
        out = pd.DataFrame({"accession": df.ACCESSION_NUMBER, "filing_date": df.filing_date, "trans_date": df.trans_date, "doc_type": df.DOCUMENT_TYPE,
                            "issuer_cik": df.ISSUERCIK, "symbol": df.ISSUERTRADINGSYMBOL, "code": df.TRANS_CODE, "acq_disp": df.TRANS_ACQUIRED_DISP_CD,
                            "shares": df.shares, "price": df.price, "owner_cik": df.owner_cik, "is_officer": df.is_officer, "is_director": df.is_director, "is_tenpct": df.is_tenpct})
        frames.append(out)
        os.remove(fn)
        print(f"  form4 {q}: {len(sub)} filings of universe issuers, {len(out)} P/S transactions, total {sum(len(f) for f in frames)}, {time.time() - t0:.0f}s", flush=True)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


# ------------------------------------------------------------------------------------------------ FINRA
def fetch_finra(tickers: list[str], path: str = FINRA_PATH) -> pd.DataFrame:
    frames, t0, empty = [], time.time(), []
    sess = requests.Session()
    for i, t in enumerate(tickers, 1):
        got = None
        for sym in dict.fromkeys([t, t.replace("-", "."), t.replace("-", "/")]):
            body = {"limit": 5000, "fields": ["settlementDate", "symbolCode", "marketClassCode", "currentShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity", "stockSplitFlag", "revisionFlag"],
                    "compareFilters": [{"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": sym}]}
            for attempt in range(4):
                try:
                    r = sess.post(FINRA_URL, json=body, headers={"User-Agent": Fd.HEADERS["User-Agent"], "Accept": "text/plain"}, timeout=60)
                except requests.RequestException:
                    time.sleep(2 + 3 * attempt)
                    continue
                if r.status_code == 200:
                    got = r.text
                    break
                if r.status_code == 204:
                    got = ""
                    break
                time.sleep(2 + 5 * attempt)
            if got and got.count("\n") > 1:
                break
        if got and got.count("\n") > 1:
            df = pd.read_csv(io.StringIO(got))
            df["ticker"] = t
            frames.append(df)
        else:
            empty.append(t)
        if i % 50 == 0 or i == len(tickers):
            print(f"  finra {i}/{len(tickers)}: {sum(len(f) for f in frames)} rows, {len(empty)} tickers without data, {time.time() - t0:.0f}s", flush=True)
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  finra: no rows for {len(empty)} tickers: {' '.join(empty[:40])}{' ...' if len(empty) > 40 else ''}")
    return out


# ------------------------------------------------------------------------------------------------ feature frames
def _latest_at(series: pd.DataFrame, mes: pd.DatetimeIndex, max_age: int = STALE_DAYS) -> pd.DataFrame:
    """For each signal date the latest row (by filed <= t) of a (end, filed, val) table; rows whose end is older than
    max_age days at t are dropped.  Returns (signal_date, end, val) with NaN where nothing is usable."""
    s = series.sort_values("filed").reset_index(drop=True)
    s = s.assign(filed=s.filed.astype("datetime64[ns]"), end=s.end.astype("datetime64[ns]"))
    mes = pd.DatetimeIndex(mes).astype("datetime64[ns]")
    # the value usable at t is the one with the largest filed <= t, but a later filing may report an OLDER period
    # (amendments); keep, per filed order, the running maximum period end
    s["end_cum"] = s.end.cummax()
    s = s[s.end == s.end_cum]
    m = pd.merge_asof(pd.DataFrame({"signal_date": mes}), s[["filed", "end", "val"]].rename(columns={"filed": "signal_date"}), on="signal_date", direction="backward")
    stale = (m.signal_date - m.end).dt.days > max_age
    m.loc[stale, ["end", "val"]] = np.nan
    return m


def _year_ago(series: pd.DataFrame, ends: pd.Series, asof: pd.Series) -> pd.Series:
    """Value of the row whose end lies 330-400 days before `ends`, filed <= asof (NaN otherwise)."""
    s = series.sort_values("end")
    out = np.full(len(ends), np.nan)
    e_arr, f_arr, v_arr = s.end.to_numpy(), s.filed.to_numpy(), s.val.to_numpy()
    for i, (e, t) in enumerate(zip(ends.to_numpy(), asof.to_numpy())):
        if pd.isna(e):
            continue
        lo, hi = e - np.timedelta64(400, "D"), e - np.timedelta64(330, "D")
        m = (e_arr >= lo) & (e_arr <= hi) & (f_arr <= t)
        if m.any():
            out[i] = v_arr[m][-1]
    return pd.Series(out, index=ends.index)


def _split_adjust(shares: pd.DataFrame) -> pd.DataFrame:
    """Add 'adj' so that val / adj is on the basis of the last observation (today's share basis).  Any jump between consecutive
    observations larger than 40% (a split, a reverse split, or a filer switching between thousands and units) is
    treated as a basis change, because a genuine one-quarter change in shares outstanding of that size is rare in
    S&P 500 companies.  Only observations filed on or before the signal date are ever compared (see _year_ago)."""
    s = shares.sort_values("end").reset_index(drop=True).copy()
    v = s.val.to_numpy()
    adj = np.ones(len(s))
    f = 1.0
    for i in range(1, len(s)):
        if v[i - 1] > 0 and v[i] > 0:
            r = v[i] / v[i - 1]
            if abs(math.log(r)) > math.log(1.4):
                f *= r
        adj[i] = f
    # val / adj is on the basis of the LAST observation, the same basis as Yahoo's split-adjusted prices, so that
    # shares x adjusted price is the market cap of the day.  Later splits enter only as this unit conversion.
    s["adj"] = adj / adj[-1]
    return s


def _plausible_mcap(shares: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Market cap from reported shares and price, with the share count rescaled by 1,000 when a filer reports
    shares in thousands (cap below 100 million dollars) or in units where thousands were expected (above 20 trillion)."""
    mcap = shares * price
    small = (mcap > 0) & (mcap < 1e8)
    big = mcap > 2e13
    mcap = np.where(small, mcap * 1e3, mcap)
    mcap = np.where(big, mcap / 1e3, mcap)
    return np.where(mcap > 0, mcap, np.nan)


def xbrl_frames(xb: pd.DataFrame, px: pd.DataFrame, mes: list[pd.Timestamp]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    mes_idx = pd.DatetimeIndex(mes)
    out = {k: pd.DataFrame(np.nan, index=mes_idx, columns=px.columns) for k in ("asset_growth", "net_issuance", "accruals", "btm", "cfo_yield")}
    covered = pd.DataFrame(False, index=mes_idx, columns=px.columns)
    pxm = px.reindex(mes_idx)
    for t, g in xb.groupby("ticker"):
        if t not in px.columns:
            continue
        tab = {m: gg[["end", "filed", "val"]] for m, gg in g.groupby("metric")}
        if "assets" not in tab:
            continue
        a = _latest_at(tab["assets"], mes_idx)
        covered[t] = a.val.notna().to_numpy()
        a_prev = _year_ago(tab["assets"], a.end, a.signal_date)
        out["asset_growth"][t] = (a.val / a_prev - 1).where(a_prev > 0).to_numpy()
        if "shares" in tab:
            sh = _split_adjust(tab["shares"])
            s_now = _latest_at(sh.assign(val=sh.val / sh.adj), mes_idx)
            s_prev = _year_ago(sh.assign(val=sh.val / sh.adj), s_now.end, s_now.signal_date)
            out["net_issuance"][t] = np.log(s_now.val / s_prev).where((s_now.val > 0) & (s_prev > 0)).to_numpy()
            mcap = _plausible_mcap(s_now.val.to_numpy(), pxm[t].to_numpy())
            if "equity" in tab:
                eq = _latest_at(tab["equity"], mes_idx)
                out["btm"][t] = eq.val.to_numpy() / mcap
            if "cfo" in tab:
                cf = _latest_at(tab["cfo"], mes_idx)
                out["cfo_yield"][t] = cf.val.to_numpy() / mcap
        if "cfo" in tab and "ni" in tab:
            # accruals per period end where both trailing-twelve-month figures exist; usable once both are filed
            j = tab["ni"].merge(tab["cfo"], on="end", suffixes=("_ni", "_cf"))
            if not j.empty:
                acc = pd.DataFrame({"end": j.end, "filed": j[["filed_ni", "filed_cf"]].max(axis=1), "val": j.val_ni - j.val_cf})
                ac = _latest_at(acc, mes_idx)
                out["accruals"][t] = (ac.val / a.val).where(a.val > 0).to_numpy()
    for k in out:
        out[k] = out[k].replace([np.inf, -np.inf], np.nan)
    # values outside these ranges come from tag or unit mix-ups, not from economics
    out["asset_growth"] = out["asset_growth"].where(out["asset_growth"].abs() < 5)
    out["net_issuance"] = out["net_issuance"].where(out["net_issuance"].abs() < 1.5)
    out["accruals"] = out["accruals"].where(out["accruals"].abs() < 1)
    out["btm"] = out["btm"].where((out["btm"] > -5) & (out["btm"] < 20))
    out["cfo_yield"] = out["cfo_yield"].where(out["cfo_yield"].abs() < 2)
    return out, covered


def form4_frames(f4: pd.DataFrame, px: pd.DataFrame, mes: list[pd.Timestamp], cik: dict[str, int], shares_now: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    mes_idx = pd.DatetimeIndex(mes)
    cik_to_t = {}
    for t, c in cik.items():
        cik_to_t.setdefault(c, t)
    f4 = f4.copy()
    f4["ticker"] = f4.issuer_cik.map(cik_to_t)
    miss = f4.ticker.isna() & f4.symbol.isin(set(px.columns))
    f4.loc[miss, "ticker"] = f4.loc[miss, "symbol"]
    f4 = f4[f4.ticker.notna() & f4.filing_date.notna() & (f4.shares > 0)]
    f4 = f4[(f4.is_officer.fillna(False)) | (f4.is_director.fillna(False)) | (f4.is_tenpct.fillna(False))]
    f4["buy"] = f4.code.eq("P")
    f4["value"] = f4.shares * f4.price.fillna(0)
    out = {k: pd.DataFrame(0.0, index=mes_idx, columns=px.columns) for k in ("ins_npr_6m", "ins_buyers_3m", "ins_netval_6m")}
    pxm = px.reindex(mes_idx)
    live = (mes_idx >= f4.filing_date.min() + pd.Timedelta(days=182)) & (mes_idx <= f4.filing_date.max())
    for k in out:
        out[k].loc[~live] = np.nan
    for i, t_end in enumerate(mes_idx):
        if not live[i]:
            continue
        w6 = f4[(f4.filing_date <= t_end) & (f4.filing_date > t_end - pd.Timedelta(days=182))]
        if w6.empty:
            continue
        g = w6.groupby("ticker")
        n_buy, n_sell = g.buy.sum(), (~w6.buy).groupby(w6.ticker).sum()
        npr = (n_buy - n_sell) / (n_buy + n_sell)
        netval = (w6.value.where(w6.buy, -w6.value)).groupby(w6.ticker).sum()
        w3 = w6[(w6.filing_date > t_end - pd.Timedelta(days=91)) & w6.buy]
        buyers = w3.groupby("ticker").owner_cik.nunique()
        for k, s in (("ins_npr_6m", npr), ("ins_buyers_3m", buyers), ("ins_netval_6m", netval)):
            s = s[s.index.isin(px.columns)]
            out[k].iloc[i, out[k].columns.get_indexer(s.index)] = s.to_numpy()
    # net traded value scaled by market cap when shares are known, otherwise by a 1e9 placeholder (rank only matters)
    if shares_now is not None:
        mcap = shares_now.reindex(index=mes_idx, columns=px.columns) * pxm
        nv = out["ins_netval_6m"]
        scaled = nv / mcap.where(mcap > 0)
        # no market cap: a name without trades stays neutral (0), a name with trades cannot be scaled (NaN)
        out["ins_netval_6m"] = scaled.where(mcap > 0, nv.where(nv == 0))
    return out


def finra_frames(si: pd.DataFrame, px: pd.DataFrame, mes: list[pd.Timestamp], shares_now: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    mes_idx = pd.DatetimeIndex(mes)
    si = si.copy()
    si["settlementDate"] = pd.to_datetime(si.settlementDate)
    si["public"] = si.settlementDate + pd.Timedelta(days=FINRA_LAG_DAYS)
    si = si[si.ticker.isin(px.columns)].sort_values(["ticker", "settlementDate"])
    si = si.groupby(["ticker", "settlementDate"], as_index=False).last()   # revisions: keep the last row
    out = {k: pd.DataFrame(np.nan, index=mes_idx, columns=px.columns) for k in ("si_ratio", "si_dtc", "si_chg_1m", "si_chg_3m")}
    for t, g in si.groupby("ticker"):
        g = g.sort_values("settlementDate").reset_index(drop=True)
        g["si"] = pd.to_numeric(g.currentShortPositionQuantity, errors="coerce")
        g["dtc"] = pd.to_numeric(g.daysToCoverQuantity, errors="coerce")
        g["chg1"] = np.log(g.si / g.si.shift(2))
        g["chg3"] = np.log(g.si / g.si.shift(6))
        m = pd.merge_asof(pd.DataFrame({"signal_date": mes_idx}), g[["public", "settlementDate", "si", "dtc", "chg1", "chg3"]].rename(columns={"public": "signal_date"}), on="signal_date", direction="backward")
        stale = (m.signal_date - m.settlementDate).dt.days > 45
        m.loc[stale, ["si", "dtc", "chg1", "chg3"]] = np.nan
        out["si_ratio"][t] = m.si.to_numpy()
        out["si_dtc"][t] = m.dtc.to_numpy()
        out["si_chg_1m"][t] = m.chg1.to_numpy()
        out["si_chg_3m"][t] = m.chg3.to_numpy()
    if shares_now is not None:
        sh = shares_now.reindex(index=mes_idx, columns=px.columns)
        out["si_ratio"] = out["si_ratio"] / sh.where(sh > 0)
    out = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}
    return out


def shares_frame(xb: pd.DataFrame, px: pd.DataFrame, mes: list[pd.Timestamp]) -> pd.DataFrame:
    """Latest known (split-adjusted) shares outstanding per signal date, for market-cap scaling."""
    mes_idx = pd.DatetimeIndex(mes)
    out = pd.DataFrame(np.nan, index=mes_idx, columns=px.columns)
    pxm = px.reindex(mes_idx)
    for t, g in xb[xb.metric == "shares"].groupby("ticker"):
        if t in px.columns:
            sh = _split_adjust(g[["end", "filed", "val"]])
            now = _latest_at(sh.assign(val=sh.val / sh.adj), mes_idx).val.to_numpy()
            out[t] = _plausible_mcap(now, pxm[t].to_numpy()) / pxm[t].to_numpy()
    return out


def build_frames(px: pd.DataFrame, mes: list[pd.Timestamp], cik: dict[str, int]) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, str], pd.DataFrame]:
    """All free-signal frames (signal date x ticker), their group and Chinese label, and the XBRL coverage mask."""
    frames, groups, labels = {}, {}, {}
    xb = pd.read_csv(XBRL_PATH, parse_dates=["end", "filed"])
    xf, covered = xbrl_frames(xb, px, mes)
    shares_now = shares_frame(xb, px, mes)
    frames.update(xf)
    if os.path.exists(FORM4_PATH):
        f4 = pd.read_csv(FORM4_PATH, parse_dates=["filing_date", "trans_date"], low_memory=False)
        frames.update(form4_frames(f4, px, mes, cik, shares_now))
    if os.path.exists(FINRA_PATH):
        si = pd.read_csv(FINRA_PATH, low_memory=False)
        frames.update(finra_frames(si, px, mes, shares_now))
    for k in frames:
        groups[k], labels[k] = FEATURES[k]
    return frames, groups, labels, covered


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["fetch-xbrl", "fetch-form4", "fetch-finra"])
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--from", dest="q_from", default="2006q1")
    p.add_argument("--to", dest="q_to", default="2024q4")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2024-12-31")
    args = p.parse_args(argv)
    from sp500_precursor_events import load_universe
    px, member, eligible, spy = load_universe(args.start, args.end)
    tickers = list(px.columns)
    cik = Fd.ticker_cik_map(tickers)
    print(f"universe {len(tickers)} tickers, {len(cik)} with a CIK")
    if args.cmd == "fetch-xbrl":
        out = fetch_xbrl(tickers, cik, args.refresh)
        print(f"wrote {XBRL_PATH}: {len(out)} rows, {out.ticker.nunique()} tickers; rows per metric:\n{out.groupby('metric').size().to_string()}")
    elif args.cmd == "fetch-form4":
        out = fetch_form4(set(cik.values()), set(tickers), args.q_from, args.q_to)
        print(f"wrote {FORM4_PATH}: {len(out)} transactions, {out.issuer_cik.nunique()} issuers, {out.filing_date.min()}..{out.filing_date.max()}")
    else:
        out = fetch_finra(tickers)
        print(f"wrote {FINRA_PATH}: {len(out)} rows, {out.ticker.nunique()} tickers, {out.settlementDate.min()}..{out.settlementDate.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
