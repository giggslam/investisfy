#!/usr/bin/env python3
"""
Technical precursors of a momentum surge (section 17, technical half).

Question. Section 15 showed that "every month hold the 5 strongest 63-day-momentum S&P 500 stocks"
has no edge. Here we ask what TECHNICAL information visible at month end t distinguishes
  (a) stocks that will ENTER the 63-day-momentum top-5 at the next month end (is_event), and
  (b) entrants whose surge CONTINUES (beat SPY over the following 63 trading days, cont63),
and whether a score built from those precursors, used in the same monthly top-N engine, beats the
plain momentum ranking out of sample.

Inputs (built by sp500_precursor_events.py): precursor_events.csv (one row per month-end x rankable
stock not in the top-5 the month before; 963 entry events; forward labels), the point-in-time price
panel from load_universe(), and daily Yahoo volume (downloaded here, cached in volume_long_cache.csv).

Features at the signal date t (all computed on the full daily frames with rolling windows / shifts so
the value stored at row t uses closes/volumes dated <= t only):
  momentum structure  10/21/63/126/252-day returns, their cross-sectional percentile ranks among
                      eligible members at t, the same ranks 21 and 63 trading days earlier, rank
                      changes, return from the 252-day low, drawdown from the 252-day high, months
                      since that high
  relative strength   21/63-day return minus SPY
  trend               close/SMA20, /SMA50, /SMA200 - 1, SMA alignment (20>50>200), trading days since
                      the close crossed above SMA200 (capped at 252; 252 when below the SMA)
  volatility          21-day realised vol, vol21/vol252, 63-day max drawdown, share of up days (21d)
  volume              volume21/volume126, up-day volume share (21d), days with volume > 2x the 63-day
                      mean in the last 21 days, dollar-volume percentile rank (liquidity)
The 63-day-momentum "family" (r63, pr63, rs63 and the pr63 rank changes) is mechanically tied to the
event definition and is reported separately.

Analyses
  1. Lift: single-feature AUC for is_event (events vs the seeded control sample) and for cont63 among
     events; within-month quintiles (cross-sectional position in the whole eligible universe) with
     P(event | quintile) / P(event) and P(cont63 | quintile); means/medians.
  2. Time-split models (anchored expanding window, retrained every year): features rank-normalised
     to [0,1] within each month (missing -> 0.5). T1 logistic(is_event) on all universe rows, T2
     logistic(is_event & cont63), T3 ridge regression of the within-month rank of ex63. Train on
     signal dates <= 31 Dec Y-1 (T1) or on rows whose 63-day label window ENDS by 31 Dec Y-1 (T2, T3,
     so no label from the test year is seen), test on year Y. Logistic regression is fitted with a
     numpy Newton/IRLS optimiser of the L2-penalised log-likelihood (scikit-learn is not installed).
  3. Walk-forward portfolios: every month end from Dec of (start_test-1), score every eligible stock
     with the model trained through the previous calendar year (T3 score primary, T2 probability
     secondary), buy the top N / 2N at the next close, hold a month (sp500_top5_monthly.run_portfolio,
     0.1% cost), with and without the QQQ > SMA200 month-end filter (T-bills when off). Compared with
     the medium_top5 / medium_top5_r200 ranking re-run on the same window, SPY, QQQ and 200 random
     top-5 portfolios.

No-look-ahead rules: every feature at t is a function of closes/volumes dated <= t (rolling windows,
shifts, cross-sectional ranks within the same date); labels (is_event at t uses the ranking at t,
cont63 uses prices after the execution date) are used only as targets; model for year Y never sees a
row whose label window ends after 31 Dec Y-1; the QQQ filter uses QQQ's close at the signal date.
The script asserts this: a chosen event's 63-day return is recomputed by hand and matched to the
feature and to mom_t in precursor_events.csv, and the whole feature set is recomputed on the price
history truncated at a date T and shown to be identical at T to the values from the full history.

Usage: python sp500_precursor_technical.py [--n 5] [--seed 42] [--start-test 2005] [--out precursor_tech] [--chart-only]
Outputs: precursor_tech_features.csv, precursor_tech_lift.csv, precursor_tech_model.csv,
         precursor_tech_model_coef.csv, precursor_tech_equity.csv, precursor_tech_summary.csv,
         docs/precursors_technical.png, volume_long_cache.csv (cache, git-ignored)
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import sp500_beat_qqq as B
import sp500_top5_monthly as T5
from sp500_precursor_events import load_universe

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREYS = ["#555555", "#8a8a8a", "#b5b5b5", "#2b2b2b"]
MECH_FAMILY = {"r63", "pr63", "rs63", "d_pr63_21", "d_pr63_63"}
VOLUME_FEATURES = ["v_ratio_21_126", "upvol_share21", "n_spike21", "dvol_rank"]
LABELS = {
    "r10": "10 日報酬", "r21": "21 日報酬", "r63": "63 日報酬", "r126": "126 日報酬", "r252": "252 日報酬",
    "pr21": "21 日動能排名", "pr63": "63 日動能排名", "pr126": "126 日動能排名", "pr252": "252 日動能排名",
    "pr21_l21": "21 日排名（1 月前）", "pr63_l21": "63 日排名（1 月前）", "pr252_l21": "252 日排名（1 月前）",
    "pr21_l63": "21 日排名（3 月前）", "pr63_l63": "63 日排名（3 月前）", "pr252_l63": "252 日排名（3 月前）",
    "d_pr63_21": "63 日排名 1 月變化", "d_pr63_63": "63 日排名 3 月變化", "d_pr21_21": "21 日排名 1 月變化", "d_pr252_63": "252 日排名 3 月變化",
    "ret_low252": "距 252 日低點漲幅", "dd_high252": "距 252 日高點回撤", "m_since_high252": "離 252 日高點月數",
    "rs21": "21 日相對 SPY", "rs63": "63 日相對 SPY",
    "sma20_ratio": "收盤 / SMA20", "sma50_ratio": "收盤 / SMA50", "sma200_ratio": "收盤 / SMA200", "sma_align": "均線多頭排列",
    "days_since_x200": "上穿 SMA200 天數", "vol21": "21 日波動", "vol_ratio": "波動壓縮 21/252", "mdd63": "63 日最大回撤",
    "up_share21": "21 日上漲日比例", "v_ratio_21_126": "量比 21/126", "upvol_share21": "上漲日成交量占比", "n_spike21": "爆量日數（21 日）",
    "dvol_rank": "成交金額排名",
}
# raw returns and SPY-relative returns rank identically to their percentile ranks within a month, so the
# model keeps only one copy of each
MODEL_DROP = {"r21", "r63", "r126", "r252", "rs21", "rs63"}


# ----------------------------------------------------------------------------- data
def load_volume(tickers: list[str], path: str = "volume_long_cache.csv", start: str = "1998-06-01", end: str = "2024-12-31") -> pd.DataFrame:
    if os.path.exists(path):
        vol = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        print(f"volume cache {path}: {vol.shape[1]} tickers, {vol.index.min().date()}..{vol.index.max().date()}")
        return vol
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        for attempt in range(2):
            try:
                d = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False, threads=True)
                v = d["Volume"] if isinstance(d.columns, pd.MultiIndex) else d[["Volume"]].rename(columns={"Volume": batch[0]})
                frames.append(v)
                print(f"  volume batch {i // 100 + 1}: {v.notna().any().sum()}/{len(batch)} tickers with data")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  volume batch {i // 100 + 1} attempt {attempt + 1} failed: {e}")
                time.sleep(3)
    vol = pd.concat(frames, axis=1)
    vol.index = pd.DatetimeIndex(vol.index).tz_localize(None)
    vol = vol.loc[:, ~vol.columns.duplicated()]
    vol = vol.loc[:, vol.notna().sum() > 0].sort_index()
    vol.to_csv(path)
    with open(".gitignore", "a", encoding="utf-8") as f:
        f.write(f"{path}\n")
    print(f"downloaded volume for {vol.shape[1]}/{len(tickers)} tickers -> {path}")
    return vol


# ----------------------------------------------------------------------------- features
def _months_since_high(px: pd.DataFrame, w: int = 252) -> pd.DataFrame:
    a = px.ffill().to_numpy(dtype=float)
    a = np.where(np.isnan(a), -np.inf, a)
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= w:
        for j in range(a.shape[1]):
            win = sliding_window_view(a[:, j], w)
            out[w - 1:, j] = (w - 1 - win.argmax(axis=1)) / 21.0
    return pd.DataFrame(out, index=px.index, columns=px.columns)


def _rolling_mdd(px: pd.DataFrame, w: int = 63) -> pd.DataFrame:
    a = px.ffill().to_numpy(dtype=float)
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= w:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j in range(a.shape[1]):
                win = sliding_window_view(a[:, j], w)
                dd = win / np.maximum.accumulate(win, axis=1) - 1
                out[w - 1:, j] = np.nanmin(dd, axis=1)
    return pd.DataFrame(out, index=px.index, columns=px.columns)


def _days_since_cross_up(px: pd.DataFrame, sma: pd.DataFrame, cap: int = 252) -> pd.DataFrame:
    above = px > sma  # False where either is NaN
    cross = above & ~above.shift(1, fill_value=False)
    pos = np.arange(len(px))[:, None]
    last = pd.DataFrame(np.where(cross.to_numpy(), pos, np.nan), index=px.index, columns=px.columns).ffill().to_numpy()
    days = np.minimum(pos - last, cap)
    days = np.where(np.isnan(days), cap, days)
    days = np.where(above.to_numpy(), days, cap).astype(float)
    days[(sma.isna() | px.isna()).to_numpy()] = np.nan
    return pd.DataFrame(days, index=px.index, columns=px.columns)


def compute_features(px: pd.DataFrame, eligible: pd.DataFrame, spy: pd.Series, vol: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Every frame's value at (t, ticker) uses data dated <= t only (shifts, rolling windows, same-date ranks)."""
    F: dict[str, pd.DataFrame] = {}
    ret = px.pct_change()
    for k in (10, 21, 63, 126, 252):
        F[f"r{k}"] = px / px.shift(k) - 1
    for k in (21, 63, 126, 252):
        F[f"pr{k}"] = F[f"r{k}"].where(eligible).rank(axis=1, pct=True)
    for k in (21, 63, 252):
        for lag in (21, 63):
            F[f"pr{k}_l{lag}"] = F[f"pr{k}"].shift(lag)
    F["d_pr63_21"] = F["pr63"] - F["pr63_l21"]
    F["d_pr63_63"] = F["pr63"] - F["pr63_l63"]
    F["d_pr21_21"] = F["pr21"] - F["pr21_l21"]
    F["d_pr252_63"] = F["pr252"] - F["pr252_l63"]
    lo = px.rolling(252, min_periods=200).min()
    hi = px.rolling(252, min_periods=200).max()
    F["ret_low252"] = px / lo - 1
    F["dd_high252"] = px / hi - 1
    F["m_since_high252"] = _months_since_high(px, 252).where(hi.notna())
    for k in (21, 63):
        F[f"rs{k}"] = F[f"r{k}"].sub(spy / spy.shift(k) - 1, axis=0)
    sma20 = px.rolling(20, min_periods=18).mean()
    sma50 = px.rolling(50, min_periods=45).mean()
    sma200 = px.rolling(200, min_periods=180).mean()
    F["sma20_ratio"] = px / sma20 - 1
    F["sma50_ratio"] = px / sma50 - 1
    F["sma200_ratio"] = px / sma200 - 1
    F["sma_align"] = ((sma20 > sma50) & (sma50 > sma200)).astype(float).where(sma200.notna() & px.notna())
    F["days_since_x200"] = _days_since_cross_up(px, sma200)
    vol21 = ret.rolling(21, min_periods=18).std() * math.sqrt(252)
    F["vol21"] = vol21
    F["vol_ratio"] = vol21 / (ret.rolling(252, min_periods=200).std() * math.sqrt(252))
    F["mdd63"] = _rolling_mdd(px, 63).where(px.rolling(63).count() >= 57)
    F["up_share21"] = (ret > 0).astype(float).where(ret.notna()).rolling(21, min_periods=18).mean()
    if vol is not None:
        v = vol.reindex(index=px.index, columns=px.columns).astype(float)
        v = v.where(px.notna())
        F["v_ratio_21_126"] = v.rolling(21, min_periods=15).mean() / v.rolling(126, min_periods=100).mean()
        up_v = (v * (ret > 0).astype(float)).where(v.notna() & ret.notna())
        F["upvol_share21"] = up_v.rolling(21, min_periods=15).sum() / v.where(ret.notna()).rolling(21, min_periods=15).sum()
        spike = (v > 2 * v.rolling(63, min_periods=50).mean()).astype(float).where(v.notna())
        F["n_spike21"] = spike.rolling(21, min_periods=15).sum()
        F["dvol_rank"] = (px * v).rolling(21, min_periods=15).mean().where(eligible).rank(axis=1, pct=True)
    for k, fr in F.items():
        F[k] = fr.replace([np.inf, -np.inf], np.nan)
    return F


def build_panel(px: pd.DataFrame, eligible: pd.DataFrame, F: dict[str, pd.DataFrame], mes: list[pd.Timestamp]) -> pd.DataFrame:
    """One row per (month end, eligible & 63-day-rankable stock) with every feature looked up at that date."""
    rankable = (eligible & F["r63"].notna()).to_numpy()
    idx = px.index
    rows, cols = [], []
    for d in mes:
        p = idx.get_loc(d)
        c = np.where(rankable[p])[0]
        rows.append(np.full(len(c), p))
        cols.append(c)
    rp, cp = np.concatenate(rows), np.concatenate(cols)
    data = {"signal_date": idx[rp], "ticker": px.columns[cp]}
    for name, fr in F.items():
        data[name] = fr.to_numpy()[rp, cp]
    return pd.DataFrame(data)


# ----------------------------------------------------------------------------- statistics
def auc(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s) & np.isfinite(y)
    y, s = y[m], s[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float = 10.0, iters: int = 40) -> np.ndarray:
    """L2-penalised logistic regression (intercept unpenalised) by Newton/IRLS with step halving."""
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    R = lam * np.eye(p + 1)
    R[0, 0] = 0.0
    b = np.zeros(p + 1)
    b[0] = math.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))

    def loss(beta):
        z = Xb @ beta
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * beta @ R @ beta

    cur = loss(b)
    for _ in range(iters):
        z = Xb @ b
        pr = 1 / (1 + np.exp(-z))
        g = Xb.T @ (pr - y) + R @ b
        W = pr * (1 - pr)
        H = (Xb * W[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4:
            cand = b - t * step
            lc = loss(cand)
            if lc <= cur:
                break
            t /= 2
        b, prev, cur = cand, cur, lc
        if abs(prev - cur) < 1e-9 * max(1.0, abs(cur)):
            break
    return b


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    R = lam * np.eye(p + 1)
    R[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + R, Xb.T @ y)


def predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return b[0] + X @ b[1:]


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation without scipy (Pearson on average ranks)."""
    m = a.notna() & b.notna()
    if m.sum() < 20:
        return np.nan
    return float(np.corrcoef(a[m].rank().to_numpy(), b[m].rank().to_numpy())[0, 1])


def monthly_ic(df: pd.DataFrame, score: str, target: str) -> pd.Series:
    return pd.Series({d: spearman(g[score], g[target]) for d, g in df.groupby("signal_date")})


# ----------------------------------------------------------------------------- portfolio helpers
def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1
    yearly = nav.resample("YE").last() / nav.resample("YE").first() - 1
    return {"cagr_pct": round(((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100, 2),
            "ann_vol_pct": round(r.std() * math.sqrt(252) * 100, 1),
            "sharpe": round(r.mean() / r.std() * math.sqrt(252), 2),
            "max_dd_pct": round(dd.min() * 100, 1),
            "worst_year_pct": round(yearly.min() * 100, 1),
            "best_year_pct": round(yearly.max() * 100, 1)}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-test", type=int, default=2005)
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--events", default="precursor_events.csv")
    p.add_argument("--random-runs", type=int, default=200)
    p.add_argument("--lam", type=float, default=10.0, help="L2 penalty for the logistic / ridge models")
    p.add_argument("--out", default="precursor_tech")
    p.add_argument("--chart", default="docs/precursors_technical.png")
    p.add_argument("--chart-only", action="store_true")
    args = p.parse_args(argv)
    if args.chart_only:
        lift = pd.read_csv(f"{args.out}_lift.csv")
        model = pd.read_csv(f"{args.out}_model.csv")
        eq = pd.read_csv(f"{args.out}_equity.csv", index_col=0, parse_dates=True)
        chart(lift, model, eq, args.n, args.chart)
        print(f"wrote {args.chart}")
        return 0

    t0 = time.time()
    px, member, eligible, spy = load_universe(end=args.end)
    ev = pd.read_csv(args.events, parse_dates=["signal_date", "exec_date"])
    print(f"universe {px.shape[1]} tickers {px.index[0].date()}..{px.index[-1].date()}; events table {len(ev)} rows, {ev.is_event.sum()} events, {ev.control_sample.sum()} sampled controls")

    # 1. volume -------------------------------------------------------------------------------------
    vol = load_volume(list(px.columns))
    have = [c for c in px.columns if c in vol.columns and vol[c].notna().any()]
    print(f"volume coverage: {len(have)}/{px.shape[1]} tickers have Yahoo volume ({len(have) / px.shape[1] * 100:.1f}%)")

    # 2. features -----------------------------------------------------------------------------------
    F = compute_features(px, eligible, spy, vol)
    feats = list(F)
    mes = [d for d in T5.month_ends(px.index) if (eligible.loc[d] & F["r63"].loc[d].notna()).sum() >= 50]
    panel = build_panel(px, eligible, F, mes)
    print(f"features: {len(feats)} ({time.time() - t0:.0f}s); panel {len(panel)} rows over {len(mes)} month-ends {mes[0].date()}..{mes[-1].date()}")

    # --- no-look-ahead checks
    e0 = ev[ev.is_event == 1].iloc[0]
    pos0 = px.index.get_loc(e0.signal_date)
    hand = px.iloc[pos0][e0.ticker] / px.iloc[pos0 - 63][e0.ticker] - 1
    feat = F["r63"].loc[e0.signal_date, e0.ticker]
    assert abs(hand - feat) < 1e-10 and abs(hand - e0.mom_t) < 1e-9, (hand, feat, e0.mom_t)
    print(f"spot check {e0.ticker} {e0.signal_date.date()}: 63-day return by hand {hand:.6f} = feature {feat:.6f} = mom_t {e0.mom_t:.6f}")
    T = pd.Timestamp("2012-06-29")
    Ft = compute_features(px.loc[:T], eligible.loc[:T], spy.loc[:T], vol.loc[:T])
    bad = []
    for k in feats:
        a = F[k].loc[T].to_numpy()
        b = Ft[k].loc[T].to_numpy()
        same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9)
        if not same.all():
            bad.append((k, int((~same).sum())))
    assert not bad, f"features differ when the history is truncated at {T.date()}: {bad}"
    print(f"truncation check at {T.date()}: all {len(feats)} features identical with and without post-{T.date()} data (no look-ahead)")

    # merge events with features; coverage
    ev = ev.merge(panel, on=["signal_date", "ticker"], how="left")
    miss = ev["r63"].isna().mean()
    assert miss < 0.001, f"{miss:.3%} of event rows missing from the panel"
    vol_cov_rows = ev[VOLUME_FEATURES[0]].notna().mean()
    vol_cov_events = ev.loc[ev.is_event == 1, VOLUME_FEATURES[0]].notna().mean()
    print(f"volume feature coverage: {vol_cov_rows * 100:.1f}% of universe rows, {vol_cov_events * 100:.1f}% of event rows")
    idx = px.index
    exec_pos = idx.get_indexer(ev.exec_date)
    ev["label_end"] = idx[np.minimum(exec_pos + 63, len(idx) - 1)]
    sub = ev[(ev.is_event == 1) | (ev.control_sample == 1)]
    keep = ["signal_date", "ticker", "is_event", "control_sample", "cont63", "ex63", "complete63", "rank_t", "rank_prev"] + feats
    sub[keep].to_csv(f"{args.out}_features.csv", index=False, encoding="utf-8-sig")

    # 3. lift ---------------------------------------------------------------------------------------
    # quintiles = position within the whole eligible universe that month
    q = panel.groupby("signal_date")[feats].rank(pct=True)
    q = np.ceil(q * 5).clip(1, 5)
    q.columns = [f"q_{c}" for c in feats]
    panel_q = pd.concat([panel[["signal_date", "ticker"]], q], axis=1)
    sub = sub.merge(panel_q, on=["signal_date", "ticker"], how="left")
    evs = sub[(sub.is_event == 1) & (sub.complete63 == 1)]
    base = sub.is_event.mean()
    rows = []
    for k in feats:
        y = sub.is_event.to_numpy(dtype=float)
        rec = {"feature": k, "label": LABELS.get(k, k), "mechanical63": int(k in MECH_FAMILY),
               "n_events": int(sub.loc[sub.is_event == 1, k].notna().sum()), "n_controls": int(sub.loc[sub.is_event == 0, k].notna().sum()),
               "auc_entry": auc(y, sub[k].to_numpy(dtype=float)),
               "auc_cont63": auc(evs.cont63.to_numpy(dtype=float), evs[k].to_numpy(dtype=float)),
               "mean_event": sub.loc[sub.is_event == 1, k].mean(), "mean_control": sub.loc[sub.is_event == 0, k].mean(),
               "median_event": sub.loc[sub.is_event == 1, k].median(), "median_control": sub.loc[sub.is_event == 0, k].median()}
        qk = sub[f"q_{k}"]
        for j in range(1, 6):
            m = qk == j
            rec[f"lift_q{j}"] = sub.loc[m, "is_event"].mean() / base if m.sum() else np.nan
            me = (evs[f"q_{k}"] == j)
            rec[f"cont_q{j}"] = evs.loc[me, "cont63"].mean() if me.sum() >= 10 else np.nan
            rec[f"n_ev_q{j}"] = int(me.sum())
        rows.append(rec)
    lift = pd.DataFrame(rows)
    lift["abs_auc_entry"] = (lift.auc_entry - 0.5).abs()
    lift["abs_auc_cont63"] = (lift.auc_cont63 - 0.5).abs()
    lift = lift.sort_values("abs_auc_entry", ascending=False)
    lift.to_csv(f"{args.out}_lift.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = ["feature", "auc_entry", "lift_q5", "lift_q1", "auc_cont63", "cont_q5", "cont_q1", "median_event", "median_control"]
    print(f"\n-- entry precursors (events vs {int(sub.control_sample.sum())} sampled controls; base P(event)={base:.3f}), all features")
    print(lift[show].head(12).round(3).to_string(index=False))
    print("\n-- entry precursors EXCLUDING the mechanical 63-day family")
    print(lift[lift.mechanical63 == 0][show].head(12).round(3).to_string(index=False))
    print(f"\n-- continuation precursors (cont63 among {len(evs)} events with complete 63-day windows; base {evs.cont63.mean():.3f})")
    print(lift.sort_values("abs_auc_cont63", ascending=False)[show].head(12).round(3).to_string(index=False))

    # 4. time-split models --------------------------------------------------------------------------
    model_feats = [k for k in feats if k not in MODEL_DROP]
    Z = panel.groupby("signal_date")[model_feats].rank(pct=True).fillna(0.5) - 0.5
    Z.columns = [f"z_{c}" for c in model_feats]
    zcols = list(Z.columns)
    panel_z = pd.concat([panel[["signal_date", "ticker"]], Z], axis=1)
    ev = ev.merge(panel_z, on=["signal_date", "ticker"], how="left")
    ev["y_t2"] = ((ev.is_event == 1) & (ev.cont63 == 1)).astype(float)
    ok63 = ev.complete63 == 1
    ev["y_t3"] = np.nan
    ev.loc[ok63, "y_t3"] = ev[ok63].groupby("signal_date")["ex63"].rank(pct=True) - 0.5
    zx = [c for c in zcols if c[2:] not in MECH_FAMILY]
    years = list(range(args.start_test - 1, pd.Timestamp(args.end).year + 1))
    models: dict[int, dict[str, np.ndarray]] = {}
    yearly_rows = []
    for Y in years:
        cutoff = pd.Timestamp(f"{Y - 1}-12-31")
        tr1 = ev[ev.signal_date <= cutoff]
        tr23 = ev[(ev.label_end <= cutoff) & ok63]
        X1, X23 = tr1[zcols].to_numpy(), tr23[zcols].to_numpy()
        b1 = fit_logistic(X1, tr1.is_event.to_numpy(dtype=float), args.lam)
        b1x = fit_logistic(tr1[zx].to_numpy(), tr1.is_event.to_numpy(dtype=float), args.lam)  # entry without the mechanical 63-day family
        b2 = fit_logistic(X23, tr23.y_t2.to_numpy(), args.lam)
        b3 = fit_ridge(X23, tr23.y_t3.to_numpy(), args.lam)
        models[Y] = {"t1": b1, "t1x": b1x, "t2": b2, "t3": b3}
        if Y < args.start_test:
            continue
        te = ev[ev.signal_date.dt.year == Y].copy()
        te["s1"], te["s2"], te["s3"] = predict(b1, te[zcols].to_numpy()), predict(b2, te[zcols].to_numpy()), predict(b3, te[zcols].to_numpy())
        te["s1x"] = predict(b1x, te[zx].to_numpy())
        te63 = te[te.complete63 == 1]
        ic3 = monthly_ic(te63, "s3", "ex63")
        ic2 = monthly_ic(te63, "s2", "ex63")
        yearly_rows.append({"year": Y, "n_train_t1": len(tr1), "n_train_t23": len(tr23), "n_test": len(te), "n_events_test": int(te.is_event.sum()),
                            "t1_auc_entry": auc(te.is_event.to_numpy(dtype=float), te.s1.to_numpy()),
                            "t1x_auc_entry_no63": auc(te.is_event.to_numpy(dtype=float), te.s1x.to_numpy()),
                            "t2_auc_entry_and_cont": auc(te63.y_t2.to_numpy(), te63.s2.to_numpy()),
                            "t2_auc_cont_within_events": auc(te63[te63.is_event == 1].cont63.to_numpy(dtype=float), te63[te63.is_event == 1].s2.to_numpy()),
                            "t3_ic_ex63": ic3.mean(), "t3_ic_hit_rate": (ic3 > 0).mean(), "t2_ic_ex63": ic2.mean(),
                            "t3_ic_within_events": spearman(te63[te63.is_event == 1].s3, te63[te63.is_event == 1].ex63)})
    model = pd.DataFrame(yearly_rows)
    model.to_csv(f"{args.out}_model.csv", index=False, encoding="utf-8-sig")
    test_years = [Y for Y in years if Y >= args.start_test]
    coef = pd.DataFrame({"feature": model_feats, "label": [LABELS.get(k, k) for k in model_feats]})
    for key in ("t1", "t1x", "t2", "t3"):
        names = model_feats if key != "t1x" else [c[2:] for c in zx]
        M = np.array([models[Y][key][1:] for Y in test_years])
        cm = pd.Series(M.mean(axis=0), index=names).reindex(model_feats)
        coef[f"{key}_coef_mean"] = cm.to_numpy()
        coef[f"{key}_sign_stability"] = pd.Series(np.where(M.mean(axis=0) >= 0, (M > 0).mean(axis=0), (M < 0).mean(axis=0)), index=names).reindex(model_feats).to_numpy()
        coef[f"{key}_coef_last"] = pd.Series(models[test_years[-1]][key][1:], index=names).reindex(model_feats).to_numpy()
    coef.to_csv(f"{args.out}_model_coef.csv", index=False, encoding="utf-8-sig")
    print("\n-- out-of-sample by year (model trained through the previous year)")
    print(model[["year", "n_train_t23", "n_events_test", "t1_auc_entry", "t1x_auc_entry_no63", "t2_auc_entry_and_cont", "t2_auc_cont_within_events", "t3_ic_ex63", "t3_ic_hit_rate", "t3_ic_within_events"]].round(3).to_string(index=False))
    for c in ("t1_auc_entry", "t1x_auc_entry_no63", "t2_auc_entry_and_cont", "t2_auc_cont_within_events"):
        print(f"  {c}: mean {model[c].mean():.3f}  range {model[c].min():.3f}..{model[c].max():.3f}  years>0.5: {(model[c] > 0.5).sum()}/{len(model)}")
    for c in ("t3_ic_ex63", "t2_ic_ex63", "t3_ic_within_events"):
        print(f"  {c}: mean {model[c].mean():.4f}  range {model[c].min():.3f}..{model[c].max():.3f}  years>0: {(model[c] > 0).sum()}/{len(model)}")
    print("\n-- coefficients (mean over yearly refits, sign stability = share of years with the majority sign), sorted by |T3|")
    print(coef.sort_values("t3_coef_mean", key=np.abs, ascending=False)[["feature", "t1_coef_mean", "t1_sign_stability", "t1x_coef_mean", "t1x_sign_stability", "t2_coef_mean", "t2_sign_stability", "t3_coef_mean", "t3_sign_stability"]].round(3).to_string(index=False))

    # 5. walk-forward portfolios --------------------------------------------------------------------
    first_sig = pd.Timestamp(f"{args.start_test - 1}-12-01")
    trade_mes = [d for d in mes if d >= first_sig and idx.get_loc(d) + 1 < len(idx)]
    qqq = B.load_series("QQQ")
    irx = B.load_series("IRX")
    qqq_on = qqq > qqq.rolling(200).mean()
    on = {d: bool(qqq_on.reindex([d], method="ffill").iloc[0]) for d in trade_mes}
    cash = (irx.reindex(idx).ffill() / 100 / 252).to_numpy()
    gate = lambda picks: {d: (v if on[d] else []) for d, v in picks.items()}  # noqa: E731
    pz = panel_z[panel_z.signal_date.isin(trade_mes)].copy()
    pz["year"] = pz.signal_date.dt.year
    for key in ("t3", "t2"):
        pz[f"s_{key}"] = np.nan
        for Y, g in pz.groupby("year"):
            pz.loc[g.index, f"s_{key}"] = predict(models[Y][key], g[zcols].to_numpy())
    ranks = T5.rank_frames(px, eligible)["medium"]
    curves: dict[str, pd.Series] = {}
    sizes = [args.n, 2 * args.n]
    for key, nm in (("t3", "score"), ("t2", "prob")):
        for n in sizes:
            picks = {d: g.nlargest(n, f"s_{key}").ticker.tolist() for d, g in pz.groupby("signal_date")}
            curves[f"precursor_{nm}_top{n}"] = T5.run_portfolio(px, picks, cash_daily=cash)
            curves[f"precursor_{nm}_top{n}_r200"] = T5.run_portfolio(px, gate(picks), cash_daily=cash)
    for n in sizes:
        picks = {d: ranks.loc[d].dropna().nlargest(n).index.tolist() for d in trade_mes}
        curves[f"medium_top{n}"] = T5.run_portfolio(px, picks, cash_daily=cash)
        curves[f"medium_top{n}_r200"] = T5.run_portfolio(px, gate(picks), cash_daily=cash)
    first = curves[f"medium_top{args.n}"].index[0]
    for nm, s in (("SPY", spy), ("QQQ", qqq)):
        s = s.reindex(idx).ffill().loc[first:args.end]
        curves[nm] = s / s.iloc[0]
    eq = pd.DataFrame(curves).loc[first:args.end]
    eq = eq / eq.iloc[0]
    eq.to_csv(f"{args.out}_equity.csv", encoding="utf-8-sig")
    print(f"\nportfolios {eq.index[0].date()}..{eq.index[-1].date()} ({time.time() - t0:.0f}s so far); random controls...")
    rng = np.random.default_rng(args.seed)
    elig_lists = {d: eligible.loc[d][eligible.loc[d]].index.tolist() for d in trade_mes}
    rand = []
    for k in range(args.random_runs):
        picks = {d: list(rng.choice(elig_lists[d], size=min(args.n, len(elig_lists[d])), replace=False)) for d in trade_mes}
        for regime, pk in (("none", picks), ("r200", gate(picks))):
            nav = T5.run_portfolio(px, pk, cash_daily=cash).loc[first:args.end]
            rand.append({"run": k, "regime": regime, "cagr_pct": B.cagr(nav) * 100, "max_dd_pct": (nav / nav.cummax() - 1).min() * 100})
    rand = pd.DataFrame(rand)
    rows = []
    for c in eq.columns:
        m = metrics(eq[c])
        rows.append({"strategy": c, **m,
                     f"cagr_{args.start_test}_2009_pct": round(T5.period_cagr(eq[c], f"{args.start_test}-01-01", "2009-12-31") * 100, 2),
                     "cagr_2010_2019_pct": round(T5.period_cagr(eq[c], "2010-01-01", "2019-12-31") * 100, 2),
                     "cagr_2020_2024_pct": round(T5.period_cagr(eq[c], "2020-01-01", args.end) * 100, 2)})
    for regime in ("none", "r200"):
        rn = rand[rand.regime == regime]
        rows.append({"strategy": f"random_top{args.n}_median" + ("" if regime == "none" else "_r200"), "cagr_pct": round(rn.cagr_pct.median(), 2), "max_dd_pct": round(rn.max_dd_pct.median(), 1)})
        rows.append({"strategy": f"random_top{args.n}_p90" + ("" if regime == "none" else "_r200"), "cagr_pct": round(rn.cagr_pct.quantile(0.9), 2), "max_dd_pct": round(rn.max_dd_pct.quantile(0.1), 1)})
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{args.out}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    yearly = (eq.resample("YE").last() / eq.resample("YE").first() - 1) * 100
    yearly.index = yearly.index.year
    print(yearly[[f"precursor_score_top{args.n}", f"precursor_score_top{args.n}_r200", f"precursor_prob_top{args.n}", f"medium_top{args.n}", f"medium_top{args.n}_r200", "SPY", "QQQ"]].round(1).to_string())

    chart(lift, model, eq, args.n, args.chart)
    print(f"\nwrote {args.out}_features.csv, _lift.csv, _model.csv, _model_coef.csv, _equity.csv, _summary.csv, {args.chart} ({time.time() - t0:.0f}s)")
    return 0


# ----------------------------------------------------------------------------- chart
def chart(lift: pd.DataFrame, model: pd.DataFrame, eq: pd.DataFrame, n: int, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.35], hspace=0.32, wspace=0.55)
    ax_a1 = fig.add_subplot(gs[0, 0]); ax_a2 = fig.add_subplot(gs[0, 1]); ax_b1 = fig.add_subplot(gs[0, 2]); ax_b2 = fig.add_subplot(gs[0, 3])
    ax_c = fig.add_subplot(gs[1, :])

    def hbars(ax, df, col, color, title):
        df = df.sort_values(col, key=lambda s: (s - 0.5).abs(), ascending=False).head(12).iloc[::-1]
        vals = df[col] - 0.5
        y = np.arange(len(df))
        ax.barh(y, vals, color=color, height=0.62)
        ax.set_yticks(y); ax.set_yticklabels(df["label"], fontsize=8.5)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v + 0.5:.3f}", (v, yi), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points", ha="left" if v >= 0 else "right", va="center", fontsize=7.5, color="#333333")
        ax.axvline(0, color="#888888", lw=0.8)
        lim = max(abs(vals.min()), abs(vals.max())) * 1.45 + 0.01
        ax.set_xlim(-lim if vals.min() < 0 else -0.02, lim)
        ax.set_xlabel("AUC - 0.5（正值 = 數值越高機率越高）", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

    hbars(ax_a1, lift[lift.mechanical63 == 0], "auc_entry", PAL[0], "(a1) 下月進入前 5 的技術前兆\n（排除 63 日動能本身，前 12 個 |AUC-0.5|）")
    hbars(ax_a2, lift, "auc_cont63", PAL[1], "(a2) 進入後續漲（63 日勝過 SPY）的前兆\n（僅事件樣本，前 12 個 |AUC-0.5|）")

    yrs = model["year"]
    ax_b1.plot(yrs, model["t1_auc_entry"], color=PAL[0], lw=2, marker="o", ms=4, label="T1 下月進入前 5")
    ax_b1.plot(yrs, model["t1x_auc_entry_no63"], color=PAL[3], lw=2, marker="D", ms=4, label="T1 下月進入（排除 63 日動能）")
    ax_b1.plot(yrs, model["t2_auc_entry_and_cont"], color=PAL[1], lw=2, marker="s", ms=4, label="T2 進入且續漲")
    ax_b1.plot(yrs, model["t2_auc_cont_within_events"], color=PAL[2], lw=1.6, marker="^", ms=4, ls="--", label="T2 事件內：續漲 vs 不續漲")
    ax_b1.axhline(0.5, color="#888888", lw=0.9, ls=":")
    ax_b1.set_ylim(0.25, 1.02)
    ax_b1.set_title("(b1) 樣本外 AUC（每年以前一年底前資料重訓）", loc="left", fontsize=10.5)
    ax_b1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    ax_b1.grid(alpha=0.25); ax_b1.spines[["top", "right"]].set_visible(False)
    ax_b1.set_ylabel("AUC")
    cols = np.where(model["t3_ic_ex63"] >= 0, PAL[0], PAL[1])
    ax_b2.bar(yrs, model["t3_ic_ex63"], color=cols, width=0.7)
    ax_b2.axhline(0, color="#888888", lw=0.9)
    ax_b2.axhline(model["t3_ic_ex63"].mean(), color=GREYS[0], lw=1, ls="--")
    ax_b2.text(0.02, 0.97, f"年平均 IC {model['t3_ic_ex63'].mean():+.3f}（虛線）\n{int((model['t3_ic_ex63'] > 0).sum())}/{len(model)} 年為正", transform=ax_b2.transAxes, va="top", fontsize=8.5, color=GREYS[0])
    ax_b2.set_title("(b2) T3 分數對未來 63 日超額報酬的\n樣本外月度 Spearman IC（年平均）", loc="left", fontsize=10.5)
    ax_b2.set_ylabel("IC")
    ax_b2.grid(axis="y", alpha=0.25); ax_b2.spines[["top", "right"]].set_visible(False)

    series = [(f"precursor_score_top{n}", f"前兆分數（T3）前 {n}", PAL[0], "-", 1.8),
              (f"precursor_score_top{n}_r200", f"前兆分數前 {n} + QQQ 200 日線濾網", PAL[1], "-", 1.8),
              (f"precursor_prob_top{n}", f"前兆機率（T2）前 {n}", PAL[2], "-", 1.3),
              (f"medium_top{n}", f"63 日動能前 {n}", GREYS[0], "--", 1.3),
              (f"medium_top{n}_r200", f"63 日動能前 {n} + 濾網", GREYS[1], "--", 1.3),
              ("SPY", "SPY", GREYS[3], ":", 1.3), ("QQQ", "QQQ", GREYS[2], ":", 1.3)]
    ends = []
    for c, lab, col, ls, lw in series:
        if c not in eq:
            continue
        ax_c.plot(eq.index, eq[c], color=col, ls=ls, lw=lw, label=lab)
        ends.append((eq[c].iloc[-1], lab, col))
    ax_c.set_yscale("log")
    ends.sort(key=lambda t: t[0])
    ys = []
    for v, lab, col in ends:
        y = math.log(v)
        if ys and y - ys[-1] < 0.13:
            y = ys[-1] + 0.13
        ys.append(y)
        ax_c.annotate(f"{lab}  {v:.1f}x", (eq.index[-1], math.exp(y)), xytext=(6, 0), textcoords="offset points", color=col, fontsize=9, va="center")
    ax_c.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=365 * 4))
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16]
    ax_c.set_yticks(ticks); ax_c.set_yticklabels([f"{t:g}x" for t in ticks])
    ax_c.set_title(f"(c) 走動式回測 {eq.index[0].year} 至 {eq.index[-1].year}：每月以「只用前一年底前資料訓練」的前兆分數選前 {n} 隻，對 63 日動能前 {n} 與基準（淨值，對數座標）", loc="left", fontsize=11.5)
    ax_c.set_ylabel("淨值（倍，對數座標）")
    ax_c.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax_c.grid(alpha=0.25); ax_c.spines[["top", "right"]].set_visible(False)
    fig.suptitle("動能爆發的技術前兆：單因子 lift、樣本外模型、走動式組合", x=0.01, ha="left", fontsize=14, y=0.995)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
