#!/usr/bin/env python3
"""
Shared event table for the "what precedes a momentum surge" study (section 17).

Event: at month end t a point-in-time S&P 500 member enters the top-N of the medium-term (63-day)
momentum ranking for the first time (it was rankable but not in the top-N at month end t-1).
The same definition that sp500_top5_monthly.py trades. Controls: every other stock that was
rankable at t and t-1 and not in the top-N at either date; a seeded sample of CONTROLS_PER_EVENT
per event month is marked (control_sample = 1) for analyses that need a balanced set.

Forward returns are measured from the execution close (first trading day after t) over 21, 63 and
126 trading days, together with SPY over the same windows. Label `cont63` = 1 when the stock beats
SPY over the next 63 trading days ("the surge continued"), 0 otherwise.

Everything here uses only data up to the signal date for features (none are computed here) and
later data only for the labels, so downstream studies can compute features at t without look-ahead.

Usage: python sp500_precursor_events.py [--definition medium] [--n 5] [--out precursor_events.csv]
Outputs: <out> (full pool, git-ignored) and <out>_sample.csv (events + sampled controls, committed).
Import: from sp500_precursor_events import load_universe, build_events
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sp500_backtest_long as L
import sp500_top5_monthly as T5

CONTROLS_PER_EVENT = 10


def load_universe(start: str = "2000-01-01", end: str = "2024-12-31"):
    """Point-in-time prices (zero closes dropped), membership and the eligibility mask used by section 15."""
    px, member = L.load("prices_long_cache.csv", "membership_long.pkl", start, end)
    px = px.replace(0.0, np.nan)
    eligible = member & px.notna() & px.shift(252).notna()
    spy = pd.read_csv("spy_long_cache.csv", index_col=0, parse_dates=True).squeeze().sort_index().reindex(px.index).ffill()
    return px, member, eligible, spy


def build_events(px: pd.DataFrame, eligible: pd.DataFrame, spy: pd.Series, definition: str = "medium", n: int = 5,
                 start: str = "2000-01-01", seed: int = 42) -> pd.DataFrame:
    ranks = T5.rank_frames(px, eligible)[definition]
    mes = [d for d in T5.month_ends(px.index) if d >= pd.Timestamp(start) - pd.Timedelta(days=40)]
    mes = [d for d in mes if ranks.loc[d].notna().sum() >= 50]
    idx = px.index
    pos = {d: idx.get_loc(d) for d in mes}
    rng = np.random.default_rng(seed)
    rows = []
    prev_top: set[str] = set()
    prev_rankable: set[str] = set()
    for k, d in enumerate(mes):
        r = ranks.loc[d].dropna()
        rank = r.rank(ascending=False, method="first")
        top = set(rank[rank <= n].index)
        rankable = set(r.index)
        if k > 0:
            e_pos = pos[d] + 1
            if e_pos >= len(idx):
                break
            exec_date = idx[e_pos]
            p0 = px.iloc[e_pos]
            s0 = spy.iloc[e_pos]
            fwd = {}
            for h in (21, 63, 126):
                j = min(e_pos + h, len(idx) - 1)
                fwd[f"fwd{h}"] = px.iloc[j] / p0 - 1
                fwd[f"spy{h}"] = spy.iloc[j] / s0 - 1
                fwd[f"complete{h}"] = e_pos + h <= len(idx) - 1
            entrants = sorted(t for t in top if t in prev_rankable and t not in prev_top)
            pool = sorted(t for t in rankable - top if t in prev_rankable and t not in prev_top)
            sample = set(rng.choice(pool, size=min(len(pool), CONTROLS_PER_EVENT * max(1, len(entrants))), replace=False)) if pool else set()
            prev_rank = ranks.loc[mes[k - 1]].dropna().rank(ascending=False, method="first")
            for t in entrants + pool:
                is_event = t in top
                rows.append({"signal_date": d, "exec_date": exec_date, "ticker": t, "is_event": int(is_event), "control_sample": int((not is_event) and t in sample),
                             "rank_t": int(rank[t]), "rank_prev": int(prev_rank.get(t, np.nan)) if t in prev_rank else np.nan,
                             "mom_t": float(r[t]), "mom_prev": float(ranks.loc[mes[k - 1]].get(t, np.nan)),
                             "n_rankable": len(rankable),
                             **{key: (float(v[t]) if isinstance(v, pd.Series) else float(v)) for key, v in fwd.items()}})
        prev_top, prev_rankable = top, rankable
    ev = pd.DataFrame(rows)
    for h in (21, 63, 126):
        ev[f"ex{h}"] = ev[f"fwd{h}"] - ev[f"spy{h}"]
    ev["cont63"] = (ev["ex63"] > 0).astype(int)
    ev["definition"] = definition
    ev["n"] = n
    return ev


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--definition", default="medium", choices=["short", "medium", "long", "12_1", "composite"])
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", default="precursor_events.csv")
    args = p.parse_args(argv)
    px, member, eligible, spy = load_universe()
    ev = build_events(px, eligible, spy, args.definition, args.n)
    ev.to_csv(args.out, index=False, encoding="utf-8-sig")  # full pool (~30 MB, git-ignored, rebuilt in a minute)
    ev[(ev.is_event == 1) | (ev.control_sample == 1)].to_csv(args.out.replace(".csv", "_sample.csv"), index=False, encoding="utf-8-sig")
    e = ev[ev.is_event == 1]
    c = ev[ev.is_event == 0]
    print(f"{args.definition} top-{args.n}: {len(e)} entry events over {ev.signal_date.nunique()} month-ends, {len(c)} control rows ({ev.control_sample.sum()} sampled)")
    for h in (21, 63, 126):
        ee = e[e[f"complete{h}"] == 1]; cc = c[c[f"complete{h}"] == 1]
        print(f"  fwd{h}: events mean excess {ee[f'ex{h}'].mean()*100:6.2f}%  median {ee[f'ex{h}'].median()*100:6.2f}%  | controls mean {cc[f'ex{h}'].mean()*100:6.2f}%  median {cc[f'ex{h}'].median()*100:6.2f}%")
    print(f"  cont63 rate: events {e.cont63.mean()*100:.1f}%  controls {c.cont63.mean()*100:.1f}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
