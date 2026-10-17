#!/usr/bin/env python3
"""
One comparison table for every strategy in the study.

Collects the saved equity curves of all stages, evaluates them on one common window
(2000-03-01 .. 2024-12-31) with one metric set, adds the same 5,000 paired random draws
(1-20 year holds, seed 42, curves extended to 2026-10 where available) and a separate
2025-01-02 .. 2026-10-16 table. Strategies without a saved curve (the 2000-2024 strategy
search winners) are listed from their grid metrics and flagged.

Outputs: strategy_table_2000_2024.csv (all rules), strategy_table_2025_2026.csv,
         docs/12-strategy-table.md (curated rows), docs/strategy_table.png
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

import sp500_random_entry_mc as MC

COMMON_START, COMMON_END = pd.Timestamp("2000-03-01"), pd.Timestamp("2024-12-31")
SHORT_START, SHORT_END = pd.Timestamp("2025-01-02"), pd.Timestamp("2026-10-16")

# (source file, column, display name, category, verdict)
CURATED = [
    ("sp500_beat_qqq_equity.csv", "QQQ_buy_hold", "QQQ 買進持有", "基準", "基準"),
    ("sp500_beat_qqq_equity.csv", "SPY_buy_hold", "SPY 買進持有", "基準", "基準"),
    ("sp500_backtest_long_equity.csv", "equal_weight_members", "S&P 500 成員等權（存活者）", "基準", "不可投資，存活偏差"),
    ("sp500_backtest_long_equity.csv", "A_max_return_5pos_63d_stop12", "選股 A：double_10，5 隻，63 日，12% 停損", "選股（第六、七節）", "輸基準，比隨機差"),
    ("sp500_backtest_long_equity.csv", "B_robust_4pos_126cap_shortneg", "選股 B：double_10，4 隻，短期分數轉負出場", "選股（第六、七節）", "輸基準，虧損"),
    ("sp500_backtest_long_equity.csv", "C_5pos_63d_nostop", "選股 C：double_10，5 隻，63 日，無停損", "選股（第六、七節）", "輸基準"),
    ("sp500_beat_qqq_equity.csv", "stocks_rev_short_m10_5pos_63d+regime", "選股：短期 -10 反轉，5 隻 + QQQ 200 日線濾網", "選股 + 濾網（9.2）", "輸基準"),
    ("sp500_beat_qqq_equity.csv", "stocks_double10_5pos_63d_stop12+regime", "選股 A + QQQ 200 日線濾網", "選股 + 濾網（9.2）", "輸基準"),
    ("sp500_top5_monthly_equity.csv", "medium_top5", "每月換成中期 63 日動能前 5 隻", "選股（第 15 節）", "輸基準，不高於隨機"),
    ("sp500_top5_monthly_equity.csv", "long_top5", "每月換成長期 252 日動能前 5 隻", "選股（第 15 節）", "同上"),
    ("sp500_top5_monthly_equity.csv", "12_1_top5", "每月換成 12-1 動能前 5 隻", "選股（第 15 節）", "同上"),
    ("sp500_top5_monthly_equity.csv", "12_1_top20", "每月換成 12-1 動能前 20 隻", "選股（第 15 節）", "追平 SPY"),
    ("sp500_top5_monthly_equity.csv", "short_top5_r200", "每月短期 21 日前 5 隻 + QQQ 200 日線濾網", "選股 + 濾網（15.6）", "勝 SPY，優勢全在 2000 年代"),
    ("sp500_top5_monthly_equity.csv", "medium_top5_r200", "每月中期 63 日前 5 隻 + QQQ 200 日線濾網", "選股 + 濾網（15.6）", "同上，最佳"),
    ("sp500_top5_monthly_equity.csv", "long_top5_r200", "每月長期 252 日前 5 隻 + QQQ 200 日線濾網", "選股 + 濾網（15.6）", "同上"),
    ("sp500_top5_monthly_equity.csv", "12_1_top5_r200", "每月 12-1 前 5 隻 + QQQ 200 日線濾網", "選股 + 濾網（15.6）", "同上，2020 後落後"),
    ("sp500_top5_monthly_equity.csv", "composite_top5_r200", "每月三階段合計前 5 隻 + QQQ 200 日線濾網", "選股 + 濾網（15.6）", "輸基準"),
    ("sp500_top5_monthly_equity.csv", "equal_weight_members_r200", "成員等權（每月）+ QQQ 200 日線濾網", "選股 + 濾網（15.6）", "濾網使等權變差"),
    ("sp500_beat_qqq_equity.csv", "qqq_sma_200_monthly", "QQQ > 200 日線，月底評估，否則國庫券", "全進全出擇時（9.2）", "回撤減半，多頭中段輸 QQQ"),
    ("sp500_beat_qqq_equity.csv", "qqq_sma_250", "QQQ > 250 日線", "全進全出擇時（9.2）", "同上"),
    ("sp500_beat_qqq_equity.csv", "qqq_sma_200_dd_15", "QQQ > 200 日線且距高點 15% 內", "全進全出擇時（9.2）", "回撤最小，報酬較低"),
    ("sp500_beat_qqq_equity.csv", "qqq_absmom_189", "QQQ 189 日絕對動能", "全進全出擇時（9.2）", "同上"),
    ("sp500_beat_qqq_equity.csv", "qqq_sma_200", "QQQ > 200 日線（每日）", "全進全出擇時（9.2）", "洗盤多"),
    ("sp500_beat_qqq_equity.csv", "dual_mom_252", "SPY / QQQ / 現金雙動能 252 日", "全進全出擇時（9.2）", "輸 SPY"),
    ("sp500_optimize_qqq_equity.csv", "voltarget_30_21", "波動率目標 30% / 21 日", "按比例調整（9.6）", "與 QQQ 打平，尾部改善"),
    ("sp500_optimize_qqq_equity.csv", "voltarget_25_21", "波動率目標 25% / 21 日", "按比例調整（9.6）", "同上"),
    ("sp500_optimize_qqq_equity.csv", "voltarget_20_63", "波動率目標 20% / 63 日", "按比例調整（9.6）", "回撤較小，報酬略低"),
    ("sp500_optimize_qqq_equity.csv", "partial_sma200m_50", "跌破 200 日線（月底）減至 50%", "按比例調整（9.6）", "介於兩者之間"),
    ("sp500_optimize_qqq_equity.csv", "vote_ensemble_monthly", "五規則投票（月底）", "按比例調整（9.6）", "不如 200 日線"),
    ("sp500_optimize_qqq_equity.csv", "asym_trend", "不對稱趨勢（慢出快進）", "按比例調整（9.6）", "不如 200 日線"),
    ("sp500_optimize_qqq_equity.csv", "breadth_gt_40", "S&P 500 寬度 > 40%", "按比例調整（9.6）", "失敗"),
    ("sp500_optimize_qqq_equity.csv", "gem_4asset_252", "QQQ / SPY / TLT / GLD 四資產動能 252 日", "按比例調整（9.6）", "失敗"),
    ("sp500_optimize_qqq2_equity.csv", "ratio_1.5_10_63", "波動擴張：10 日波動 > 1.5 × 63 日才減碼", "波動率目標改良（9.8）", "優勢集中在 2008 與 2020"),
    ("sp500_optimize_qqq2_equity.csv", "combo_ratio+vt_floor50", "組合：擴張規則 + 波動率目標（下限 0.5）", "波動率目標改良（9.8）", "最佳誠實版本"),
    ("sp500_optimize_qqq2_equity.csv", "floor_30_21_50", "波動率目標 30% / 21 日，權重下限 0.5", "波動率目標改良（9.8）", "略優於基準"),
    ("sp500_optimize_qqq2_equity.csv", "ewma_35_94", "EWMA 波動（λ 0.94）目標 35%", "波動率目標改良（9.8）", "不如基準"),
    ("sp500_optimize_qqq2_equity.csv", "down_30_21", "只看下跌波動 30% / 21 日", "波動率目標改良（9.8）", "不如基準"),
    ("sp500_optimize_qqq2_equity.csv", "vix_30", "VIX 隱含波動目標 30%", "波動率目標改良（9.8）", "失敗"),
]


def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1
    yearly = nav.resample("YE").last() / nav.resample("YE").first() - 1
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    return {"CAGR_pct": round(c * 100, 2), "ann_vol_pct": round(r.std() * math.sqrt(252) * 100, 1), "sharpe": round(r.mean() / r.std() * math.sqrt(252), 2),
            "max_dd_pct": round(dd.min() * 100, 1), "calmar": round(c / abs(dd.min()), 2), "worst_year_pct": round(yearly.min() * 100, 1),
            "best_year_pct": round(yearly.max() * 100, 1), "pct_years_positive": round((yearly > 0).mean() * 100, 0), "final_100k": round(100_000 * nav.iloc[-1] / nav.iloc[0])}


def load_all() -> dict[str, pd.Series]:
    curves: dict[str, pd.Series] = {}
    for f in ("sp500_backtest_long_equity.csv", "sp500_beat_qqq_equity.csv", "sp500_optimize_qqq_equity.csv", "sp500_optimize_qqq2_equity.csv", "sp500_top5_monthly_equity.csv"):
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        for c in df.columns:
            curves[f"{f}::{c}"] = df[c].dropna()
    return curves


def chart_yearly(yearly: pd.DataFrame, total: pd.DataFrame, path: str) -> None:
    """Heatmap of yearly returns (years x strategies) with a CAGR-to-today bar column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#fbe9e1", "#f7f7f7", "#dfe9f7", "#2a78d6"])
    cmap.set_bad("#d9d9d9")
    vals = yearly.T  # strategies x years
    fig = plt.figure(figsize=(18, 0.42 * len(vals) + 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[vals.shape[1], 4], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1], sharey=ax)
    lim = 60.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax.imshow(np.ma.masked_invalid(vals.to_numpy(dtype=float)), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(vals.shape[1])); ax.set_xticklabels([c.replace(" 至 10-16", "*") for c in vals.columns], rotation=0, fontsize=9)
    ax.set_yticks(range(vals.shape[0])); ax.set_yticklabels(vals.index, fontsize=10)
    ax.set_xticks(np.arange(-0.5, vals.shape[1], 1), minor=True); ax.set_yticks(np.arange(-0.5, vals.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals.iat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.6 * lim else "#222222")
    ax.set_title("各策略逐年報酬（%）：2000 年到 2026-10-16。標 * 的選股策略 2025 年起接今日名單；2026 為年初至 10-16", loc="left", fontsize=12)
    cag = total.loc[vals.index, "CAGR %"].to_numpy(dtype=float)
    colors = ["#2a78d6" if c >= 0 else "#eb6834" for c in cag]
    axb.barh(range(len(cag)), cag, color=colors, height=0.7)
    for i, c in enumerate(cag):
        axb.text(c + 0.3, i, f"{c:.1f}%", va="center", fontsize=8.5)
    axb.set_xlim(0, max(cag) * 1.35)
    axb.set_title("至今 CAGR", loc="left", fontsize=10)
    axb.tick_params(axis="y", left=False, labelleft=False); axb.set_xticks([])
    for spine in axb.spines.values():
        spine.set_visible(False)
    axb.invert_yaxis() if False else None
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:,.0f}" if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer() and abs(v) >= 1000) else str(v)) for v in r.tolist()) + " |")
    return "\n".join(out)


def main() -> int:
    curves = load_all()
    # ---- 2000-03 .. 2024-12 table for every curve, plus MC on the common draws
    qqq_full = curves["sp500_beat_qqq_equity.csv::QQQ_buy_hold"]
    mc_idx = qqq_full.index
    rng = np.random.default_rng(42)
    st, en, yr = MC.draw_windows(mc_idx, 5000, 1, 20, rng)
    gq, cq, _ = MC.window_stats(qqq_full.reindex(mc_idx).to_numpy(), st, en, yr)
    rows = []
    for key, s in curves.items():
        src, col = key.split("::")
        w = s.loc[COMMON_START:COMMON_END]
        if len(w) < 200 or w.index[0] > COMMON_START + pd.Timedelta(days=45) or w.index[-1] < COMMON_END - pd.Timedelta(days=10):
            continue
        m = metrics(w / w.iloc[0])
        row = {"source": src, "strategy": col, **m}
        if s.index[-1] >= mc_idx[-1] - pd.Timedelta(days=10):  # curve reaches 2026-10: paired MC
            nav = s.reindex(mc_idx).ffill().to_numpy()
            g, cg, _ = MC.window_stats(nav, st, en, yr)
            row.update({"mc_win_vs_QQQ_pct": round(float((g > gq).mean() * 100), 1), "mc_median_CAGR_pct": round(float(np.median(cg)) * 100, 2),
                        "mc_p10_CAGR_pct": round(float(np.percentile(cg, 10)) * 100, 2), "mc_worst_CAGR_pct": round(float(cg.min()) * 100, 1),
                        "mc_prob_loss_pct": round(float((g < 1).mean() * 100), 1)})
            sh = s.loc[SHORT_START:SHORT_END]
            row["CAGR_2025_2026_pct"] = metrics(sh / sh.iloc[0])["CAGR_pct"] if len(sh) > 100 else np.nan
        rows.append(row)
    full = pd.DataFrame(rows).sort_values("CAGR_pct", ascending=False)
    full.to_csv("strategy_table_2000_2024.csv", index=False, encoding="utf-8-sig")

    # ---- curated rows
    cur = []
    for src, col, name, cat, verdict in CURATED:
        r = full[(full.source == src) & (full.strategy == col)]
        if r.empty:
            continue
        r = r.iloc[0].to_dict()
        r.update(名稱=name, 類別=cat, 評價=verdict)
        cur.append(r)
    cur = pd.DataFrame(cur)
    # search winners from the grid (no saved curve)
    grid = pd.read_csv("sp500_strategy_search_grid.csv")
    extra = []
    for _, g in grid.sort_values("cagr_pct", ascending=False).head(2).iterrows():
        extra.append({"名稱": f"搜尋第一名：{g.entry}，{int(g.max_pos)} 隻，{int(g.hold)} 日（2000-01 起）", "類別": "選股搜尋（7.6）", "評價": "起始日運氣，平移後中位數 6% 到 12%",
                      "CAGR_pct": g.cagr_pct, "max_dd_pct": g.max_dd_pct, "sharpe": g.sharpe, "worst_year_pct": g.worst_year_pct, "pct_years_positive": g.pct_years_positive})
    cur = pd.concat([cur, pd.DataFrame(extra)], ignore_index=True)

    # ---- 2025-2026 table
    sh = pd.read_csv("benchmark_compare_short_equity.csv", index_col=0, parse_dates=True)
    short_rows = []
    names_short = {"A 最大報酬版": ("選股 A：double_10，5 隻，63 日，12% 停損", "選股（第六節）"), "B 穩健版": ("選股 B：double_10，4 隻，短期分數轉負出場", "選股（第六節）"),
                   "成份股等權": ("S&P 500 成份股等權（今日名單）", "基準"), "SPY": ("SPY 買進持有", "基準"), "QQQ": ("QQQ 買進持有", "基準")}
    for c in sh.columns:
        m = metrics(sh[c] / sh[c].iloc[0]); n, cat = names_short.get(c, (c, "")); short_rows.append({"名稱": n, "類別": cat, **m})
    for key, s in curves.items():
        src, col = key.split("::")
        if col in ("qqq_sma_200_monthly", "qqq_sma_200_dd_15", "voltarget_30_21", "combo_ratio+vt_floor50", "ratio_1.5_10_63") and s.index[-1] >= SHORT_END - pd.Timedelta(days=5):
            w = s.loc[SHORT_START:SHORT_END]; m = metrics(w / w.iloc[0])
            nm = {k[1]: k[2] for k in CURATED}.get(col, col); short_rows.append({"名稱": nm, "類別": "QQQ 擇時 / 波動率目標", **m})
    if os.path.exists("sp500_top5_monthly_equity_2025_2026.csv"):
        t5 = pd.read_csv("sp500_top5_monthly_equity_2025_2026.csv", index_col=0, parse_dates=True)
        names5 = {k[1]: k[2] for k in CURATED}
        for col in ("medium_top5_r200", "long_top5_r200", "short_top5_r200", "12_1_top5_r200", "long_top5", "medium_top5"):
            if col in t5:
                w = t5[col].dropna(); m = metrics(w / w.iloc[0])
                short_rows.append({"名稱": names5.get(col, col) + "（今日名單，2025-02-03 起）", "類別": "選股（第 15 節）", **m})
    short = pd.DataFrame(short_rows).drop_duplicates("名稱").sort_values("CAGR_pct", ascending=False)
    short.to_csv("strategy_table_2025_2026.csv", index=False, encoding="utf-8-sig")

    # ---- 12.5 yearly returns 2000 .. today and total return
    YEARLY = [  # (long source, long column, short source or None, short column, display name)
        ("sp500_beat_qqq_equity.csv", "QQQ_buy_hold", None, None, "QQQ"),
        ("sp500_beat_qqq_equity.csv", "SPY_buy_hold", None, None, "SPY"),
        ("sp500_backtest_long_equity.csv", "equal_weight_members", "benchmark_compare_short_equity.csv", "成份股等權", "成員等權*"),
        ("sp500_backtest_long_equity.csv", "A_max_return_5pos_63d_stop12", "benchmark_compare_short_equity.csv", "A 最大報酬版", "選股 A*"),
        ("sp500_backtest_long_equity.csv", "B_robust_4pos_126cap_shortneg", "benchmark_compare_short_equity.csv", "B 穩健版", "選股 B*"),
        ("sp500_backtest_long_equity.csv", "C_5pos_63d_nostop", None, None, "選股 C"),
        ("sp500_top5_monthly_equity.csv", "medium_top5", "sp500_top5_monthly_equity_2025_2026.csv", "medium_top5", "每月中期前 5*"),
        ("sp500_top5_monthly_equity.csv", "medium_top5_r200", "sp500_top5_monthly_equity_2025_2026.csv", "medium_top5_r200", "每月中期前 5 + 濾網*"),
        ("sp500_top5_monthly_equity.csv", "short_top5_r200", "sp500_top5_monthly_equity_2025_2026.csv", "short_top5_r200", "每月短期前 5 + 濾網*"),
        ("sp500_top5_monthly_equity.csv", "long_top5_r200", "sp500_top5_monthly_equity_2025_2026.csv", "long_top5_r200", "每月長期前 5 + 濾網*"),
        ("sp500_top5_monthly_equity.csv", "12_1_top20", "sp500_top5_monthly_equity_2025_2026.csv", "12_1_top20", "每月 12-1 前 20*"),
        ("sp500_beat_qqq_equity.csv", "qqq_sma_200_monthly", None, None, "QQQ 200 日線月底"),
        ("sp500_beat_qqq_equity.csv", "qqq_sma_200_dd_15", None, None, "QQQ 200 日線 + 15% 回撤"),
        ("sp500_optimize_qqq_equity.csv", "voltarget_30_21", None, None, "波動率目標 30/21"),
        ("sp500_optimize_qqq_equity.csv", "voltarget_20_63", None, None, "波動率目標 20/63"),
        ("sp500_optimize_qqq2_equity.csv", "ratio_1.5_10_63", None, None, "波動擴張"),
        ("sp500_optimize_qqq2_equity.csv", "combo_ratio+vt_floor50", None, None, "組合規則"),
    ]
    stitched: dict[str, pd.Series] = {}
    starts: dict[str, str] = {}
    for src, col, ssrc, scol, name in YEARLY:
        long = pd.read_csv(src, index_col=0, parse_dates=True)[col].dropna()
        long = long / long.iloc[0]
        if ssrc and os.path.exists(ssrc):
            sh2 = pd.read_csv(ssrc, index_col=0, parse_dates=True)[scol].dropna()
            sh2 = sh2 / sh2.iloc[0] * long.iloc[-1]
            long = pd.concat([long, sh2[sh2.index > long.index[-1]]])
        stitched[name] = long
        starts[name] = long.index[0].date().isoformat()
    ydf = pd.DataFrame(stitched)
    ye = ydf.resample("YE").last()
    yfirst = ydf.resample("YE").first()
    # a year's return is last close / previous year's last close; first year uses its own first close
    prev = ye.shift(1)
    yearly = (ye / prev.where(prev.notna(), yfirst) - 1) * 100
    yearly.index = [str(y) if y < 2026 else "2026 至 10-16" for y in yearly.index.year]
    total = pd.DataFrame({"起始日": pd.Series(starts), "結束日": pd.Series({k: v.index[-1].date().isoformat() for k, v in stitched.items()}),
                          "總報酬 %": pd.Series({k: round((v.iloc[-1] - 1) * 100, 0) for k, v in stitched.items()}),
                          "10 萬終值": pd.Series({k: round(100_000 * v.iloc[-1]) for k, v in stitched.items()}),
                          "CAGR %": pd.Series({k: round(((v.iloc[-1]) ** (365.25 / (v.index[-1] - v.index[0]).days) - 1) * 100, 1) for k, v in stitched.items()})})
    yt = yearly.round(1).T
    yt.index.name = "策略"
    out_y = pd.concat([yt, total], axis=1)
    out_y.to_csv("strategy_table_yearly.csv", encoding="utf-8-sig")
    yearly_md = md(yearly.round(1).reset_index().rename(columns={"index": "年"}))
    chart_yearly(yearly, total, "docs/strategy_yearly.png")
    total_md = md(total.reset_index().rename(columns={"index": "策略"}))

    # ---- markdown page
    c1 = ["名稱", "類別", "CAGR_pct", "ann_vol_pct", "sharpe", "max_dd_pct", "calmar", "worst_year_pct", "pct_years_positive", "final_100k", "評價"]
    t1 = cur[c1].rename(columns={"CAGR_pct": "CAGR %", "ann_vol_pct": "年化波動 %", "sharpe": "Sharpe", "max_dd_pct": "最大回撤 %", "calmar": "Calmar", "worst_year_pct": "最差年 %", "pct_years_positive": "正報酬年 %", "final_100k": "10 萬終值"})
    c2 = ["名稱", "mc_win_vs_QQQ_pct", "mc_median_CAGR_pct", "mc_p10_CAGR_pct", "mc_worst_CAGR_pct", "mc_prob_loss_pct", "CAGR_2025_2026_pct"]
    t2 = cur.dropna(subset=["mc_win_vs_QQQ_pct"])[c2].rename(columns={"mc_win_vs_QQQ_pct": "勝 QQQ %", "mc_median_CAGR_pct": "CAGR 中位數 %", "mc_p10_CAGR_pct": "CAGR p10 %", "mc_worst_CAGR_pct": "最差 CAGR %", "mc_prob_loss_pct": "虧損機率 %", "CAGR_2025_2026_pct": "2025-26 CAGR %"})
    c3 = ["名稱", "類別", "CAGR_pct", "ann_vol_pct", "sharpe", "max_dd_pct", "worst_year_pct", "final_100k"]
    t3 = short[c3].rename(columns={"CAGR_pct": "CAGR %", "ann_vol_pct": "年化波動 %", "sharpe": "Sharpe", "max_dd_pct": "最大回撤 %", "worst_year_pct": "最差年 %", "final_100k": "10 萬終值"})
    f15 = cur[cur["類別"] == "選股 + 濾網（15.6）"].set_index("strategy")
    f15_rng = f"{f15.loc[['short_top5_r200','medium_top5_r200','long_top5_r200','12_1_top5_r200'], 'CAGR_pct'].min():.1f}% 到 {f15.loc[['short_top5_r200','medium_top5_r200','long_top5_r200','12_1_top5_r200'], 'CAGR_pct'].max():.1f}%"
    f15_dd = f"{f15.loc[['short_top5_r200','medium_top5_r200','long_top5_r200','12_1_top5_r200'], 'max_dd_pct'].max():.0f}% 到 {f15.loc[['short_top5_r200','medium_top5_r200','long_top5_r200','12_1_top5_r200'], 'max_dd_pct'].min():.0f}%"
    page = f"""## 12. 全部策略總表

所有階段的策略放在同一個比較表。共同期間 2000-03-01 至 2024-12-31（選股策略從 2000-01 起算，這裡統一截到 2000-03 以與 QQQ 擇時規則對齊）。指標定義相同：CAGR、年化波動、Sharpe（無風險利率 0）、最大回撤、Calmar、最差年份、正報酬年份比例、10 萬本金終值。蒙地卡羅為同一組 5,000 次（進場日、1 到 20 年持有）抽樣，曲線延伸到 2026-10-16，只有有完整曲線的規則才有此欄。腳本 `sp500_strategy_table.py`，全部 {len(full)} 條曲線在 `strategy_table_2000_2024.csv`。

### 12.1 2000-03 至 2024-12 共同期間

{md(t1)}

搜尋第一名的兩列來自 646 組網格（2000-01 起算、時點宇宙），沒有保存淨值曲線，其餘欄位空白；它們的 CAGR 在起始日平移後中位數只有 6% 到 12%。

### 12.2 同一組 5,000 次隨機進場（1 到 20 年持有，2000-03 至 2026-10）

{md(t2)}

### 12.3 2025-01-02 至 2026-10-16

{md(t3)}

選股 A 與 B 用今日成份股名單，這段期間的 62% 到 83% CAGR 全部來自 2026 年的記憶體與儲存股行情，第六節已證明隨機剔除 36 隻股票就會讓它在 17% 到 102% 之間跳動。第 15 節的每月前 5 隻同樣用今日名單，且從 2025-02-03 起算（需 252 日歷史），期間比其他列短一個月，數字不能直接相減。

### 12.4 怎麼讀這張表

- **報酬**：2000 至 2024 年最高的是成員等權（12.4%），但它是存活者等權，不可投資。可投資的最高是「每月動能前 5 隻 + QQQ 200 日線濾網」{f15_rng}（第 15.6 節，四種定義），其次組合規則 9.8%、波動率目標 10.0%、200 日線月底評估 9.4%，QQQ 與 SPY 各 7.2% 到 7.9%。不加濾網的選股策略全部在 8% 以下。
- **加濾網的選股要打折看**：它的優勢全來自 2000 至 2002 與 2008 年持現金（2000 年代 CAGR 8% 到 14% 對 SPY -0.9%），2010 年以後多數年份輸 SPY、全部輸 QQQ；最大回撤仍 {f15_dd}，與 SPY 相當；對濾網長度敏感（250 日時短期定義掉到 4%）；曲線只到 2024-12，沒有同一組蒙地卡羅可比。第 13、14 節已證明濾網本身在樣本外不加分。
- **回撤**：QQQ -83%、SPY -55%、選股 -48% 到 -82%、全進全出擇時 -26% 到 -41%、波動率目標 -47% 到 -71%。回撤最小的是「200 日線且距高點 15% 內」（-26%）與「波動率目標 20% / 63 日 + 200 日線」（-25%）。
- **任意時點進場**：只有波動率目標家族的 CAGR 中位數不低於 QQQ（14.5% 到 15.3% 對 14.5%），同時第 10 百分位與最差情境改善。全進全出擇時的中位數低 3 到 6 點，選股低 8 點以上。
- **全期 CAGR 與蒙地卡羅中位數的落差**：全期從 2000 年 3 月泡沫頂點起算，是 QQQ 最差的進場日，因此全期數字對擇時有利；蒙地卡羅把進場日平均掉，才是「隨時進場」的期望值。
- 第 9.8 節已說明波動擴張規則與組合規則的優勢集中在 2008 年 10 月與 2020 年 3 月兩個事件。

### 12.5 逐年報酬與至今總報酬（2000 年起至 2026-10-16）

每年報酬 = 該年最後收盤 ÷ 前一年最後收盤 − 1（首年用該曲線的第一個收盤）。標 * 的選股策略是兩段接起來的：2000 至 2024 年用時點成份股宇宙，2025 年起接上今日名單的曲線（選股 A、B 與成員等權從 2025-01-02，每月前 N 隻從 2025-02-03，接點前視為持平）；今日名單有存活偏差，2025 年以後的數字偏樂觀，與前面各年不是同一個宇宙。QQQ 擇時與波動率目標規則從 2000-03-01 起算（QQQ 資料 1999 年 3 月起，需 252 日暖身），其餘從 2000-01-03。完整數字在 `strategy_table_yearly.csv`。

![逐年報酬熱圖](./strategy_yearly.png)

{yearly_md}

{total_md}

熱圖怎麼讀：每格是該策略該年的報酬，藍色為正、橘色為負，深淺按絕對值；灰格為沒有資料。右側的 CAGR 是各曲線從自己的起始日到 2026-10-16。加濾網的選股列在 2001、2002、2008 年接近零（持國庫券），這三格就是它 25 年領先的全部來源；2025 到 2026 年的深藍格是今日名單的存活偏差。
"""
    nav = "[回索引](../README.md) · [完整報告](./ANALYSIS.md) · [上一節](./11-reproduce.md) · [下一節](./13-walk-forward.md)"
    body = page.split("\n", 2)[2].replace("\n### 12.", "\n## 12.")
    open("docs/12-strategy-table.md", "w").write(f"# 12. 全部策略總表\n\n{nav}\n\n> 資料：S&P 500 成份股，Yahoo Finance 調整後收盤價。短期分析期間 2024-01-02 至 2026-10-16；長期回測 2000-01-03 至 2024-12-31。\n\n" + body + "\n\n---\n\n" + nav + "\n")
    open("/tmp/strategy_table_section.md", "w").write(page)
    pd.set_option("display.width", 300)
    print(t1.to_string(index=False)); print(); print(t2.to_string(index=False)); print(); print(t3.to_string(index=False))
    print(f"\nwrote strategy_table_2000_2024.csv ({len(full)} rows), strategy_table_2025_2026.csv, docs/12-strategy-table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
