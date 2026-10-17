# 11. 重跑方式與檔案清單

[回索引](../README.md) · [完整報告](./ANALYSIS.md) · [上一節](./10-conclusions.md) · [下一節](./12-strategy-table.md)

> 資料：S&P 500 成份股，Yahoo Finance 調整後收盤價。短期分析期間 2024-01-02 至 2026-10-16；長期回測 2000-01-03 至 2024-12-31。

```bash
pip install -r requirements.txt
python sp500_momentum.py --mode both     # 快照 + 每日歷史 + 校準表 + IC（會建立 prices_cache.csv）
python sp500_momentum.py --asof 2026-04-08   # 任一天的快照，只用該日以前的價格
python sp500_entry_points.py             # 贏家行情、訊號排名、低點剖面、目前候選
python sp500_backtest.py --grid          # 組合回測網格（960 組）
python sp500_backtest.py --run double_10 5 63 none 0.12 none ret10 --out sp500_backtest_maxreturn
python sp500_backtest.py --run double_10 4 126 none none short_neg ret10 --out sp500_backtest_robust
python sp500_backtest_long.py --build    # 2000-2024 時點宇宙回測（首次會下載歷史名單與 26 年股價）
python sp500_strategy_search.py --random-runs 200   # 2000-2024 策略搜尋與隨機對照（約 45 分鐘）
python sp500_benchmark_compare.py         # 與 SPY、QQQ 比較，輸出表格與圖
python sp500_beat_qqq.py                  # 允許空倉的擇時策略對 QQQ（含選股加濾網，約 10 分鐘；--no-stocks 只跑 ETF 規則）
python sp500_beat_qqq.py --chart-only qqq_sma_200_monthly qqq_sma_200_dd_15 qqq_sma_250
python sp500_random_entry_mc.py --n 5000 --min-years 1 --max-years 20   # 隨機進場蒙地卡羅
python sp500_optimize_qqq.py              # 第二代規則（波動率目標、部分減碼、投票、寬度、四資產、混合）
python sp500_optimize_qqq2.py             # 第三輪：波動率目標的改良與組合
python sp500_strategy_table.py            # 全部策略總表
python sp500_walk_forward.py              # 走動式樣本外測試（約 5 秒）
python sp500_cross_asset.py               # 跨資產驗證（首次下載 18 檔 ETF，約 30 秒）
python sp500_top5_monthly.py              # 每月換成動能最高 N 隻，含 QQQ 200 日線濾網版與 1,600 組隨機對照（約 9 分鐘）
```

| 檔案 | 內容 |
|---|---|
| `sp500_momentum.py` | 成份股、股價下載與快取、三階段動能評分、歷史模式、校準與 IC |
| `sp500_entry_points.py` | zigzag 贏家行情、即時訊號事件研究、候選名單 |
| `sp500_momentum.csv` | 當前快照，503 隻 |
| `sp500_momentum_2026.csv` | 2026 YTD 快照（長期 = YTD） |
| `sp500_momentum_calibration.csv` | 分數對未來 21 / 63 日報酬的校準表 |
| `sp500_momentum_ic.csv`、`sp500_momentum_ic_daily.csv` | 資訊係數摘要與每日序列 |
| `sp500_momentum_history.csv` | 每日每股的即時分數與未來報酬（339,235 行，未提交，重跑即得） |
| `sp500_entry_runs.csv` | 1,458 次贏家行情與各訊號的首次觸發 |
| `sp500_entry_signals.csv` | 訊號排名表 |
| `sp500_entry_trough_profile.csv` | 低點後 0 / 5 / 10 / 20 天的剖面 |
| `sp500_entry_candidates.csv` | 最近 10 個交易日的候選 |
| `sp500_backtest.py` | 3 到 5 隻持倉的組合回測引擎與網格搜尋 |
| `sp500_backtest_grid.csv` | 960 組回測結果 |
| `sp500_backtest_maxreturn_best_trades.csv`、`_equity.csv` | 最大報酬策略的交易紀錄與淨值曲線 |
| `sp500_backtest_robust_best_trades.csv`、`_equity.csv` | 穩健策略的交易紀錄與淨值曲線 |
| `sp500_backtest_long.py` | 2000-2024 時點成份股宇宙的長期回測 |
| `sp500_backtest_long_summary.csv`、`_yearly.csv`、`_equity.csv`、`_trades_A.csv`、`_trades_B.csv` | 長期回測的摘要、逐年報酬、淨值曲線與交易紀錄 |
| `sp500_strategy_search.py` | 2000-2024 時點宇宙的策略搜尋（24 種進場 × 持倉 × 持有期 × 排序 × 出場，含隨機對照） |
| `sp500_strategy_search_calibration.csv`、`_grid.csv`、`_random.csv`、`_best_trades.csv` | 25 年訊號校準、646 組網格、200 組隨機對照、網格第一名交易紀錄 |
| `sp500_benchmark_compare.py` | 策略與 SPY、QQQ 的比較表與淨值曲線圖 |
| `benchmark_compare_long_*.csv`、`benchmark_compare_short_*.csv`、`docs/benchmark_comparison.png` | 兩個期間的淨值、指標、逐年、分期與圖 |
| `sp500_beat_qqq.py` | 允許空倉的擇時規則與加濾網的選股策略對 QQQ，含任意進場時點檢驗 |
| `sp500_beat_qqq_summary.csv`、`_entry_windows.csv`、`_yearly.csv`、`_equity.csv`、`_examples_100k.csv`、`docs/beat_qqq.png` | 27 組策略的摘要、進場視窗統計、逐年、淨值、10 萬本金範例與圖 |
| `sp500_random_entry_mc.py` | 5,000 次隨機進場、持有 1 到 20 年的蒙地卡羅 |
| `sp500_random_entry_mc_summary.csv`、`_by_hold.csv`、`_by_entry_year.csv`、`_draws.csv`、`docs/random_entry_mc.png` | 整體、依持有年數、依進場年份的統計，每筆抽樣，與圖 |
| `sp500_optimize_qqq.py` | 第二代可空倉規則：波動率目標、部分減碼、投票、不對稱趨勢、寬度濾網、四資產動能、混合 |
| `sp500_optimize_qqq_summary.csv`、`_entry_windows.csv`、`_mc_summary.csv`、`_mc_by_hold.csv`、`_equity.csv`、`docs/optimize_qqq.png` | 29 組規則的全期指標、進場視窗、配對蒙地卡羅與圖 |
| `sp500_optimize_qqq2.py` | 第三輪：EWMA、下跌波動、VIX、波動擴張、權重下限、每週、回補、門檻帶、組合 |
| `sp500_optimize_qqq2_summary.csv`、`_equity.csv`、`docs/optimize_qqq2.png` | 39 組改良規則的指標、配對蒙地卡羅、2009 後與無崩盤切面，與圖 |
| `sp500_strategy_table.py` | 把所有階段的策略放到同一比較表 |
| `strategy_table_2000_2024.csv`、`strategy_table_2025_2026.csv`、`docs/12-strategy-table.md` | 103 條曲線的共同期間指標與蒙地卡羅、2025-26 表、策略總表頁 |
| `sp500_walk_forward.py` | 走動式樣本外測試：五個規則家族、118 組參數、三段展開式與逐年重選 |
| `sp500_walk_forward_params.csv`、`_equity.csv`、`_summary.csv`、`docs/walk_forward.png` | 每次重選的參數、拼接的樣本外淨值、指標與折價、配對蒙地卡羅，與圖 |
| `sp500_cross_asset.py` | 跨資產驗證：凍結參數的 8 條規則套到 18 檔 ETF |
| `sp500_cross_asset_metrics.csv`、`_mc.csv`、`_summary.csv`、`_equity_post2009.csv`、`docs/cross_asset.png` | 資產 × 規則 × 期間指標、配對蒙地卡羅、彙總與符號檢定、2010 後淨值，與熱圖 |
| `sp500_top5_monthly.py` | 每月換成動能最高 N 隻：五種動能定義 × N = 5 / 10 / 20 / 50，每個組合另跑 QQQ 均線濾網版（`--sma`，預設 200；`--chart-only` 只重畫圖），隨機對照（有無濾網各 200 組），2025-26 今日名單 |
| `sp500_top5_monthly_summary.csv`、`_yearly.csv`、`_equity.csv`、`_equity_2025_2026.csv`、`_random.csv`、`docs/top5_monthly.png` | 指標與分期 CAGR、逐年、淨值、2025-26 淨值、1,600 組隨機組合（有無濾網），與圖 |
| `prices_long_cache.csv`、`membership_long.pkl`、`spy_long_cache.csv` | 26 年股價、歷史成員矩陣與 SPY 快取（未提交，`--build` 重建） |

---

[回索引](../README.md) · [完整報告](./ANALYSIS.md) · [上一節](./10-conclusions.md) · [下一節](./12-strategy-table.md)
