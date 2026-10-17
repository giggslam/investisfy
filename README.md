# investisfy

S&P 500 動能研究：從事後股價反推動能、校準分數的預測力、從贏家反推買入點、以 3 到 5 隻持倉回測，最後用 2000 至 2024 年的時點成份股宇宙驗證。所有結果都可用本 repo 的腳本重算。

**一句話結論**：2024 至 2026 年看起來有效的動能策略（CAGR 62% 到 83%），放到 2000 至 2024 年的時點宇宙只有 -0.6% 到 5.7%，輸給 SPY，且比隨機選股差。對 24 種進場規則做 646 組搜尋，沒有任何 3 到 5 隻持倉的策略在 25 年達到 30% CAGR；穩健上限約 12% 到 14%，與等權基準相同。S&P 500 內部跨 25 年穩定存在的是極端下跌後的反轉，不是動能延續。長期而言被動持有 SPY 或 QQQ（25 年各 7.7% CAGR）優於這套策略。允許空倉的全進全出擇時規則也無法在多數進場時點勝過 QQQ；按比例減碼的波動率目標能與 QQQ 打平並把最差情境從 -63% 改善到 -42%，但不能穩定跑贏。

## 文件索引

| # | 文件 | 內容 | 關鍵數字 |
|---|---|---|---|
| 1 | [方法](docs/01-method.md) | 三階段動能定義、橫斷面十分位評分、事後反推的兩種用法 | 短期 21 日、中期 63 日、長期 252 日；+1 到 +10、-1 到 -10 |
| 2 | [當前快照](docs/02-snapshot.md) | 2026-10-16 的 503 隻評分、前後十名、產業表 | 長期上升 332 隻；能源 21 隻中 14 隻三階段全升 |
| 3 | [2024 至 2026 的接續性](docs/03-persistence-2024-2026.md) | 去年排名對今年的預測力、大反轉與大翻身名單 | Spearman 0.094；29 隻大反轉、33 隻大翻身 |
| 4 | [事後校準](docs/04-calibration.md) | 每日即時分數對之後 21 / 63 日報酬的校準表、資訊係數、分數持續性 | 只有 +10 有超額（21 日 +2.4% 到 +3.5%）；-10 平均反彈 |
| 5 | [從贏家反推買入點](docs/05-entry-points.md) | 1,458 次贏家行情的低點剖面、15 種即時訊號排名、目前候選 | 下跌後急拉訊號：低點後 9 天觸發、63 日超額 +7.0% |
| 6 | [組合回測：3 到 5 隻持倉](docs/06-portfolio-backtest.md) | 960 組網格、壓力測試、入選策略 | double_10 5 隻 63 日：CAGR 83%、回撤 -40%；但隨機剔除 36 隻股票可使 CAGR 在 17% 到 99% 間跳動 |
| 7 | [2000 至 2024 年長期回測](docs/07-backtest-2000-2024.md) | 時點成份股宇宙、逐年與分期報酬、隨機基準、存活偏差量化、646 組策略搜尋 | CAGR 3.4% / -0.6% / 5.7% vs SPY 7.7%；今日名單多出 9 到 20 個百分點；搜尋中 0 組達 30%，最高 22.4% 且為起始日運氣 |
| 8 | [與 SPY、QQQ 的比較](docs/08-benchmark-comparison.md) | 兩個期間的策略對 SPY、QQQ 淨值曲線、指標、逐年、分期、滾動三年勝率 | 2000-2024：策略 3.4% / -0.6% / 5.7% 對 SPY 7.7%、QQQ 7.7%，全輸；2025-2026：82.6% 對 17.4% / 20.9%，領先全來自 2026 |
| 9 | [允許空倉：任意時點進場能否跑贏 QQQ](docs/09-beat-qqq.md) | 27 組全進全出擇時規則、進場視窗勝率、10 萬本金範例、5,000 次隨機進場蒙地卡羅、29 組第二代按比例調整規則、39 組第三輪波動率目標改良 | 全進全出規則在 82% 到 88% 的情境輸給 QQQ；波動率目標與其組合版能打平或略勝（CAGR 中位數 14.8% 到 15.3% 對 14.5%、最差 -41% 對 -63%），但優勢全來自崩盤視窗，2009 年後進場仍有約九成小輸；再調參數只是擬合 2008 與 2020 兩個事件 |
| 10 | [結論與限制](docs/10-conclusions.md) | 全部結論與方法限制 | |
| 11 | [重跑方式與檔案清單](docs/11-reproduce.md) | 指令與每個輸出檔的說明 | |
| | [完整報告（單一檔）](docs/ANALYSIS.md) | 第 1 到 11 節合併版 | |

## 腳本

| 腳本 | 功能 | 主要輸出 |
|---|---|---|
| `sp500_momentum.py` | 成份股、股價下載與快取、三階段動能評分；`--asof` 重建任一天的快照；`--mode history` 每日即時分數、未來報酬、校準表、資訊係數 | `sp500_momentum.csv`、`sp500_momentum_calibration.csv`、`sp500_momentum_ic.csv` |
| `sp500_entry_points.py` | zigzag 找出贏家行情，事後對照 15 種即時進場訊號 | `sp500_entry_runs.csv`、`sp500_entry_signals.csv`、`sp500_entry_trough_profile.csv`、`sp500_entry_candidates.csv` |
| `sp500_backtest.py` | 3 到 5 隻持倉的組合回測引擎與網格搜尋 | `sp500_backtest_grid.csv`、`sp500_backtest_maxreturn_*`、`sp500_backtest_robust_*` |
| `sp500_backtest_long.py` | 2000 至 2024 年時點成份股宇宙回測、隨機基準、存活偏差對照 | `sp500_backtest_long_summary.csv`、`_yearly.csv`、`_equity.csv`、`_trades_A.csv`、`_trades_B.csv` |
| `sp500_strategy_search.py` | 2000 至 2024 年 24 種進場規則的策略搜尋與隨機多重測試對照 | `sp500_strategy_search_calibration.csv`、`_grid.csv`、`_random.csv`、`_best_trades.csv` |
| `sp500_benchmark_compare.py` | 策略與 SPY、QQQ 的比較表與淨值曲線圖 | `benchmark_compare_*.csv`、`docs/benchmark_comparison.png` |
| `sp500_beat_qqq.py` | 允許空倉的擇時規則與加濾網選股對 QQQ，任意進場時點檢驗，10 萬本金範例 | `sp500_beat_qqq_*.csv`、`docs/beat_qqq.png` |
| `sp500_random_entry_mc.py` | 5,000 次隨機進場、持有 1 到 20 年的蒙地卡羅，策略對 QQQ 配對比較 | `sp500_random_entry_mc_*.csv`、`docs/random_entry_mc.png` |
| `sp500_optimize_qqq.py` | 第二代可空倉規則（波動率目標、部分減碼、投票、不對稱趨勢、寬度、四資產動能、混合）對 QQQ | `sp500_optimize_qqq_*.csv`、`docs/optimize_qqq.png` |
| `sp500_optimize_qqq2.py` | 第三輪：波動率目標的改良（EWMA、下跌波動、VIX、波動擴張、下限、每週、回補、門檻帶）與組合 | `sp500_optimize_qqq2_*.csv`、`docs/optimize_qqq2.png` |

## 快速開始

```bash
pip install -r requirements.txt
python sp500_momentum.py --mode both        # 快照 + 每日歷史 + 校準 + IC（建立 prices_cache.csv）
python sp500_entry_points.py                # 贏家行情與進場訊號
python sp500_backtest.py --grid             # 3 到 5 隻持倉網格回測
python sp500_backtest_long.py --build       # 2000-2024 回測（首次下載歷史名單與 26 年股價，約 10 分鐘）
python sp500_strategy_search.py --random-runs 200   # 2000-2024 策略搜尋（約 45 分鐘）
python sp500_benchmark_compare.py           # 與 SPY、QQQ 比較
python sp500_beat_qqq.py                    # 允許空倉的擇時策略對 QQQ
python sp500_random_entry_mc.py             # 5,000 次隨機進場蒙地卡羅
python sp500_optimize_qqq.py                # 第二代規則對 QQQ
python sp500_optimize_qqq2.py               # 第三輪波動率目標改良
```

快取檔（`prices_cache.csv`、`prices_long_cache.csv`、`membership_long.pkl`、`spy_long_cache.csv`、`spy_full_cache.csv`、`qqq_full_cache.csv`、`tlt_full_cache.csv`、`gld_full_cache.csv`、`irx_cache.csv`、`vix_cache.csv`）與 28 MB 的 `sp500_momentum_history.csv` 不進 git，重跑即可產生。

## 資料來源

- 成份股名單：[datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies)（現行），[fja05680/sp500](https://github.com/fja05680/sp500)（1996 年起每日歷史成員）。
- 股價：Yahoo Finance 調整後收盤價，透過 `yfinance`。已下市的歷史成份股沒有價格資料，長期回測因此仍有殘餘的存活偏差。

## 評分規則摘要

每天在全體成份股中做橫斷面十分位排名：上升的股票依速度給 +1（最慢）到 +10（最快），下降的股票依跌速給 -1（最慢）到 -10（最快），持平為 0。分數是相對排名，不是絕對報酬門檻。

本 repo 是研究紀錄，不是投資建議。
