# investisfy

S&P 500 動能研究：從事後股價反推動能、校準分數的預測力、從贏家反推買入點、以 3 到 5 隻持倉回測，最後用 2000 至 2024 年的時點成份股宇宙驗證。所有結果都可用本 repo 的腳本重算。

**一句話結論**：2024 至 2026 年看起來有效的動能策略（CAGR 62% 到 83%），放到 2000 至 2024 年的時點宇宙只有 -0.6% 到 5.7%，輸給 SPY，且比隨機選股差。對 24 種進場規則做 646 組搜尋，沒有任何 3 到 5 隻持倉的策略在 25 年達到 30% CAGR；穩健上限約 12% 到 14%，與等權基準相同。S&P 500 內部跨 25 年穩定存在的是極端下跌後的反轉，不是動能延續。長期而言被動持有 SPY 或 QQQ（25 年各 7.7% CAGR）優於這套策略。允許空倉的全進全出擇時規則也無法在多數進場時點勝過 QQQ；按比例減碼的波動率目標能與 QQQ 打平並把最差情境從 -63% 改善到 -42%，但不能穩定跑贏。走動式樣本外測試與 18 檔 ETF 的跨資產驗證進一步顯示，波動率目標與均線擇時的優勢是 QQQ 單一標的加 2000 年起點的樣本內結果，樣本外只有波動擴張規則能與買進持有打平並略減回撤。最直接的「每月換成動能最高 5 隻」在 2000 至 2024 年也只有 3.6% 到 6.9%，不高於隨機選股；加 QQQ 200 日線濾網後升到 10% 到 11.5%、勝過 SPY，但優勢全來自避開兩次熊市，2010 年後仍輸指數。反推進榜前的技術與財報前兆，找到的只是動能本身，沒有先行指標。最終判定（第 18 節）：買進持有指數，想降低最深回撤就加一個簡單的趨勢或波動尖峰開關並接受多頭年少賺；選股與前兆不建議投入。要達到「年回報 30% 且 MDD 30% 以內」需要選股 IC 約 0.5 或擇時命中率約 85%，兩者合用最低也要 IC 0.15 配命中 70%，現有訊號差一個數量級；合理目標是以做得到的 IC 0.03 到 0.05 配 55% 到 60% 命中率，每月前 20 隻加均線開關，約 13% 到 14% CAGR、MDD −25% 到 −28%（第 19 節）。按這套流程正向稽核免費資料（第 20 節）：0 / 57 個特徵、0 / 29 條擇時規則通過事前門檻，唯一量到的 1 個月反轉訊號 IC 0.018 且 2020 年後失效，實際交易 8% 到 16% CAGR、MDD −30% 到 −78%，仍是持有指數最好。

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
| 12 | [全部策略總表](docs/12-strategy-table.md) | 所有階段的策略在同一共同期間的 CAGR、波動、Sharpe、最大回撤、Calmar、最差年、10 萬終值，同一組蒙地卡羅，2025-26 表，以及 2000 年起逐年報酬與至今總報酬（表與熱圖） | 可投資最高 CAGR：每月動能前 5 隻 + QQQ 200 日線濾網 8.5% 到 10.6%（優勢全在 2000 年代）、波動率目標 20% / 63 日 10.2%、組合規則 9.8%、200 日線月底 9.4%；QQQ 7.2%、SPY 7.9%；不加濾網的選股全部 8% 以下 |
| 13 | [走動式樣本外測試](docs/13-walk-forward.md) | 五個規則家族、118 組參數，三段展開式與逐年重選，折價與參數穩定性，2010-2026 配對蒙地卡羅 | 樣本外沒有任何家族 CAGR 或 p10 贏 QQQ（18.7%）：波動率目標 14.0% 到 14.3%、均線 11.7% 到 13.1%、波動擴張 18.8% 打平且回撤 -30% 對 -35%；折價不到 1.7 點，問題是本來就沒贏 |
| 14 | [跨資產驗證](docs/14-cross-asset.md) | 凍結參數的 8 條規則套到 18 檔從未調參的 ETF，完整期間與 2010 後，配對蒙地卡羅與符號檢定 | 波動率目標報酬面 3 到 4 / 18 不輸（p 0.01 到 0.03，方向是輸），均線 0 到 3 / 18；只有波動擴張 14 / 18 為正但平均僅 +0.6 點；QQQ 是離群值 |
| 15 | [每月換成動能最高 5 隻](docs/15-top5-monthly.md) | 五種動能定義 × 持倉 5 / 10 / 20 / 50，每月再平衡，時點宇宙 2000-2024、分期、逐年、隨機對照、2025-26 今日名單；加 QQQ 200 日線濾網版與 150 / 250 日敏感度 | 前 5 隻 CAGR 3.6% 到 6.9%，低於 SPY 7.7% 與隨機 5 隻中位數 6.5%，回撤 -59% 到 -82%；加 QQQ 200 日線濾網後 9.8% 到 11.5%，勝 SPY 且高於隨機加濾網最高值，但優勢全在 2000 年代，2010 後仍輸指數，回撤 -49% 到 -58%，對濾網長度敏感 |
| 16 | [各策略操作方法](docs/16-playbook.md) | 總表中每個策略的逐步操作：基準、+10/+10 選股 A/B/C、反轉選股、每月前 N 隻（含濾網）、QQQ 全進全出擇時、按比例調整曝險，與實務注意 | 操作手冊，無新數據 |
| 17 | [反推動能爆發的前兆](docs/17-precursors.md) | 963 個進榜事件的事件研究：37 個技術特徵（含成交量）與 SEC EDGAR 時點財報 15 個特徵，單因子 lift、逐年樣本外模型、走動式組合對動能前 5 | 能預測進榜的全是動能本身的化身；續漲最好只有營收加速度 AUC 0.61；分數組合樣本外全輸單純動能（技術 8.6% 對 8.7%，基本面 4.8% 到 7.9% 對 14.0%） |
| 18 | [最終判定：哪些可用、哪些不可用](docs/18-verdict.md) | 全部策略的可用性判定：不可用（假象來源）、勉強可用（買的是什麼、代價）、穩的結論、建議 | 沒有主動規則在樣本外穩定勝過買進持有指數；擇時只是尾端保險；選股與前兆證據全為負面 |
| 19 | [要多準才夠：反推訊號能否找到 30% / 30%](docs/19-skill-needed.md) | 注入已知程度的預知（選股 IC、擇時命中率），量 2000-2024 的 CAGR 與 MDD；反推訊號的正確流程、假象來源，與網格的六種用法（放行門檻、審核宣稱、力氣分配、可驗證性、付費資料門檻、合理目標） | 年回報 30% 加 MDD 30% 以內需要選股 IC 約 0.5 或擇時命中率約 85%，兩者合用最低也要 IC 0.15 配命中 70%；現有訊號（IC 0.005、命中 55% 到 60%）差一個數量級；反推只會在樣本內成立 |
| 20 | [按第 19 節流程正向找 30% / 30%](docs/20-forward-search.md) | 事前門檻、57 個特徵的月度 IC 稽核（亂數對照、走動式合成）、29 條擇時規則的命中率稽核、把稽核分數配開關實際交易並對照網格 | 0 / 57 特徵過 IC 0.03，合成樣本外 IC 0.018；0 / 29 擇時規則過 70%，最佳 61% 到 67% 與永遠看多無法區分；唯一量到的是 1 個月反轉，2020 後失效、換手 88%；實際 8% 到 16% / −30% 到 −78%，不達標 |
| | [完整報告（單一檔）](docs/ANALYSIS.md) | 第 1 到 20 節合併版 | |

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
| `sp500_strategy_table.py` | 所有策略的統一比較表（共同期間指標、配對蒙地卡羅、2025-26、逐年報酬與至今總報酬） | `strategy_table_2000_2024.csv`、`strategy_table_2025_2026.csv`、`strategy_table_yearly.csv`、`docs/12-strategy-table.md`、`docs/strategy_yearly.png` |
| `sp500_walk_forward.py` | 走動式樣本外測試：展開式視窗重選參數，拼接樣本外淨值，折價與參數穩定性 | `sp500_walk_forward_*.csv`、`docs/walk_forward.png` |
| `sp500_cross_asset.py` | 跨資產驗證：QQQ 上凍結的規則套到 18 檔 ETF，彙總與符號檢定 | `sp500_cross_asset_*.csv`、`docs/cross_asset.png` |
| `sp500_top5_monthly.py` | 每月換成動能最高 N 隻（五種定義、N = 5 到 50、QQQ 均線濾網版、隨機對照、2025-26） | `sp500_top5_monthly_*.csv`、`docs/top5_monthly.png` |
| `sp500_precursor_events.py` | 動能進榜事件表（事件、對照、事後報酬、續漲標籤） | `precursor_events*_sample.csv` |
| `sp500_precursor_technical.py` | 技術面前兆：特徵、lift、逐年樣本外模型、走動式組合 | `precursor_tech_*.csv`、`docs/precursors_technical.png` |
| `sp500_precursor_fundamental.py` | 基本面前兆：EDGAR 時點季報、特徵、模型、走動式組合 | `precursor_fund_*.csv`、`docs/precursors_fundamental.png` |
| `sp500_skill_needed.py` | 要多準才夠：注入已知 IC 或命中率的訊號，量 CAGR 與 MDD | `sp500_skill_needed_*.csv`、`docs/skill_needed.png`、`docs/skill_needed_combined.png` |
| `sp500_signal_audit.py` | 橫截面訊號 IC 稽核：57 個特徵的月度 IC、亂數對照、錨定走動式合成分數 | `signal_audit_*.csv`、`docs/signal_audit.png` |
| `sp500_timing_audit.py` | 擇時命中率稽核：29 條月底規則對 SPY / QQQ 下月方向，Wilson 區間、兩種亂數對照、開關策略 CAGR / MDD | `timing_audit_*.csv`、`docs/timing_audit.png` |
| `sp500_forward_search.py` | 正向搜尋：把稽核分數配開關實際交易，對照第 19.3 節網格 | `forward_search_*.csv`、`docs/forward_search.png` |

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
python sp500_strategy_table.py              # 全部策略總表
python sp500_walk_forward.py                # 走動式樣本外測試
python sp500_cross_asset.py                 # 跨資產驗證（下載 18 檔 ETF）
python sp500_top5_monthly.py                # 每月換成動能最高 N 隻
python sp500_precursor_events.py            # 動能進榜事件表
python sp500_precursor_technical.py         # 技術面前兆
python sp500_precursor_fundamental.py       # 基本面前兆（EDGAR 首次下載約 50 分鐘）
python sp500_skill_needed.py                # 要多準才夠（約 3 分鐘）
python sp500_signal_audit.py                # 訊號 IC 稽核（約 2 分鐘）
python sp500_timing_audit.py                # 擇時命中率稽核（數秒）
python sp500_forward_search.py              # 正向搜尋（約 5 分鐘，需前兩者輸出）
```

快取檔（`prices_cache.csv`、`prices_long_cache.csv`、`membership_long.pkl`、`spy_long_cache.csv`、`spy_full_cache.csv`、`qqq_full_cache.csv`、`tlt_full_cache.csv`、`gld_full_cache.csv`、`irx_cache.csv`、`vix_cache.csv`）與 28 MB 的 `sp500_momentum_history.csv` 不進 git，重跑即可產生。

## 資料來源

- 成份股名單：[datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies)（現行），[fja05680/sp500](https://github.com/fja05680/sp500)（1996 年起每日歷史成員）。
- 股價：Yahoo Finance 調整後收盤價，透過 `yfinance`。已下市的歷史成份股沒有價格資料，長期回測因此仍有殘餘的存活偏差。

## 評分規則摘要

每天在全體成份股中做橫斷面十分位排名：上升的股票依速度給 +1（最慢）到 +10（最快），下降的股票依跌速給 -1（最慢）到 -10（最快），持平為 0。分數是相對排名，不是絕對報酬門檻。

本 repo 是研究紀錄，不是投資建議。
