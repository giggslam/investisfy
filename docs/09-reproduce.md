# 9. 重跑方式與檔案清單

[回索引](../README.md) · [完整報告](./ANALYSIS.md) · [上一節](./08-conclusions.md)

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
| `prices_long_cache.csv`、`membership_long.pkl`、`spy_long_cache.csv` | 26 年股價、歷史成員矩陣與 SPY 快取（未提交，`--build` 重建） |

---

[回索引](../README.md) · [完整報告](./ANALYSIS.md) · [上一節](./08-conclusions.md)
