# 2. 當前快照（2026-10-16）

[回索引](../README.md) · [完整報告](./ANALYSIS.md) · [上一節](./01-method.md) · [下一節](./03-persistence-2024-2026.md)

> 資料：S&P 500 成份股，Yahoo Finance 調整後收盤價。短期分析期間 2024-01-02 至 2026-10-16；長期回測 2000-01-03 至 2024-12-31。

## 2.1 上升與下降家數

| 階段 | 上升 | 下降 |
|---|---|---|
| 短期 | 289 | 214 |
| 中期 | 247 | 256 |
| 長期 | 332 | 171 |

趨勢標記：三階段全升 137 隻，全跌 68 隻，混合 298 隻。

## 2.2 綜合分數前十

| Symbol | Security | Sector | 短期 | 中期 | 長期 | 綜合 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|---|---|---|
| MPC | Marathon Petroleum | Energy | 10 | 10 | 10 | 10.00 | -6.58 | 19.17 | 197.37 |
| MRNA | Moderna | Health Care | 10 | 10 | 10 | 10.00 | -63.04 | -29.08 | 639.27 |
| AMGN | Amgen | Health Care | 10 | 9 | 9 | 9.33 | -9.71 | 29.67 | 34.84 |
| APA | APA Corporation | Energy | 8 | 10 | 10 | 9.33 | -33.86 | 11.54 | 105.65 |
| VLO | Valero Energy | Energy | 8 | 10 | 10 | 9.33 | -3.45 | 36.97 | 177.22 |
| HAS | Hasbro | Consumer Discretionary | 10 | 9 | 8 | 9.00 | 17.43 | 52.52 | 23.18 |
| PSX | Phillips 66 | Energy | 7 | 10 | 10 | 9.00 | -12.38 | 17.51 | 131.95 |
| XOM | ExxonMobil | Energy | 9 | 9 | 9 | 9.00 | 8.67 | 15.98 | 51.71 |
| COP | ConocoPhillips | Energy | 9 | 9 | 9 | 9.00 | -13.30 | -2.34 | 59.96 |
| CRL | Charles River Laboratories | Health Care | 7 | 10 | 10 | 9.00 | -21.20 | 8.06 | 50.47 |

## 2.3 綜合分數後十

| Symbol | Security | Sector | 短期 | 中期 | 長期 | 綜合 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|---|---|---|
| FSLR | First Solar | Information Technology | -9 | -9 | -8 | -8.67 | 2.37 | 48.22 | -33.91 |
| FDXF | FedEx Freight | Industrials | -8 | -10 | -8 | -8.67 | | | -25.03 |
| WYNN | Wynn Resorts | Consumer Discretionary | -8 | -10 | -9 | -9.00 | -7.92 | 41.02 | -36.72 |
| MOS | Mosaic Company | Materials | -10 | -8 | -9 | -9.00 | -30.60 | 1.10 | -19.74 |
| CHTR | Charter Communications | Communication Services | -10 | -8 | -10 | -9.33 | -12.45 | -39.10 | -47.25 |
| AXON | Axon Enterprise | Industrials | -9 | -10 | -9 | -9.33 | 136.19 | -4.44 | -31.28 |
| ARE | Alexandria Real Estate | Real Estate | -10 | -9 | -10 | -9.67 | -21.02 | -46.60 | -11.16 |
| SKYD | Skydance Corporation | Communication Services | -10 | -9 | -10 | -9.67 | -26.01 | 30.04 | -47.68 |
| APP | AppLovin | Communication Services | -10 | -10 | -10 | -10.00 | 735.04 | 108.08 | -60.50 |
| FICO | Fair Isaac | Information Technology | -10 | -10 | -10 | -10.00 | 76.01 | -15.08 | -54.38 |

FDXF 於 2026 年分拆上市，沒有 2024 與 2025 報酬。

## 2.4 產業（依平均綜合分數排序）

| Sector | 股票數 | 2024 中位數 | 2025 中位數 | 2026 YTD 中位數 | 平均綜合分數 | 全升 | 全跌 |
|---|---|---|---|---|---|---|---|
| Energy | 21 | 3.74 | 10.10 | 41.11 | 5.70 | 14 | 0 |
| Health Care | 60 | -2.02 | 8.28 | 18.76 | 4.33 | 32 | 1 |
| Consumer Staples | 34 | -0.92 | -5.39 | 13.72 | 2.48 | 17 | 4 |
| Information Technology | 75 | 17.24 | 10.88 | 26.19 | 1.19 | 14 | 4 |
| Materials | 24 | 2.46 | -1.58 | 13.78 | 0.96 | 8 | 4 |
| Industrials | 83 | 20.72 | 13.15 | 6.88 | -0.04 | 18 | 11 |
| Consumer Discretionary | 47 | 19.17 | 8.59 | -3.20 | -0.23 | 14 | 13 |
| Financials | 76 | 27.40 | 11.32 | 4.02 | -0.36 | 15 | 13 |
| Utilities | 31 | 16.69 | 15.31 | 5.55 | -0.67 | 2 | 4 |
| Real Estate | 30 | 9.91 | -5.44 | 5.27 | -1.74 | 2 | 7 |
| Communication Services | 22 | 37.92 | 13.54 | -11.18 | -2.05 | 1 | 7 |

能源與醫療保健是 2026 年的動能核心：能源 21 隻中 14 隻三階段全升，沒有一隻全跌。2024 年領漲的通訊服務、金融、工業在 2026 年明顯降速，通訊服務與非必需消費的 2026 中位數轉負。

---

[回索引](../README.md) · [完整報告](./ANALYSIS.md) · [上一節](./01-method.md) · [下一節](./03-persistence-2024-2026.md)
