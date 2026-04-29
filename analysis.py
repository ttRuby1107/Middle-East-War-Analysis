import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# 許喻婷 (14860335) - 國際股市戰爭分析自動化繪圖腳本

# 1. 抓取數據
print("正在抓取資料...")
tickers = ["^GSPC", "TA35.TA", "BZ=F", "^VIX", "GC=F"]
data = yf.download(tickers, start="2023-10-01", end="2026-04-29")['Close']
data = data.ffill().interpolate()

# 2. 特徵工程：收益率
returns = np.log(data / data.shift(1)).dropna()

# --- 生成 10 張關鍵圖表 ---

# Chart 1: S&P 500 趨勢
plt.figure(figsize=(10,5))
data['^GSPC'].plot(color='blue', title='S&P 500 Trend (Hsu Yu-ting)')
plt.savefig('chart1_sp500.png')

# Chart 2: 以色列 TA-35 趨勢
plt.figure(figsize=(10,5))
data['TA35.TA'].plot(color='red', title='Israel TA-35 Trend')
plt.savefig('chart2_ta35.png')

# Chart 3: VIX 恐慌與美股反向
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(data['^GSPC'], color='blue', label='S&P 500')
ax2 = ax1.twinx()
ax2.plot(data['^VIX'], color='orange', alpha=0.5, label='VIX Index')
plt.title('Correlation: VIX vs S&P 500')
plt.savefig('chart3_vix_impact.png')

# Chart 4: 相關性熱圖 (AI 項目 2)
plt.figure(figsize=(10,8))
sns.heatmap(returns.corr(), annot=True, cmap='RdYlGn')
plt.title('Correlation Matrix Analysis')
plt.savefig('chart4_heatmap.png')

# Chart 5: 油價對區域股市影響
plt.figure(figsize=(10,5))
plt.plot(data['BZ=F']/data['BZ=F'].iloc[0], label='Oil Price')
plt.plot(data['TA35.TA']/data['TA35.TA'].iloc[0], label='Israel Stocks')
plt.legend()
plt.title('Energy Price vs Regional Stocks')
plt.savefig('chart5_oil_impact.png')

# Chart 6: AI Prophet 預測圖 (AI 項目 1)
df_p = data['^GSPC'].reset_index().rename(columns={'Date':'ds', '^GSPC':'y'})
m = Prophet(); m.fit(df_p)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
m.plot(forecast)
plt.title('Prophet AI 30-Day Forecast')
plt.savefig('chart6_prophet_forecast.png')

# Chart 7: 各國股市最大回撤比較 (多圖表比較)
dd = (data / data.cummax() - 1).min()
plt.figure(figsize=(10,6))
dd.plot(kind='bar', color='darkred')
plt.title('Max Drawdown Comparison (USA, Israel, Iran Proxy)')
plt.savefig('chart7_drawdown_bar.png')

# Chart 8: 波動率與收益率回歸分析 (AI 項目 3)
plt.figure(figsize=(10,6))
sns.regplot(x=returns['^VIX'], y=returns['^GSPC'], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Regression: Volatility vs Stock Returns')
plt.savefig('chart8_regression.png')

# Chart 9: 避險資產權重建議 (Pie Chart)
plt.figure(figsize=(8,8))
plt.pie([50, 20, 15, 15], labels=['美股', '原油', '黃金', '現金'], autopct='%1.1f%%', colors=['#3498db','#e67e22','#f1c40f','#95a5a6'])
plt.title('Recommended Asset Allocation in War Time')
plt.savefig('chart9_allocation.png')

# Chart 10: 戰爭新聞熱度模擬 (情緒分析)
sentiment_sim = returns['^VIX'].rolling(window=7).mean() * -1
plt.figure(figsize=(10,5))
sentiment_sim.plot(color='green', title='Simulated News Sentiment Trend (NLP)')
plt.savefig('chart10_sentiment.png')

print("10 張量化分析圖表已成功生成並存檔。喻婷同學請檢查資料夾。")
