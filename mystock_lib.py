import numpy as np
import re
import ta


#日本株をYfinance用に変換する関数
def checkTicker(ticker):
    # 有効な英大文字を定義
    valid_letters = "ACDFGHJKLMPNRSTUWX-Y"
    # 正規表現パターン
    pattern = rf"^[0-9][0-9{valid_letters}][0-9][0-9{valid_letters}]$"
    if not re.match(pattern, ticker):
        return ticker
    else:
        return ticker + ".T"

#コナーズRSI（CRSI）    
def calculate_connors_rsi(df, rsi_period=3, streak_period=2, roc_period=100):
    # 終値の差分を計算
    delta = df['Close'].diff()
    
    # 上昇と下降のストリークを計算
    df['Direction'] = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    streaks = df['Direction'].replace(0, np.nan).ffill().groupby(df['Direction'].ne(0).cumsum()).cumcount() + 1
    df['Streak'] = np.where(df['Direction'] != 0, streaks, 0)

    # CRSIの各成分を計算
    rsi = ta.momentum.RSIIndicator(df['Close'], window=rsi_period).rsi()
    streak_rsi = ta.momentum.RSIIndicator(df['Streak'], window=streak_period).rsi()
    roc = ta.momentum.ROCIndicator(df['Close'], window=roc_period).roc()

    # CRSIを計算
    connors_rsi = (rsi + streak_rsi + roc) / 3
    return connors_rsi

#HMA移動平均
def hma(series, window):
    """Hull Moving Average (HMA) を計算する関数"""
    half_window = window // 2
    sqrt_window = int(np.sqrt(window))

    wma1 = series.rolling(half_window).apply(lambda x: np.average(x, weights=np.arange(1, half_window + 1)), raw=True)
    wma2 = series.rolling(window).apply(lambda x: np.average(x, weights=np.arange(1, window + 1)), raw=True)
    delta_wma = 2 * wma1 - wma2

    hma_series = delta_wma.rolling(sqrt_window).apply(lambda x: np.average(x, weights=np.arange(1, sqrt_window + 1)), raw=True)
    return hma_series