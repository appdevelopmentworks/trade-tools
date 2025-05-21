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

#stooq用に変換する関数
def checkTicker_stooq(ticker):
    # 文字列型でない場合は、そのまま返す（エラーハンドリングとして）
    if not isinstance(ticker, str):
        return ticker

    # 1. ^で始まる文字列はそのまま返す (例：^NKX)
    if ticker.startswith("^"):
        return ticker

    # 2. 日本株の判定
    # 2a. 4桁の数字の場合 (例：8151 -> 8151.JP)
    if re.fullmatch(r"^[0-9]{4}$", ticker):
        return ticker + ".JP"

    # 2b. 元の関数の日本株パターンを流用し、末尾に .JP を追加
    #     (1桁目:数字、2桁目:数字or特定大文字、3桁目:数字、4桁目:数字or特定大文字)
    valid_letters_jp = "ACDFGHJKLMPNRSTUWX-Y"  # 元の関数で定義されていた有効な文字
    # f-string内で {} をリテラルではなく変数展開として使う
    pattern_jp_original_logic = rf"^[0-9][0-9{valid_letters_jp}][0-9][0-9{valid_letters_jp}]$"
    if re.fullmatch(pattern_jp_original_logic, ticker):
        return ticker + ".JP"

    # 3. 米国株の判定 (1文字以上の英大文字のみ) (例：META -> META.US)
    #    上記の日本株のパターンに一致しなかった場合に評価
    if re.fullmatch(r"^[A-Z]+$", ticker):
        return ticker + ".US"

    # 4. 上記のいずれにも当てはまらない場合はそのまま返す
    return ticker


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