import streamlit as st
import yfinance as yf
import ta as ta
import pandas as pd
import numpy as np
import re
from PIL import Image
import plotly.graph_objects as go


def checkTicker(ticker):
    # 有効な英大文字を定義
    valid_letters = "ACDFGHJKLMPNRSTUWX-Y"
    # 正規表現パターン
    pattern = rf"^[0-9][0-9{valid_letters}][0-9][0-9{valid_letters}]$"
    if not re.match(pattern, ticker):
        return ticker
    else:
        return ticker + ".T"
    
#HMA(ハル移動平均)
def hma(series, window):
    """Hull Moving Average (HMA) を計算する関数"""
    half_window = window // 2
    sqrt_window = int(np.sqrt(window))

    wma1 = series.rolling(half_window).apply(lambda x: np.average(x, weights=np.arange(1, half_window + 1)), raw=True)
    wma2 = series.rolling(window).apply(lambda x: np.average(x, weights=np.arange(1, window + 1)), raw=True)
    delta_wma = 2 * wma1 - wma2

    hma_series = delta_wma.rolling(sqrt_window).apply(lambda x: np.average(x, weights=np.arange(1, sqrt_window + 1)), raw=True)
    return hma_series

def plot_chart(df, coname):
    layout = {
        "height":1200,
        "title":{"text": "{}".format(coname), "x": 0.5},
        "xaxis":{"title": "日付", "rangeslider":{"visible":False}},
        "yaxis1":{"domain":[.46, 1.0], "title": "価格（円）", "side": "left", "tickformat": ","},
        "yaxis2":{"domain":[.40,.46]},
        "yaxis3":{"domain":[.30,.395], "title":"出来高", "side":"right"},
        "yaxis4":{"domain":[.20,.295], "title":"RSI", "side":"right"},
        "yaxis5":{"domain":[.10,.195], "title":"MACD", "side":"right"},
        "yaxis6":{"domain":[.00,.095], "title":"HV", "side":"right"},
        "plot_bgcolor":"light blue"
    }

    data = [
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                    increasing_line_color="red",
                    increasing_line_width=1.0,
                    increasing_fillcolor="red",
                    decreasing_line_color="blue",
                    decreasing_line_width=1.0,
                    decreasing_fillcolor="blue"
                    ),
        #移動平均
        go.Scatter(x=df.index, y=df['hma'], name="HMA10",
                line={"color": "red", "width":1.2}),
        go.Scatter(x=df.index, y=df['emaF'], name="EMA_F",
                line={"color": "royalblue", "width":1.2}),
            go.Scatter(x=df.index, y=df['emaS'], name="EMA_S",
                line={"color": "lightseagreen", "width":1.2}),   
        go.Scatter(x=df.index, y=df['sma200'], name="SMA200",
                line={"color": "darkred", "width":1.0}),
        #出来高
        go.Bar(yaxis="y3", x=df.index, y=df['Volume'], name="Volume",marker={"color":"slategray"}),
        #RSI
        go.Scatter(yaxis="y4", x=df.index, y=df['rsiF'], name="RSI_F",
                line={"color":"magenta", "width":1}),
        go.Scatter(yaxis="y4", x=df.index, y=df['rsiS'], name="RSI_S",
                line={"color":"green", "width":1}),
        go.Scatter(yaxis="y4", x=df.index, y=df['30'], name="30",
                line={"color":"black", "width":0.5}),
        go.Scatter(yaxis="y4", x=df.index, y=df['70'], name="70",
                line={"color":"black", "width":0.5}),
        #MACD
        go.Scatter(yaxis="y5", x=df.index, y=df['macd'], name="MACD",
                line={"color":"magenta", "width":1}),
        go.Scatter(yaxis="y5", x=df.index, y=df['signal'], name="MACDSIG",
                line={"color":"green", "width":1}),
        go.Bar(yaxis="y5", x=df.index, y=df['hist'], name="MACDHIST",marker={"color":"slategrey"}),    
        
        #HV
        go.Scatter(yaxis="y6", x=df.index, y=df['hv'], name="HV",
                line={"color":"red", "width":1}),    
    ]

    fig = go.Figure(data = data, layout = go.Layout(layout))
    return fig


######################################
st.title("plotlyテクニカルチャート")

st.text("HMAとHVを使ったチャートです短期売買（スイング）に最適化")
image = Image.open("headermoneymoney.png")
st.image(image)
st.caption("パラメーターはスイングトレードで最適化しています")

col1, col2 = st.columns(2)

with col1:
    #ティッカー（初期値は、金/ドル）
    ticker = st.text_input("コードを入力:", value="GC=F")
with col2:
    #HMA
    hmap = int(st.text_input("HMA期間", value=10))

    
with col1:
    #EMA
    emafp = int(st.text_input("EMA Fast", value=5))
    emasp = int(st.text_input("EMA Slow", value=13))
with col2:
    #RSI
    rsifp = int(st.text_input("RSI Fast", value=3))
    rsisp = int(st.text_input("RSI Slow", value=5))

col1, col2, col3 = st.columns(3)

with col1:
    #MACD
    macdfp = int(st.text_input("MACD Fast", value=12))
with col2:
    macdsp = int(st.text_input("MACD Slow", value=26))
with col3:
    macdsigp = int(st.text_input("MACD Signal", value=9))
hvp = int(st.text_input("HV(ヒストリカルボラティリティー)期間", value=20))
#表示期間
dispp = int(st.slider("表示期間(日):", min_value=30, max_value=1000, value=100, step=1))
#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 30%;
        background-color: #4b0082;  /* 背景色 */
        color: white;  /* 文字色 */
        padding: 15px;  /* パディング */
        text-align: center;  /* テキストを中央揃え */
        text-decoration: none;  /* テキストの下線をなし */
        font-size: 16px;  /* フォントサイズ */
        border-radius: 4px;  /* 角を丸くする */
        cursor: pointer;  /* カーソルをポインタに */
    }
    </style>
    """,
    unsafe_allow_html=True
) 
btnshow = st.button("描画")

if btnshow:
    df = yf.download(checkTicker(ticker), progress=False)
    df.columns = [row[0] for row in df.columns]

    # Moving Average
    df['hma'] = hma(df['Close'], window=hmap)
    df["emaF"] = ta.trend.ema_indicator(df['Close'], window=emafp)
    df["emaS"] = ta.trend.ema_indicator(df['Close'], window=emasp)
    df["sma200"] = ta.trend.sma_indicator(df['Close'], window=200)

    # RSI
    df['rsiF'] = ta.momentum.rsi(df['Close'], window=rsifp)
    df['rsiS'] = ta.momentum.rsi(df['Close'], window=rsisp)
    df['70'], df['30'] = [70 for _ in df['Close']], [30 for _ in df['Close']]

    # MACD
    df["macd"] = ta.trend.macd(df["Close"], window_fast=12, window_slow=26)
    df["signal"] = ta.trend.macd_signal(df["Close"], window_fast=macdfp, window_slow=macdsp, window_sign=macdsigp)
    df["hist"] = ta.trend.macd_diff(df["Close"], window_fast=macdfp, window_slow=macdsp, window_sign=macdsigp)

    # Historical Volatility
    df['hv'] = df['Close'].pct_change().rolling(window=hvp).std() * np.sqrt(252) * 100
    #インデックスを文字列型に（休日の抜けを無くす）
    df.index = pd.to_datetime(df.index).strftime('%m-%d-%Y')
    
    #描画
    plt = plot_chart(df.tail(dispp), yf.Ticker(checkTicker(ticker)).info['shortName'])
    st.plotly_chart(plt)