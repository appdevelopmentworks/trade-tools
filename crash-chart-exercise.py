import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
import re
import numpy as np
from PIL import Image
#自作ライブラリー
from mystock_lib import *

#騰落データセット取得関数
@st.cache_data
def get_stock_data(code,maspan=20, rsispan=14):
    df = yf.download(code, progress=False)
    df.columns = [col[0] for col in df.columns]
    dfc = df.copy()
    #CRSI
    df["CRSI"] = calculate_connors_rsi(df)
    #RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsispan).rsi()
    #BBand
    df["MA"] = df["Close"].rolling(window=maspan).mean()
    #標準偏差
    df["σ"] = df["Close"].rolling(window=maspan).std()
    #Z値
    df["Z値"] = (df["Close"] - df["MA"]) / df["σ"]
    #騰落幅
    df["前日終値"] =df["Close"].shift(1)
    df["前日終-始値騰落率"] = (df["Open"] - df["前日終値"] ) /df["前日終値"] *100
    df["前日終-終値騰落率"] = (df["Close"] - df["前日終値"] ) /df["前日終値"] *100
    df["当日騰落率"] = (df["Close"] - df["Open"] ) / df["Open"] *100
    del df["Direction"],df["Streak"]
    #
    df.rename(columns={'Close':'終値', 'High':'高値', 'Low':'安値', 'Open':'始値', 'Volume':'出来高'}, inplace=True)    
    return df[df["出来高"]!=0][100:], dfc

#ラジオボタンの値をブール型に変換
def bl_asd_option(value):
    flg =False
    if value == "降順":
        flg = True
    return flg

#チャート描画用のデータセットを返す  
def get_chart_df(df, baseday, mae, ato, hmap=10, emafp=5, emasp=13, rsifp=3, rsisp=5, macdfp=12, macdsp=26,macdsigp=9,hvp=20):
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
    #
    start = baseday - datetime.timedelta(ato)
    end = baseday + datetime.timedelta(mae)
    return df[(df.index >= start) & (df.index <= end)]

#チャート描画関数
def plot_chart(df, title):    
    #インデックスを文字列型に（休日の抜けを無くす）
    df.index = pd.to_datetime(df.index).strftime('%m-%d-%Y')

    layout = {
        "height":800,
        "title":{"text": "{}".format(title), "x": 0.5},
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

    return (fig)




#############################################################
# セッションステートに初期値を設定
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = datetime.date.today() 

st.title("暴落暴騰エクササイズ")

st.text("暴落暴騰チャートで予行演習")
image = Image.open("headercrash.png")
st.image(image)
st.caption("暴落暴騰時のチャートを予習してリスク対応！")


col1, col2 = st.columns(2)
with col1:
    ticker = checkTicker(st.text_input("コード：", value="^N225"))
    optsigma =st.radio("Z値", ["降順", "昇順"], horizontal=True)
with col2:
    optrsi =st.radio("RSI", ["降順", "昇順"], horizontal=True)
    optcrsi =st.radio("CRSI", ["降順", "昇順"], horizontal=True)

#ランキング表とチャート描画用二つのデータセットを受け取る
df, dfc = get_stock_data(ticker)
#コードから企業名を取得
title = yf.Ticker(ticker).info['shortName']

#"Z値","CRSI","RSI"の降順でソートしデータフレームに入れる
dftouraku = df.sort_values(by=["Z値","CRSI","RSI"], ascending=[bl_asd_option(optsigma),bl_asd_option(optcrsi),bl_asd_option(optrsi)])
#先頭の行（最悪の日）をセッション変数に代入
st.session_state.selected_date = dftouraku.index[0]

st.subheader("騰落日ランキング")
#デフォルトで日経225の最悪の順に並び替えて表示
st.dataframe(dftouraku, height=200)

#最高・最悪の日をセット、前10日、後50日を表示
col3, col4, col5 = st.columns(3)
with col3:
    crashday = st.date_input("暴落・暴騰した日", value=st.session_state.selected_date)
with col4:
    mae = int(st.text_input("前何日", value=10))
with col5:
    ato = int(st.text_input("後何日", value=50))

#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #c71585;  /* 背景色 */
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
btnplot = st.button("チャート描画")

#チャートの処理は先にやっておく
baseday = pd.to_datetime(crashday)
#グラフオブジェクトを取得
dfcc = get_chart_df(dfc, baseday, mae, ato)

if btnplot:
    fig = plot_chart(dfcc,title)
    st.plotly_chart(fig)
        
