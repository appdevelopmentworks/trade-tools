import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.stats import norm
import datetime
from PIL import Image


@st.cache_data
def get_stockdata_stooq(ticker, start):
    data = pd.DataFrame()
    #stooqから株価データをダウンロード
    data[ticker] = web.DataReader(ticker, data_source="stooq", start=start)["Close"]
    #Stooqの場合、日付降順になっているので日付昇順に変更
    data = data.sort_index(ascending=True)
    return data    

@st.cache_data
def montecarlo_simulation(data, iterations, t_intervals):
    log_returns = np.log(1 + data.pct_change())
    #利益の平均値と分散
    u = log_returns.mean()
    var = log_returns.var()
    #ドリフト
    drift = u - (0.5 * var)
    #利益の標準偏差
    stdev = log_returns.std()
    #日々リターンのマトリクス
    dairy_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))
    #全て０の配列を用意
    price_list = np.zeros_like(dairy_returns)
    #シュミレーション初日を設定（今日）
    S0 = data.iloc[-1]
    price_list[0] = S0
    #シュミレーションのループ
    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * dairy_returns[t]

    return price_list

def plot_chart(data, title):
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('期間')
    plt.ylabel('株価')
    return plt


############################################################################
today = datetime.date.today()
initial_date =  today.replace(year=today.year - 10)

############################################################################
st.title("株価モンテカルロシュミレーション")

st.text("モンテカルロシュミレーションで株価予測")
image = Image.open("./images/headermontecarlo.png")
st.image(image)
st.caption("ティッカーコードを入力してください")
st.text("例 S&P500:^SPX, 日経225:^NKX, 日本株:4902.JP, 米国株:IYR.US, 米国10年債:10YUSY.B")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("ティッカーコードを入力:", value="^NKX")
    iterations = int(st.text_input("試行回数:", value="100"))
with col2:
    start = st.date_input("標本データ開始期間:", value=initial_date)
    intervals = int(st.text_input("予測期間(日):", value="30"))

#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #ff00ff;  /* 背景色 */
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
btnSimulation = st.button("シュミレーション")

if btnSimulation:
    data = get_stockdata_stooq(ticker, start)
    plt = plot_chart(data, f"{ticker}:現在までの株価")
    st.pyplot(plt)
    pred = montecarlo_simulation(data, iterations, intervals)
    pltpred = plot_chart(pred, f"{ticker}:シュミレーション結果")
    st.pyplot(pltpred)