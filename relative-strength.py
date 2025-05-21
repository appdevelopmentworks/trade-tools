import streamlit as st
from  pandas_datareader import data as web
import pandas as pd
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from PIL import Image
#自作ライブラリー
from mystock_lib import *

@st.cache_data
def get_relative_data(tickers):
    data = pd.DataFrame()
    for ticker in tickers:
        data[ticker] = web.DataReader(ticker, 'stooq', start="2020-01-01", end=datetime.datetime.now())["Close"]
    data = data.sort_index(ascending=True)
    #日時で何パーセント上昇したか、Na行は削除
    returns = data.pct_change().dropna()
    # 業種内の平均リターンを計算
    industry_average = returns.mean(axis=1)
    # 各株式のレラティブストレングスを計算
    relative_strength = returns.sub(industry_average, axis=0)
    return relative_strength


#
def plot_relative_chart(df):
    plt.figure(figsize=(12, 6))
    for ticker in df.columns:
        plt.plot(df.index, df[ticker], label=ticker)
    
    plt.title('レラティヴ・ストレングス')
    plt.xlabel('日付')
    plt.ylabel('レラティブストレングス')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)  # 基準線
    plt.legend(loc='upper left')
    plt.grid()
    return plt     







st.title("レラティヴ・ストレングス")

st.text("株価の相対的な強弱を可視化")
image = Image.open("./images/headerrelative.png")
st.image(image)
st.caption("ティッカーコードをカンマ区切りで")


txtTickers = st.text_input("コード：", value="GOOGL, AAPL, META, AMZN, MSFT, TSLA, NVDA")
#表示期間
span = st.number_input("表示期間：", value=20) 
#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #a52a2a;  /* 背景色 */
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
btnplot = st.button("グラフ表示")


if btnplot:
    tickers = txtTickers.replace(" ", "").split(',')
    tickers = [checkTicker_stooq(ticker) for ticker in tickers]
    df = get_relative_data(tickers)
    plt = plot_relative_chart(df.tail(span))
    st.subheader("相対的強弱チャート")
    st.pyplot(plt)
    st.subheader("相対リターン")
    st.dataframe(df)
