import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image




def get_stocks_data(tickers):
    tickers = tickers.split(',')
    data = yf.download(tickers, start="2020-01-01")["Close"]
    data.columns.name = None
    return data















st.title("ベータと資産評価モデル(CAPM)")

st.text("ポートフォリオのベータと資産を評価")
image = Image.open("./images/headerbetacapm.png")
st.image(image)
st.caption("お持ちの資産のコードをカンマ区切りで入力")

tickers = st.text_input("ティッカーコードを入力:", value="GOOGL,AAPL,META,AMZN,MSFT,NVDA,TSLA")

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
btncal = st.button("算定")


if btncal:
    df = get_stocks_data(tickers)
    st.dataframe(df)