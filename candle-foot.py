import streamlit as st
from  pandas_datareader import data as web
import pandas as pd
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
#自作ライブラリー
from mystock_lib import *


@st.cache_data
def get_stock_data(ticker):
    data = pd.DataFrame()
    data = web.DataReader(ticker, data_source="stooq").sort_index(ascending=True) 
    #始値からの下げ幅の全体に対する割合
    data["Open-Low-Ratio"]= (data["Open"] - data["Low"]) / (data["High"] - data["Low"])
    #始値からの上げ幅の全体に対する割合
    data["Open-High-Ratio"]= (data["High"] - data["Open"]) / (data["High"] - data["Low"])
    #その日の上げ幅
    data["Close-Open"] = data["Close"] - data["Open"]
    #真の値幅
    data["TR"] = np.maximum(np.maximum(data.High - data.Low, data.High - data.Close.shift(1)), data.Low - data.Close.shift(1))
    data["Open-Low-TRratio"]= (data["Open"] - data["Low"]) / data["TR"]
    #ATR
    data["ATR3"] = data["TR"].rolling(window=3).mean()
    data["Open-Low-ATR3"]= (data["Open"] - data["Low"]) / data["ATR3"]
    
    return data


def plot_scatter(df, colname, title, xlabel, ylabel, color):
    sns.set_theme(font="IPAexGothic")
    plt.figure(figsize=(10, 5))
    plt.scatter(x=df[colname], y=df['Close-Open'], c=color, alpha=0.5)
    plt.axhline(0, color='blue', lw=1) 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    return plt

def plot_linechart(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close-Open'])
    plt.axhline(0, color='red', lw=1) 
    plt.title('利益が出る下ヒゲと値幅の比率')
    plt.xlabel('下ヒゲと値幅(高値-安値)比率')
    plt.ylabel('金額(終値-始値)')
    plt.grid(True)
    return plt



st.title("ローソク足下髭と当日上昇の関係")

st.text("下髭と上昇幅の関係をグラフで表示")
image = Image.open("./images/headercandlelow.png")
st.image(image)
st.caption("ティッカーコードを入力してください")

col1, col2 = st.columns(2)
with col1:
    ticker = checkTicker_stooq(st.text_input("コード：", value="^NKX"))
with col2:
    #書式付きボタン
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80%;
            background-color: #32cd32;  /* 背景色 */
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
    #データ取得
    df = get_stock_data(ticker)

    
    st.subheader("下ヒゲと当日変動幅の関係")
    plt = plot_scatter(df, "Open-Low-ATR3", "下ヒゲ/ATR3期間", "下ヒゲ-ATR3比率", "金額(終値-始値)", "#ff1493")
    st.pyplot(plt)
    st.write("本日の３期間ATR:" + str(round(df["ATR3"][-1],2)))
    plt2 = plot_scatter(df, "Open-Low-TRratio", "下ヒゲ/真の値幅", "下ヒゲ-TR比率", "金額(終値-始値)", "green")
    st.pyplot(plt2)
    st.write("本日の真の値幅:" + str(round(df["TR"][-1],2)))
    
    df2 = df.groupby('Open-Low-Ratio').mean()
    #df2 = df2[df2['Close-Open'] > 0]
    st.subheader("始値からどれくらい下がったら損失になるか")
    plt2 = plot_linechart(df2)
    st.pyplot(plt2)
    st.write("本日の値幅:" + str(round((df["High"][-1] - df["Low"][-1]),2)))
    st.subheader("上昇幅順株価")
    st.dataframe(df2.sort_values("Close-Open", ascending=False))


