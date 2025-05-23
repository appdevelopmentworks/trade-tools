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
def get_tdwm_data(ticker):
    data = pd.DataFrame()
    data = web.DataReader(ticker, data_source="stooq").sort_index(ascending=True) 

    #前日終値から今日の初値差分
    data["前日終値当日始値上昇幅"] = data["Open"] - data["Close"].shift(1)
    #今日上昇したか
    data["当日上昇幅"] = data["Close"] - data["Open"]
    #前日の引けから今日の引け
    data["前日引け当日引け上昇幅"] = data["前日終値当日始値上昇幅"] + data["当日上昇幅"]
    #年月日曜日
    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day"] = data.index.day
    data["weekday"] = data.index.weekday
    #月の営業日
    data["Bday"] = data.groupby(["month","year"]).cumcount() + 1
    #単純利益
    data["simple_return"] = (data["Close"] / data["Close"].shift(1)) - 1
    #Logリターン
    data["log_return"] =  np.log(data["Close"] / data["Close"].shift(1))
    return data


def plt_bar(x, y, title, xlbl, ylbl, lglbl, color):
    # 棒グラフの作成
    sns.set_theme(font="IPAexGothic")
    plt.figure(figsize=(12, 6))
    plt.bar(x, y, label=lglbl, color=color)
    plt.axhline(0, color='red', linewidth=1, linestyle='--')  
    # グラフのカスタマイズ
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # グラフの表示
    return plt

def plt_line_bar(x, y1, y2, y3, lbl1, lbl2, lbl3, title, xlbl, ylbl, color):
    # 
    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, label=lbl1, c="orange" )
    plt.plot(x, y2, label=lbl2, c="green")
    plt.bar(x, y3, label=lbl3, color=color)
    plt.axhline(0, color='red', linewidth=1, linestyle='--')  
    # グラフのカスタマイズ
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    return plt

#年度別表示
def plt_month_chart(df):
    plt.figure(figsize=(12, 6))

    # MultiIndexの最初のレベル（'year'）のユニークな値を取得
    for year in df.index.get_level_values('year').unique():
        # 特定の年のデータを抽出
        year_data = df.xs(year, level='year')
        # X軸に月 (month), Y軸に log_return をプロット
        # year_data.index には月の値が入っている
        plt.plot(year_data.index, year_data['simple_return'], marker='o', linestyle='-', label=f'{year}年')

    plt.title('年度別・月別 平均リターン')
    plt.xlabel('月')
    plt.ylabel('平均利益率')
    plt.xticks(np.arange(1, 13)) # X軸の目盛りを1月から12月まで表示
    plt.legend(title='年度', loc='upper left')
    plt.axhline(0, color='red', linewidth=1, linestyle='--') 
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    return plt

###################################################################
weekdayj = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金'}
###################################################################

st.title("TDW・TDM・月別リターン")

st.text("銘柄のトレードしやすい日可視化")
image = Image.open("./images/headertdwtdm.png")
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
    btnplot = st.button("グラフ表示")
    
    
if btnplot:
    #データ取得
    data = get_tdwm_data(ticker)
    #週の傾向
    st.subheader("一週間の傾向:")
    dataw = data.groupby("weekday").mean()[["前日終値当日始値上昇幅","当日上昇幅","前日引け当日引け上昇幅","simple_return","log_return"]]
    dataw.index = dataw.index.map(weekdayj)
    plt11 = plt_line_bar(dataw.index, dataw["前日終値当日始値上昇幅"],
                         dataw["前日引け当日引け上昇幅"],
                         dataw["当日上昇幅"],
                         "前日終値当日始値上昇幅",
                         "前日引け当日引け上昇幅",
                         "当日上昇幅",
                         "曜日別平均上昇幅", "曜日", "上昇幅", "#4682b4")
    st.pyplot(plt11)
    #
    plt12 = plt_bar(dataw.index, dataw["log_return"] * 100,
                  "曜日別平均上昇率", "曜日", "上昇率(%)", "前日からの上昇率(%)", "#4169e1")
    st.pyplot(plt12)
    
    #営業日別上昇日
    st.subheader("営業日別の傾向:")
    dataBday = data.groupby("Bday").mean()
    plt21 = plt_line_bar(dataBday.index, dataBday["前日終値当日始値上昇幅"],
                         dataBday["前日引け当日引け上昇幅"],
                         dataBday["当日上昇幅"],
                         "前日終値当日始値上昇幅",
                         "前日引け当日引け上昇幅",
                         "当日上昇幅",
                         "営業日別平均上昇幅", "営業日", "上昇幅", "#9932cc")
    st.pyplot(plt21)
    plt22 = plt_bar(dataBday.index, dataBday["simple_return"] * 100,
                  "営業日別平均利益率", "営業日", "上昇率(%)", "前日からの上昇率(%)", "#800080")
    st.pyplot(plt22)
    
    #月別の傾向
    st.subheader("月別の傾向:")
    dataym = data.groupby(["year", "month"]).sum()
    plt31 = plt_month_chart(dataym)
    st.pyplot(plt31)
    
    dataMonth = data.groupby("month").mean()
    plt32 = plt_bar(dataMonth.index, dataMonth["simple_return"] * 20 * 100,
                  "月別平均上昇率", "月", "上昇率(%)", "月平均上昇率(%)", "#db7093")
    st.pyplot(plt32)
    