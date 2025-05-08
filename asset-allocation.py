import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas_datareader.data as web
import datetime
from PIL import Image


@st.cache_data
def get_asset_matrix(assets, colname ,start, end):
    pf_data = pd.DataFrame()
    for ticker in assets:
        pf_data[ticker] = web.DataReader(ticker, 'stooq', start=start, end=end)["Close"]
    pf_data.columns = colname
    return pf_data.sort_index(ascending=True)

def plot_chart(data, title):
    plt.figure(figsize=(10, 5))
    plt.plot(data,label=data.columns)
    plt.title(title)
    plt.xlabel('期間')
    plt.ylabel('資産')
    plt.legend()
    return plt

############################################################################
today = datetime.date.today()
startday =  datetime.date(2020, 1, 6)
business_days = 250

############################################################################
st.title("資産配分ツール")

st.text("ポートフォリオの構成に役立ててください")
image = Image.open("./images/headerassetallocation.png")
st.image(image)
st.caption("ティッカーコードをカンマ区切りスペース無しで入力")

inputassets = st.text_input("stooqのティッカーコード:", value="^SPX,^NKX,GOLD.US,1345.JP,IYR.US,10YUSY.B")
inputassetsjp = st.text_input("対応する日本語列名:", value="S&P500,日経225,金,東証REIT,米国不動産,米国10年債")

col1, col2 = st.columns(2)
with col1:
    start = st.date_input("評価開始日:", value=startday)
    inputamount = st.number_input("投資額:", value=100)
with col2:
    end = st.date_input("評価終了日:", value=today)
    
#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #db7093;  /* 背景色 */
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
btnExec = st.button("実行")

if btnExec:
    pf_data = get_asset_matrix(inputassets.split(","),inputassetsjp.split(","), start=start,end=end)
    chartdata =(pf_data / pf_data.iloc[0] * inputamount)
    #日次ログリターン
    log_returns = np.log(pf_data / pf_data.shift(1))
    #チャート
    pltasset = plot_chart(chartdata, "各資産の推移")
    #
    st.subheader("同金額を投資した場合の資産推移")
    st.pyplot(pltasset)
    st.subheader("各資産の関連性（相関）")
    st.write(log_returns.corr())
    st.subheader("同方向移動性(共分散)")
    st.write(log_returns.cov() * business_days)
    st.subheader("平均リターン")
    df = pd.DataFrame(log_returns.mean() * 250)
    df.columns = ["平均リターン"]
    st.write(df)