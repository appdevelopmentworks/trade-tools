import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas_datareader.data as web
import datetime
from PIL import Image


############################################################################
today = datetime.date.today()
startday =  datetime.date(2020, 1, 6)
business_days = 250

############################################################################

@st.cache_data
def get_asset_matrix(assets, colname ,start, end):
    pf_data = pd.DataFrame()
    for ticker in assets:
        pf_data[ticker] = web.DataReader(ticker, 'stooq', start=start, end=end)["Close"]
    pf_data.columns = colname
    return pf_data.sort_index(ascending=True)

@st.cache_data
def get_portfolios(assets, log_returns, inputattempts=1000):
    num_assets = len(assets.split(","))
    #numpy arrayに代入
    pfolio_returns = []
    pfolio_volatilities = []
    pfolio_allocation = []

    for x in range(inputattempts):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        pfolio_returns.append(np.sum(weights * log_returns.mean()) * business_days)
        pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * business_days, weights))))
        pfolio_allocation.append(weights)
    
    dfweights = pd.DataFrame(np.array(pfolio_allocation))
    dfweights.columns = inputassetsjp.split(",")    
    pfolio_returns = np.array(pfolio_returns)
    pfolio_volatilities = np.array(pfolio_volatilities)
    dfreturns = pd.DataFrame({"期待リターン": pfolio_returns, "ボラティリティー": pfolio_volatilities})
    portfolios = pd.concat([dfreturns, dfweights], axis=1).sort_values(by=['期待リターン', 'ボラティリティー'], ascending=[False, True])
    return portfolios

def plot_chart(data, title):
    plt.figure(figsize=(10, 5))
    plt.plot(data,label=data.columns)
    plt.title(title)
    plt.xlabel('期間')
    plt.ylabel('資産')
    plt.legend()
    return plt


def plt_scatter(data, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(data.iloc[:,1], data.iloc[:,0])
    plt.title(title)
    plt.xlabel('ボラティリティー(リスク)')
    plt.ylabel('期待リターン')
    return plt


####################################################################################
st.title("資産配分ツール")

st.text("ポートフォリオの構成に役立ててください")
image = Image.open("./images/headerassetallocation.png")
st.image(image)
st.caption("ティッカーコードをカンマ区切りスペース無しで入力")

inputassets = st.text_input("stooqのティッカーコード:", value="^SPX,^NKX,1326.JP,1345.JP,IYR.US,10YUSY.B")
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
    dfpf = get_portfolios(inputassetsjp, log_returns)
    st.subheader("リスク・リターン相関図")
    pltscatter = plt_scatter(dfpf.iloc[:,:2], "リターンとボラティリティー(リスク)")
    st.pyplot(pltscatter)
    st.subheader("最適アセットアロケーション")
    st.dataframe(dfpf)