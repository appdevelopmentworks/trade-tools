import streamlit as st
from  pandas_datareader import data as web
import yfinance as yf
import pandas as pd
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from PIL import Image
#自作ライブラリー
from mystock_lib import *


#セクターのランキング取得
def getsectorrank(urlsecter):
    for page in range(1, 4):
        load_url = urlsecter + str(page)
        dfw = pd.read_html(load_url)
        dfw = dfw[2]
        if page ==1:
            df = dfw.copy()
        else:
            df = pd.concat([df, dfw], ignore_index=True)
            
    df = df[['コード', '銘柄名', '銘柄数', '株価','前日比', '前日比.1', 'ＰＥＲ', 'ＰＢＲ', '利回り']]
    return df.rename(columns={"前日比.1":"前日比(%)"})

#セクター内銘柄を取得
def getsectordata(url, pages):
    for page in range(1, pages+1):
        load_url = urls + str(page)
        dfw = pd.read_html(load_url)
        dfw = dfw[2]
        if page ==1:
            dfs = dfw.copy()
        else:
            dfs = pd.concat([dfs, dfw], ignore_index=True)
            
    dfs = dfs[['コード', '銘柄名', '市場', '株価', '前日比', '前日比.1', 'ＰＥＲ', 'ＰＢＲ', '利回り']]
    return dfs.rename(columns={"前日比.1":"前日比(%)"})

@st.cache_data
def addinfdata(df):
    tickers = yf.Tickers([str(x) + ".T" for x in df["コード"]])
    data = {'コード':[], '前日終値':[], '始値':[], '現在値':[], '現在安値':[], '現在高値':[], '現在出来高':[], '平均出来高3M':[], '平均出来高10D':[]}

    for ticker in tickers.tickers:
        if tickers.tickers[ticker].info["typeDisp"]=="Fund":
            continue 
        data["コード"].append(ticker[:4])
        data["前日終値"].append(tickers.tickers[ticker].info["previousClose"])
        data["始値"].append(tickers.tickers[ticker].info["open"])
        data["現在値"].append(tickers.tickers[ticker].info["ask"])
        data["現在安値"].append(tickers.tickers[ticker].info["dayLow"])
        data["現在高値"].append(tickers.tickers[ticker].info["dayHigh"])
        data["現在出来高"].append(tickers.tickers[ticker].info["volume"])
        data["平均出来高3M"].append(tickers.tickers[ticker].info["averageVolume"])
        data["平均出来高10D"].append(tickers.tickers[ticker].info["averageDailyVolume10Day"])
    
    dfnew = pd.DataFrame(data)
    dfnew["当日上昇率"] = (dfnew["現在値"] - dfnew["始値"]) / dfnew["始値"] *100
    dfnew["前日上昇率"] = (dfnew["現在値"] - dfnew["前日終値"]) / dfnew["前日終値"] * 100
    dfnew["出来高上昇率3M"] = dfnew["現在出来高"] / dfnew["平均出来高3M"] -1
    dfnew["出来高上昇率10D"]= dfnew["現在出来高"] / dfnew["平均出来高10D"] -1
    dfnew.insert(1, '銘柄名', df["銘柄名"])
    dfnew.insert(14, '利回り', df["利回り"])
    dfnew.insert(14, 'PBR', df["ＰＢＲ"])
    dfnew.insert(14, 'PER', df["ＰＥＲ"])    
    return dfnew

@st.cache_data
def get_relative_data(tickers):
    end =  datetime.datetime.now()
    start = end - datetime.timedelta(50)
    data = pd.DataFrame()
    for ticker in tickers:
        data[ticker] = web.DataReader(ticker, 'stooq', start=start, end=end)["Close"]
    data = data.sort_index(ascending=True)
    #日時で何パーセント上昇したか、Na行は削除
    returns = data.pct_change().dropna()
    # 業種内の平均リターンを計算
    industry_average = returns.mean(axis=1)
    # 各株式のレラティブストレングスを計算
    relative_strength = returns.sub(industry_average, axis=0)
    return relative_strength

#チャート描画関数
def plot_relative_chart(df):
    plt.figure(figsize=(12, 6))
    for ticker in df.columns:
        plt.plot(df.index, df[ticker], label=ticker)
    
    plt.title('レラティヴ・ストレングス')
    plt.xlabel('日付')
    plt.ylabel('Relative Strength')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)  # 基準線
    plt.legend(loc='upper left')
    plt.grid()
    return plt

st.title("セクター別銘柄 Relative Strength")

st.text("強いセクターの銘柄からレラティヴストレングス")
image = Image.open("./images/headersecrelative.png")
st.image(image)
st.caption("データは毎朝9:15以降に更新されます!")
st.subheader("セクター別ランキング：")

#セクターのランキングを表示
urlsecter = "https://kabutan.jp/warning/?mode=9_1&market=0&capitalization=-1&stc=zenhiritsu&stm=1&col=zenhiritsu&page="
dfrank = getsectorrank(urlsecter)

col1, col2 = st.columns(2)
with col1:
    #セレクトボックス
    options = []
    for code, name in zip(dfrank["コード"], dfrank["銘柄名"]):
        data = str(code) + "|" + name
        options.append(data)
    sector = st.selectbox("セクターを選択", options, 0)[:3]
    #選択されているセクターコードを変数に代入
    seccnt = dfrank[dfrank["コード"]==int(sector)].iloc[0,2]
                        
    #1ページに何行表示にされているか
    pagelstvew = 15
    #Webページ用のコードに変換
    seccode = int(sector) - 250
    #行数から取得ページ数を算定
    pages =  (seccnt + pagelstvew-1) // pagelstvew

    #セクター内でデータ取得
    #st.subheader("セクターランキング：")
    urls = f"https://kabutan.jp/themes/?industry={seccode}&market=1&capitalization=-1&stc=&stm=1&col=zenhiritsu&page="
    dfsctr = getsectordata(urls, pages)
    #st.dataframe(dfsctr)
with col2:
    numacq = st.number_input("チャート表示件数", value=10, max_value=180)
    
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #191970;  /* 背景色 */
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
btaddinf = st.button("セクター内情報")
    
    
if btaddinf:
    data_lode_state = st.text("読み込み中・・・")
    dfaddinf = addinfdata(dfsctr)
    tickers = [checkTicker_stooq(ticker) for ticker in dfaddinf["コード"].head(numacq)]
    dfrelative = get_relative_data(tickers)
    pltrelative = plot_relative_chart(dfrelative)
    data_lode_state.text("読み込み完了!")
    st.subheader("同業種内の相対的強さ:")
    st.pyplot(pltrelative)
    st.subheader("セクター内ランキング:")
    st.dataframe(dfaddinf)

    