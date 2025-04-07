import streamlit as st
import yfinance as yf
import pandas as pd
from PIL import Image
import datetime
import re
import ta


def checkTicker(ticker):
    # 有効な英大文字を定義
    valid_letters = "ACDFGHJKLMPNRSTUWX-Y"
    # 正規表現パターン
    pattern = rf"^[0-9][0-9{valid_letters}][0-9][0-9{valid_letters}]$"
    if not re.match(pattern, ticker):
        return ticker
    else:
        return ticker + ".T"
    
#エクセルbookから該当リストをデータフレームで取得
def get_meigaralst(name):
    diclst = {"日経225":"N225", "日経500":"N500", "JPX400":"JPX400", "読売333":"Y333", "S&P500":"SP500"}
    #
    dfnikkei = pd.read_excel("./Stocklist.xlsx", sheet_name=diclst[name])
    return dfnikkei

#ボリンジャーのσとRSIを計算し売られすぎ（σが小さい順）リストをデータフレームで返す
@st.cache_data
def cal_bbandrsi_data(dflist, bbspan=20, rsispan=14, start = datetime.datetime.now() - datetime.timedelta(40)):
    totalcnt = len(dflist)
    cnt = 0
    #
    data = {"コード":[], "銘柄名":[], "業種":[], "標準偏差":[],"Z値":[],"RSI":[]}
    #
    for row in dflist.values:
        #株価データダウンロード60日分
        #print(row[0])
        df = yf.download(checkTicker(str(row[0])), start=start, progress=False)
        df.columns = [col[0] for col in df.columns]
        #プログレスバーの処理
        cnt += 1
        prgval = int(cnt / totalcnt * 100)
        prgbar.progress(prgval, text=f"処理中... {prgval}%")
        #データ無かったら次へ
        if len(df)==0:
            continue    
        #移動平均
        df["MABB"] = df["Close"].rolling(window=bbspan).mean()
        #標準偏差
        df["std"] = df["Close"].rolling(window=bbspan).std()
        #2σバンド
        df["-2σ"] = df["MABB"] - 2*df["std"]
        df["+2σ"] = df["MABB"] + 2*df["std"]
        #その値は何σにあたるか
        df["siguma"] = (df["Close"] - df["MABB"]) / df["std"] 
        #RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsispan).rsi()
        #書き出し
        data["コード"].append(str(row[0]))
        data["銘柄名"].append(row[1])
        data["業種"].append(row[3])
        data["標準偏差"].append(df["std"].iloc[-1])
        data["Z値"].append(df["siguma"].iloc[-1])
        data["RSI"].append(df["RSI"].iloc[-1])
    
    return pd.DataFrame(data).sort_values(by="Z値", ascending=True)




####################################################################################
st.title("ボリンジャー＆RSIで探す！")

st.caption("ボリンジャーバンドとRSIを使って売り込まれた銘柄を探します")
image = Image.open("headermoneymoney.png")
st.image(image)
st.caption("ボリンジャの-σが小さい順にソートしています。")

col1, col2 = st.columns(2)
with col1:
    bbspan = int(st.text_input("ボリンジャーバンド期間", value=20))
with col2:
    rsispan = int(st.text_input("RSI期間", value=14))
    nikkeilst = st.selectbox("銘柄リスト種類:", ["日経225", "日経500", "JPX400", "S&P500"], 0)

#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 30%;
        background-color: #6495ed;  /* 背景色 */
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
btncal = st.button("計算")
prgbar = st.progress(0, "")


if btncal:
    dfnikkei =get_meigaralst(nikkeilst)
    st.subheader(nikkeilst + "銘柄リスト：")
    st.dataframe(dfnikkei, width=1000)
    dfbuy = cal_bbandrsi_data(dfnikkei, bbspan, rsispan, datetime.datetime.now() - datetime.timedelta(30))
    st.subheader("売られすぎ順表示：")
    st.dataframe(dfbuy, width=1000)