import streamlit as st
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import datetime
import re
from PIL import Image

def checkTicker(ticker):
    # 有効な英大文字を定義
    valid_letters = "ACDFGHJKLMPNRSTUWX-Y"
    # 正規表現パターン
    pattern = rf"^[0-9][0-9{valid_letters}][0-9][0-9{valid_letters}]$"
    if not re.match(pattern, ticker):
        return ticker
    else:
        return ticker + ".T"

@st.cache_data
def get_train_data(code, start, end):
    df = yf.download(code, start=start, end=end)
    df.columns = [col[0] for col in df.columns]
    #実体長さ
    df["Bodysize"] = abs(df["Close"] - df["Open"])
    #前日終値
    df["PrevClose"] = df["Close"].shift(1)
    #上髭長さ
    df["Uhige"] = [h - max(o, c) for o, c, h in zip(df["Open"], df["Close"], df["High"])]
    #下髭長さ
    df["Lhige"] = [min(o, c)-l for o, c, l in zip(df["Open"], df["Close"], df["Low"])]
    #前日との終値差
    df["DefClose"] = df["Close"] - df["PrevClose"]
    #実体サイズの相対値:
    df["RVAS"]= df["Bodysize"] / df["PrevClose"]
    df["ScRVAS"] = (df["RVAS"] - df["RVAS"].mean()) / df["RVAS"].std()
    #上ヒゲ/実体サイズ
    df["Uhigeratio"] = df["Uhige"] / df["Bodysize"]
    #下ヒゲ/実体サイズ
    df["Lhigeratio"] = df["Lhige"] / df["Bodysize"]
    #ボラティリティ
    df["Volatility"] = (df["High"] - df["Low"]) / df["PrevClose"]
    #値動き
    df["PriceMV"] = df["Close"] - df["Open"]
    #曜日
    df["WeekDay"] = df.index.dayofweek    
    #Target
    df["Target"] = df["PriceMV"].shift(-1)
    #作業列削除
    del df["RVAS"], df["PriceMV"]
    
    return df


st.title("ローソク足分析予測")

st.text("LightGBMを使ってローソク足から予測モデル作成")
image = Image.open("headermoneymoney.png")
st.image(image)
st.caption("予測データは上がりそうか下がれそうかの目安として使ってください")

col1, col2 = st.columns(2)

with col1:
    #ティッカー（初期値は、日経225、金/ドル(GC=F)）
    ticker = st.text_input("コードを入力:", value="^N225")
    #startday = pd.to_datetime("2020-01-01")
    startday = datetime.date.today() - datetime.timedelta(7)
    startlearn = st.date_input("学習開始日", value=startday)
with col2:
    today = datetime.date.today()
    predday = st.date_input("予測したい日", value=today + datetime.timedelta(1))
    endlearn = st.date_input("学習終了日", value=today)
     

#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 30%;
        background-color: #008080;  /* 背景色 */
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
btnstart = st.button("データ取得＆学習")
    
if btnstart:
    data = get_train_data(checkTicker(ticker), startlearn, endlearn + datetime.timedelta(1))
    st.dataframe(data[1:-1],height=200)
    #X = data[1:-1][['Close', 'High', 'Low', 'Open', 'Bodysize', 'PrevClose', 'Uhige','Lhige', 'DefClose', 'ScRVAS', 'Uhigeratio', 'Lhigeratio', 'Volatility']]
    X = data[1:-1][['Close', 'High', 'Low', 'Open', 'Bodysize', 'PrevClose', 'Uhige','Lhige', 'DefClose', 'ScRVAS', 'Uhigeratio', 'Lhigeratio', 'Volatility', 'WeekDay']]
    y = data[1:-1]['Target']
    #LightGBM
    train_data = lgb.Dataset(X, label=y)
    #パラメータ（要チューニング）
    params = {'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt'}
    #学習
    model = lgb.train(params, train_data, num_boost_round=100)
    #当日のデータで明日の変動幅を予測
    #X_test = data[-1:][['Close', 'High', 'Low', 'Open', 'Bodysize', 'PrevClose', 'Uhige','Lhige', 'DefClose', 'ScRVAS', 'Uhigeratio', 'Lhigeratio', 'Volatility']]
    #X_test = data[-1:][['Close', 'High', 'Low', 'Open', 'Bodysize', 'PrevClose', 'Uhige','Lhige', 'DefClose', 'ScRVAS', 'Uhigeratio', 'Lhigeratio', 'Volatility', 'WeekDay']]
    currday = predday - datetime.timedelta(1)
    X_test = data.loc[currday:currday][['Close', 'High', 'Low', 'Open', 'Bodysize', 'PrevClose', 'Uhige','Lhige', 'DefClose', 'ScRVAS', 'Uhigeratio', 'Lhigeratio', 'Volatility', 'WeekDay']]
    
    predictions = model.predict(X_test)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader(f"{predday}の [終値 - 始値]")
    with col4:
        st.subheader(predictions)