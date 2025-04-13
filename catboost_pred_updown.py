import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from catboost import CatBoostClassifier, Pool
from PIL import Image



#差文データ前処理
def data_pre(df, col):
    df = pd.concat([df, df[col].diff(1).rename(f"{col}_diff1"), 
               df[col].diff(1).shift(1).rename(f"{col}_diff1_1sft"),
               df[col].diff(1).shift(2).rename(f"{col}_diff1_2sft"),
               df[col].diff(1).shift(3).rename(f"{col}_diff1_3sft"),
               df[col].diff(1).shift(4).rename(f"{col}_diff1_4sft"),
               df[col].diff(1).shift(5).rename(f"{col}_diff1_5sft"),
               df[col].diff(1).shift(6).rename(f"{col}_diff1_6sft"),
               df[col].diff(1).shift(7).rename(f"{col}_diff1_7sft"),
               df[col].diff(1).shift(8).rename(f"{col}_diff1_8sft"),
               df[col].diff(1).shift(9).rename(f"{col}_diff1_9sft"),
               df[col].diff(1).shift(10).rename(f"{col}_diff1_10sft")],
              axis=1) 
    return df

@st.cache_data
def stock_data_1mo(code="^N225"):
    df = yf.download(code, start=datetime.datetime.now() - datetime.timedelta(30),progress=False)
    df.columns = [col[0] for col in df.columns]
    #正解列作成
    df["Tgt1"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.drop(["Volume"], axis=1)

    df = data_pre(df, "High")
    df = data_pre(df, "Low")
    df = data_pre(df, "Open")
    df = data_pre(df, "Close")    
    return df

#モデルの読み込み
loadmodel = CatBoostClassifier()
loadmodel.load_model('./models/TrumpUpDown.cbm')

df = stock_data_1mo("^N225")
imageUp = Image.open("./images/up.jpg")
imageDown = Image.open("./images/down.jpg")

###################################################
st.title("CatBoostで株価UpDown予測")

st.text("CatBoostで日経225株価が終値ベースで明日上がるか下がるかを予測")
image = Image.open("./images/catboost.png")
st.image(image)
st.caption("モデルの更新は3か月ごと実施する予定です！")

#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #0000ff;  /* 背景色 */
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
btnPush = st.button("押すだけ！")

if btnPush:
    X_test = df.iloc[-1, 5:]
    #予測
    pred = loadmodel.predict(X_test)
    st.subheader("明日の日経225平均株価予想")

    if pred==1:
        st.image(imageUp)
    else:
        st.image(imageDown) 