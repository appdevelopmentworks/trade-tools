import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import re

def checkTicker(ticker):
    # 有効な英大文字を定義
    valid_letters = "ACDFGHJKLMPNRSTUWX-Y"
    # 正規表現パターン
    pattern = rf"^[0-9][0-9{valid_letters}][0-9][0-9{valid_letters}]$"
    if not re.match(pattern, ticker):
        return ticker
    else:
        return ticker + ".T"

#
def geteventdata(ticker, kenrlast, byago):
    df = yf.download(ticker, progress=False)
    df.columns = [col[0] for col in df.columns]
    
    df["Month"] = df.index.month
    df["MonthSft"] = df["Month"].shift(-1)
    df.dropna(inplace=True)
    df["MonthSft"] = df["MonthSft"].astype(int)
    df["権利確定日"] = df["Month"] != df["MonthSft"]
    df["権利付最終日"] = df["権利確定日"].shift(-kenrlast).fillna(False)
    df["買入日終値"] = df["Close"].shift(byago)
    df["リターン(%)"]= (df.Close -  df["買入日終値"]) / df["買入日終値"] * 100
    del df["MonthSft"]
    df.rename(columns={'Close': '終値','High': '高値','Low': '安値','Open': '始値','Volume': '出来高',
                    'Month': '月', '権利確定日': '権利確定日', '権利付最終日': '権利付最終日',
                    '買入日終値': '買入日終値','リターン(%)': 'リターン(%)'}, inplace=True)
    return df

#グラフ表示
def plot_soneki(dfwin, dflose, coname):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Osaka']
    plt.figure(figsize=(10, 5))
    plt.title(label=f"{coname} Event investment return")
    plt.scatter(x=dfwin.index, y=dfwin["リターン(%)"], color="green", label="Win", marker="o")
    plt.scatter(x=dflose.index, y=dflose["リターン(%)"], color="red", label="Lose", marker="x")
    plt.axhline(0, color='blue', linestyle=':', label="Breakeven")
    plt.ylabel("Return(%)")
    plt.legend()
    return plt

st.title("イベント投資期待値算定ツール")

st.text("株主優待・配当の権利確定日を使ったイベント投資に使ってね")
image = Image.open("headereventinv.png")
st.image(image)
st.caption("株主優待・配当の権利付最終日と事前買入リターンを求めます。")

ticker = st.text_input("ティッカーコードを入力:", value="9202")

kenrmonth = st.selectbox("権利確定月(月):", [str(x)+"月末" for x in range(1, 13)], 2) 
buyago = st.slider("買入日(何営業日前)", min_value=1, max_value=60, value=20, step=1)
kenrlast = st.selectbox("決済日【権利付最終日】(日:2営業日前,米:1営業日前):", [1,2], 1) 
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #800000;  /* 背景色 */
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
btncol = st.button("シュミレート")

if btncol:
    coname = yf.Ticker(checkTicker(ticker)).info["shortName"]
    #データセット取得
    data = geteventdata(checkTicker(ticker), int(kenrlast), int(buyago))
    #
    dswin = data[data["権利付最終日"] & (data["リターン(%)"] > 0) & (data["月"]==int(kenrmonth[:-2]))]
    dslose = data[data["権利付最終日"] & (data["リターン(%)"] <= 0) & (data["月"]==int(kenrmonth[:-2]))]
    #勝率
    wincnt = len(dswin)
    losecnt = len(dslose)
    winev = round(dswin["リターン(%)"].mean(),2)
    winmax = round(dswin["リターン(%)"].max(),2)
    loseev = round(dslose["リターン(%)"].mean(),2)
    losemin = round(dslose["リターン(%)"].min(),2)
    winrate = round((wincnt / (wincnt + losecnt)), 2)
    expvalue = (winev * winrate) + (loseev * (1 - winrate))
    #結果表示
    st.subheader("結果:")
    st.pyplot(plot_soneki(dswin, dslose, coname))
    #
    col1, col2 = st.columns(2)
    with col1:
        st.write("勝率: " + str(winrate * 100) + "%")
        st.write("勝数: " + str(wincnt))
        st.write("勝平均リターン: " + str(winev))
        st.write("最大リターン: " + str(winmax))
    with col2:
        st.write("期待値: " + str(round(expvalue, 2)))
        st.write("負数: " + str(losecnt))
        st.write("負平均損失: " + str(loseev))
        st.write("最大損失: " + str(losemin))    
    
    #st.text("勝ちトレード:")
    st.subheader("勝ちトレード:")
    st.dataframe(dswin)
    st.subheader("負けトレード")
    st.dataframe(dslose)
    