import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
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

@st.cache_data
def get_expected_list(code, kenrlast, kenrmonth, buyago):
    df = yf.download(code, progress=False)
    df.columns = [col[0] for col in df.columns]

    data= {"何日前":[],"勝数":[],"負数":[],"勝平均値":[],"勝最大":[],"負平均値":[],"最大損失":[],"勝率":[], "期待値":[]}

    df["Month"] = df.index.month
    #
    df["MonthSft"] = df["Month"].shift(-1)
    df.dropna(inplace=True)
    #
    df["MonthSft"] = df["MonthSft"].astype(int)
    #権利確定日にフラグ立てる
    df["権利確定日"] = df["Month"] != df["MonthSft"]
    #権利確定日のフラグを指定した日数分(kenrlast)ずらす
    df["権利付最終日"] = df["権利確定日"].shift(-kenrlast).fillna(False)
    del df["MonthSft"]

    for buyday in range(1, buyago+1):
        #指定した日数分終値をずらし、引けで買った値段とする
        df["買入日終値"] = df["Close"].shift(buyday)
        #リターンを計算
        df["リターン(%)"]= (df.Close -  df["買入日終値"]) / df["買入日終値"] * 100
        #勝ちトレードのデータセット
        dfwin = df[df["権利付最終日"] & (df["リターン(%)"] > 0) & (df["Month"]==kenrmonth)]
        #負けトレードのデータセット
        dflose =df[df["権利付最終日"] & (df["リターン(%)"] <= 0) & (df["Month"]==kenrmonth)]
        
        wincnt = dfwin.shape[0]
        losecnt = dflose.shape[0]
        winev = round(dfwin["リターン(%)"].mean(),2)
        winmax = round(dfwin["リターン(%)"].max(),2)
        loseev = round(dflose["リターン(%)"].mean(),2)
        losemin = round(dflose["リターン(%)"].min(),2)
        winrate = wincnt / (wincnt + losecnt)
        expvalue = (winev * winrate) + (loseev * (1 - winrate))
        #
        data["何日前"].append(buyday)
        data["勝率"].append(winrate)
        data["期待値"].append(expvalue)
        data["勝数"].append(wincnt)
        data["負数"].append(losecnt)
        data["勝平均値"].append(winev)
        data["勝最大"].append(winmax)
        data["負平均値"].append(loseev)
        data["最大損失"].append(losemin)
        
    return pd.DataFrame(data)

def plot_exp_chart(dfexp, coname, winavg, loseavg, winmax, losemin):    
    # グリッドスペックを作成
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    #
    fig = plt.figure(figsize=(10, 8))
    # 上のサブプロット
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(f"{coname} Benefits and dividend events investment returns")
    ax1.plot(dfexp["何日前"], dfexp["期待値"],linestyle='-', marker='o', label="Expected value")
    if winavg:
        ax1.plot(dfexp["何日前"], dfexp["勝平均値"],linestyle='--', marker='^',color="green",label="winning average")
    if loseavg:
        ax1.plot(dfexp["何日前"], dfexp["負平均値"],linestyle=':', marker='x', color="orange", label="losing average")
    if winmax:
        ax1.plot(dfexp["何日前"], dfexp["勝最大"],linestyle='--', marker='^', color="blue", label="win maximum")
    if losemin:
        ax1.plot(dfexp["何日前"], dfexp["最大損失"],linestyle=':', marker='x', color="red", label="maximum loss")
        
    ax1.axhline(0, color='blue', linestyle=':', label="breakeven")
    ax1.set_ylabel('Return(%)')
    ax1.legend()

    # 下のサブプロット
    ax2 = fig.add_subplot(gs[1])
    #ax2.set_title('下のグラフ')
    ax2.bar(dfexp["何日前"], dfexp["勝率"], color='gray', label='Winning rate(%)')
    ax2.set_ylabel('Winning rate(%)')
    ax2.set_xlabel('How many days ago did I buy it?')
    ax2.legend()

    plt.tight_layout()
    return plt   


######################################################################

st.title("イベント投資最適日算定ツール")

st.text("株主優待・配当狙い投資の最適な投資日を発掘")
image = Image.open("headereventinv.png")
st.image(image)
st.caption("株主優待・配当の権利付最終日の何日前が最適かの意思決定支援")


col1, col2 ,col3= st.columns(3)
with col1:
    ticker = st.text_input("ティッカーコードを入力:", value="8151")
    chkwinavg = st.checkbox("勝平均表示",value=True)
    chkwinemax = st.checkbox("最大リターン表示")
with col2:
    kenrlast = st.selectbox("決済日【権利付最終日】", [1,2], 1) 
    chkloseavg = st.checkbox("負平均表示",value=True)
    chklosemin = st.checkbox("最大損失表示")
with col3:
    kenrmonth = st.selectbox("権利確定月(月):", [str(x)+"月末" for x in range(1, 13)], 2)[:-2]
    #chkwinrate = st.checkbox("勝率表示",value=True)
    chkdisphyou = st.checkbox("確率表表示",value=True)
    
buyago = st.slider("買入日(何営業日前)", min_value=1, max_value=120, value=60, step=1)

st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #4682b4;  /* 背景色 */
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
btncal = st.button("シュミレート")

if btncal:
    #
    coname = yf.Ticker(checkTicker(ticker)).info["shortName"]
    #
    dfex = get_expected_list(checkTicker(ticker), int(kenrlast), int(kenrmonth), int(buyago))
    fig = plot_exp_chart(dfex,coname,chkwinavg,chkloseavg,chkwinemax,chklosemin)
    st.pyplot(fig)
    if chkdisphyou:
        st.dataframe(dfex, width=1000)