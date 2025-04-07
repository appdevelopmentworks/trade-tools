import streamlit as st
import yfinance as yf
import pandas as pd
from PIL import Image

#
def checkTicker(ticker):
    # 有効な英大文字を定義
    valid_letters = "ACDFGHJKLMPNRSTUWX-Y"
    # 正規表現パターン
    pattern = rf"^[0-9][0-9{valid_letters}][0-9][0-9{valid_letters}]$"
    if not re.match(pattern, ticker):
        return ticker
    else:
        return ticker + ".T"

#調査するシートをデータフレームにして返す
def get_stock_list(stocklist):
    diclst = {"日経225":"N225", "日経500":"N500", "JPX400":"JPX400", "読売333":"Y333", "S&P500":"SP500"}
    return pd.read_excel("./Stocklist.xlsx", sheet_name=diclst[stocklist])


def get_sort_beard(dfslst, stocklist, sortcol):
    #
    if stocklist=="S&P500":
        stklst = yf.Tickers(dfslst["コード"].to_list())
    else:
        stklst = yf.Tickers([str(x) + ".T" for x in dfslst["コード"]])
    #
    data = {"コード":[],"銘柄名":[],"前日比":[], "陽陰":[], "胴体":[], "下髭(%)":[], "上髭(%)":[], "下髭全体比":[],}
    totalcnt = len(dfslst)
    cnt = 0
    
    for tikr, row in zip(stklst.tickers, dfslst.values):
        #プログレスバーの処理
        cnt += 1
        prgval = int(cnt / totalcnt * 100)
        prgbar.progress(prgval, text=f"処理中... {prgval}%")
        try:
            cbody = abs(stklst.tickers[tikr].info["ask"] - stklst.tickers[tikr].info["open"])
            crange = stklst.tickers[tikr].info["dayHigh"] - stklst.tickers[tikr].info["dayLow"]
            yo_in = int(stklst.tickers[tikr].info["ask"] >= stklst.tickers[tikr].info["open"])
            pre_ask = stklst.tickers[tikr].info["ask"] - stklst.tickers[tikr].info["previousClose"]
            lhige = min(stklst.tickers[tikr].info["ask"], stklst.tickers[tikr].info["open"]) - stklst.tickers[tikr].info["dayLow"]
            hhige = stklst.tickers[tikr].info["dayHigh"] - max(stklst.tickers[tikr].info["ask"], stklst.tickers[tikr].info["open"])
            lhigetotalratio = lhige / crange
            
            if cbody==0:
                lhigeratio = round(lhige / crange *100, 2)
                hhigeratio = round(hhige / crange *100, 2)
            elif lhige==0 and hhige==0:
                print(code)
                lhigeratio=0
                hhigeratio=0
            else:
                lhigeratio = round(lhige / cbody *100, 2)
                hhigeratio = round(hhige / cbody *100, 2)
            #
            data["コード"].append(str(row[0])) 
            data["銘柄名"].append(row[1])
            data["前日比"].append(pre_ask)
            data["陽陰"].append(yo_in)
            data["胴体"].append(cbody)
            data["下髭(%)"].append(abs(lhigeratio))
            data["上髭(%)"].append(abs(hhigeratio))
            data["下髭全体比"].append(abs(round(lhigetotalratio, 2)))
        except KeyError:
            print(row[0], "無効なインデックス")
            
    prgbar.progress(100, text="完了！")
    return pd.DataFrame(data).sort_values(by=sortcol, ascending=False)



###########################################################################################
st.title("ローソクのヒゲ分析")

st.caption("ローソク足のヒゲの長さを算定（15分間隔で更新）")
image = Image.open("headerami.png")
st.image(image)
st.caption("下ヒゲの長さから反転銘柄を探します")

col1, col2 = st.columns(2)
with col1:
    stocklist = st.selectbox("銘柄リスト種類:", ["日経225", "日経500", "JPX400", "S&P500"], 0)
with col2:
    sortcol = st.selectbox("ソート項目:", ["陽陰", "胴体", "下髭(%)", "上髭(%)", "下髭全体比"], 4)

#
dfslst = get_stock_list(stocklist)

#
st.subheader(stocklist)
st.dataframe(dfslst, width=1000, height=250)
#

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
btncal = st.button("計算")
prgbar = st.progress(0, "")

if btncal:
    resdf = get_sort_beard(dfslst, stocklist, sortcol)
    st.subheader(stocklist + "の下ヒゲ分析結果")
    st.dataframe(resdf, width=1000)
    