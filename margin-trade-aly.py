import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image

#信用倍率取り込み用

def get_urldata_yf(url, pages=1):
    #データ格納用配列
    data = []
    
    for page in range(1,pages + 1):
        #ページの読み込み
        load_url = url + str(page)
        #print(load_url)
        html = requests.get(load_url)
        soup = BeautifulSoup(html.text, "html.parser")
        #ヘッダーの処理
        tblHead = soup.find_all("thead")
        headers = tblHead[0].find_all("th", attrs={'class':'RankingTable__head__2mLL'})
        #列名を変数に
        col1 = headers[0].text
        col2 = headers[1].text.split('・')[1]
        col3 = headers[1].text.split('・')[0]
        col4 = headers[1].text.split('・')[2]
        col5 = headers[2].text
        col6 = headers[3].text
        col7 = headers[4].text
        col8 = headers[5].text
        col9 = headers[6].text
        #データテーブル抽出
        tblData = soup.find_all("tr", class_ = "RankingTable__row__1Gwp")
        #一行ずつ走査
        for tblRow in tblData:
            #１件分辞書型の初期化
            datarow = {}
            #順位
            datarow[col1] = int(tblRow.th.text)
            #コード
            datarow[col2] = tblRow.li.text
            #名称
            datarow[col3] = tblRow.a.text
            #市場
            datarow[col4] = tblRow.find_all("li")[1].text
            #終値、買残、増減、売残、信用倍率
            tblRowCells = tblRow.find_all("span", class_ = "StyledNumber__value__3rXW")
            datarow[col5] = float(tblRowCells[0].text.replace(',',""))
            datarow[col6] = int(tblRowCells[1].text.replace(',',"").replace('---', '0'))
            datarow[col7] = int(tblRowCells[2].text.replace(',',""))
            datarow[col8] = int(tblRowCells[3].text.replace(',',"").replace('---', '0'))
            datarow[col9] = float(tblRowCells[4].text.replace(',',"").replace('---', '0'))
            #データ一件追加
            data.append(datarow)
    #データフレームに入れる       
    df = pd.DataFrame(data)
    return df


st.title("信用残分析")

st.text("信用残を使った買い分析を行います。")
image = Image.open("headermoneymoney.png")
st.image(image)
st.caption("データは毎週１回、金曜日のみ更新です")

col1, col2 = st.columns(2)
with col1:
    marketgrp = st.selectbox("市場を選択:", ["東証ALL | tokyoAll", "東証PRM | tokyo1", "東証STD | tokyo2", "東証GRT | tokyoM"], 1)[8:]
with col2:
    pages = st.selectbox("取得ページ数:", [x for x in range(1, 5)], 2)

######################################
urlkaiz = f"https://finance.yahoo.co.jp/stocks/ranking/creditBuybackIncrease?market={marketgrp}&term=weekly&page="
urlurig = f"https://finance.yahoo.co.jp/stocks/ranking/creditShortfallDecrease?market={marketgrp}&term=weekly&page="
urlkaig = f"https://finance.yahoo.co.jp/stocks/ranking/creditBuybackDecrease?market={marketgrp}&term=weekly&page="
urluriz = f"https://finance.yahoo.co.jp/stocks/ranking/creditShortfallIncrease?market={marketgrp}&term=weekly&page="
######################################

st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #da70d6;  /* 背景色 */
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

col1, col2 = st.columns(2)
with col1:
    btnbuy = st.button("買候補取得")
with col2:
    btnsell = st.button("売候補取得")

if btnbuy:
    data_lode_state = st.text("読み込み中・・・")
    dfkaizou = get_urldata_yf(urlkaiz, pages)
    dfurigen = get_urldata_yf(urlurig, pages)
    dfcommonkai = pd.merge(dfurigen, dfkaizou[["コード", "順位"]], on=['コード', 'コード'], how='inner')
    dfcommonkai["順位和"] = dfcommonkai["順位_x"] + dfcommonkai["順位_y"]
    dfcommonkai = dfcommonkai[['コード', '名称', '市場', '終値', '買残', '増減', '売残', '信用倍率','順位_x', '順位_y','順位和']]
    dfcommonkai = dfcommonkai.rename(columns={'順位_x': '信用売減順位', '順位_y': '信用買残順位'})
    dfcommonkai.sort_values(by='順位和', ascending=True, inplace=True)
    data_lode_state.text("")
    st.subheader("買い候補")
    st.dataframe(dfcommonkai)
    
if btnsell:
    data_lode_state = st.text("読み込み中・・・")
    dfkaigen = get_urldata_yf(urlkaig, pages)
    dfurizou = get_urldata_yf(urluriz, pages) 
    dfcommonuri = pd.merge(dfkaigen, dfurizou[["コード", "順位"]], on=['コード', 'コード'], how='inner')
    dfcommonuri["順位和"] = dfcommonuri["順位_x"] + dfcommonuri["順位_y"]
    dfcommonuri = dfcommonuri[['コード', '名称', '市場', '終値', '買残', '増減', '売残', '信用倍率','順位_x', '順位_y','順位和']]
    dfcommonuri = dfcommonuri.rename(columns={'順位_x': '買残減順位', '順位_y': '売残増順位'})
    dfcommonuri.sort_values(by='順位和', ascending=True, inplace=True)
    data_lode_state.text("")
    st.subheader("売り候補")
    st.dataframe(dfcommonuri)    