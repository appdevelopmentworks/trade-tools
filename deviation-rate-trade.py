import streamlit as st
import yfinance as yf
import mplfinance as mpf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import datetime
import ta
#from tqdm import tqdm

#乖離率データ取得関数
@st.cache_data
def get_urldata_kairi(url, pages=1):
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
            datarow[col6] = float(tblRowCells[1].text.replace(',',"").replace('---', '0'))
            datarow[col7] = float(tblRowCells[2].text.replace(',',""))
            #データ一件追加
            data.append(datarow)
    #データフレームに入れる       
    df = pd.DataFrame(data)
    return df

def search_url(marketgrp):
    if marketgrp=="tokyo1":
        url = "https://finance.yahoo.co.jp/stocks/ranking/highSeparationRate25minus?market=tokyo1&term=daily&page="
    elif marketgrp=="tokyo2":
        url = "https://finance.yahoo.co.jp/stocks/ranking/highSeparationRate25minus?market=tokyo2&term=daily&page="
    else:
        url = "https://finance.yahoo.co.jp/stocks/ranking/highSeparationRate25minus?market=tokyoM&term=daily&page="
    return url

@st.cache_data    
def candidate_to_buy(dfkairi):
    start = datetime.datetime.now() - datetime.timedelta(250)
    data = {'コード':[], '銘柄名':[], '市場':[], '取引値':[], '乖離率(%)':[], 'MACDヒストグラム':[], 'RSI':[]}

    for row in dfkairi.values:
        try:
            df = yf.download(row[1] + ".T", start=start, progress=False)
            df.columns = [col[0] for col in df.columns]
            if len(df)==0:
                continue         
            #MACD
            MACD = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD_Hist'] = MACD.macd_diff()
            #RSI
            df["RSI"] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()     
            #if df["RSI"].iloc[-1] >= df["RSI"].iloc[-2]:            
            if df["MACD_Hist"].iloc[-1] >= df["MACD_Hist"].iloc[-2]:
                data["コード"].append(row[1])
                data["銘柄名"].append(row[2])
                data["市場"].append(row[3])
                data["取引値"].append(row[4])
                data["乖離率(%)"].append(row[6])
                data["MACDヒストグラム"].append(df["MACD_Hist"].iloc[-1])
                data["RSI"].append(df["RSI"].iloc[-1])
                print(row[1])
        except:
            print(row[1], "でエラー発生")
    return pd.DataFrame(data)

def get_data_mpf_plot(ticker):
    shortname = yf.Ticker(ticker + ".T").info["shortName"]
    title = ticker + " " + shortname
    df = yf.download(ticker + ".T", start=datetime.datetime.now() - datetime.timedelta(100), progress=False)
    df.columns = [col[0] for col in df.columns]
    
    if not df.empty:
        return -1
        
    df["SMA_S"] = ta.trend.SMAIndicator(df['Close'], window=5).sma_indicator()
    df["SMA_M"] = ta.trend.SMAIndicator(df['Close'], window=25).sma_indicator()
    df["SMA_L"] = ta.trend.SMAIndicator(df['Close'], window=60).sma_indicator()
    
    df['MACD'] = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['MACD_Hist'] = ta.trend.MACD(df['Close']).macd_diff()
    
    df['RSI_F'] = ta.momentum.RSIIndicator(df['Close'], window=5).rsi()
    df['RSI_S'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['70'] = [70 for _ in df['Close']]
    df['30'] = [30 for _ in df['Close']]

    apd = [mpf.make_addplot(df['SMA_S'], color='blue', panel=0, width=0.5),
           mpf.make_addplot(df['SMA_M'], color='red', panel=0, width=0.5),
           mpf.make_addplot(df['SMA_L'], color='green', panel=0, width=0.5),
           
           mpf.make_addplot(df['MACD'], panel=2, color='red', width=0.5),
           mpf.make_addplot(df['MACD_Signal'], panel=2, color='blue', width=0.5),
           mpf.make_addplot(df['MACD_Hist'], panel=2, type='bar'),
           
           mpf.make_addplot(df['RSI_F'], panel=3, color='red', width=0.5),
           mpf.make_addplot(df['RSI_S'], panel=3, color='blue', width=0.5),
           mpf.make_addplot(df['70'], panel=3, color='green', width=0.5),
           mpf.make_addplot(df['30'], panel=3, color='green', width=0.5),
            ]

    fig, axes = mpf.plot(df, type='candle', figratio=(3,2),style='yahoo' , addplot=apd, title=title ,returnfig=True, volume=True)
    
    axes[0].legend(["SMA_L","SMS_S","SMA_M"])
    axes[3].legend(["MACD","SIGNAL"])
    axes[6].legend(["RSI_F","RSI_S"])
    
    fig.show()

######################################################################
st.title("乖離率トレード")

st.caption("小手川隆さんの手法をプログラムにしました。")
image = Image.open("headermoneymoney.png")
st.image(image)
st.caption("下げ相場でも買い候補を探すため、ヒットしない場合があります。")

col1, col2 = st.columns(2)
with col1:
    marketgrp = st.selectbox("市場を選択:", ["東証PRM | tokyo1", "東証STD | tokyo2", "東証GRT | tokyoM"], 0)[8:]
    chkcal = st.checkbox("買い候補算出")
with col2:
    pages = st.selectbox("取得ページ数:", [x for x in range(1, 5)], 0)
    #chkchart = st.checkbox("チャート表示")


#書式付きボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 30%;
        background-color: #6b8e23;  /* 背景色 */
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
btnkairi =st.button("データ取得")

if btnkairi:
    data_lode_state = st.text("読み込み中・・・")
    dfkairi = get_urldata_kairi(search_url(marketgrp), int(pages))
    data_lode_state.text("読み込み完了！")
    st.dataframe(dfkairi)
    data_lode_state.text("計算中！")
    if chkcal:
        dfbuy = candidate_to_buy(dfkairi)
        data_lode_state.text("完了！")
        st.subheader("買い候補リスト")
        st.dataframe(dfbuy)
    # if chkchart:
    #     print("Chart Plot")

        
    


