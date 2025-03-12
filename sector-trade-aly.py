import streamlit as st
import yfinance as yf
import pandas as pd
from PIL import Image


def getsectorrank():
    #セクター別騰落ＵＲＬ
    urlsecter = "https://kabutan.jp/warning/?mode=9_1&market=0&capitalization=-1&stc=zenhiritsu&stm=1&col=zenhiritsu&page="

    for page in range(1, 4):
        # df = pd.read_html(urlsecter)
        # df = df[2]
        load_url = urlsecter + str(page)
        dfw = pd.read_html(load_url)
        dfw = dfw[2]
        if page ==1:
            df = dfw.copy()
        else:
            df = pd.concat([df, dfw], ignore_index=True)
            
    df = df[['コード', '銘柄名', '銘柄数', '株価','前日比', '前日比.1', 'ＰＥＲ', 'ＰＢＲ', '利回り']]
    return df.rename(columns={"前日比.1":"前日比(%)"})

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

st.title("セクター別銘柄選定")

st.text("強いセクターから銘柄選定します")
image = Image.open("headermoneymoney.png")
st.image(image)
st.caption("データは毎朝9:15以降に更新されます!")
st.subheader("セクター別ランキング：")

dfrank = getsectorrank()
st.dataframe(dfrank)

options = []
for code, name in zip(dfrank["コード"], dfrank["銘柄名"]):
    data = str(code) + "|" + name
    options.append(data)

sector = st.selectbox("セクターを選択", options, 0)[:3]

seccnt = dfrank[dfrank["コード"]==int(sector)].iloc[0,2]
                    
st.subheader("セクターランキング：")
#
pagelstvew = 15

seccode = int(sector) - 250
pages =  (seccnt + pagelstvew-1) // pagelstvew

urls = f"https://kabutan.jp/themes/?industry={seccode}&market=1&capitalization=-1&stc=&stm=1&col=zenhiritsu&page="

dfsctr = getsectordata(urls, pages)
st.dataframe(dfsctr)


