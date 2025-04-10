import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import re
import numpy as np
from PIL import Image




def handle_selection():
    st.write("セレクト")
    st.write(dfview.id)




st.title("アップロードファイルからチャート描画")

st.text("CSVファイルからチャート表示")
image = Image.open("headermoneymoney.png")
st.image(image)
st.caption("自分の調べたい銘柄をCSVファイルでアップしてチャート表示")

upfile = st.file_uploader("CSVファイルをアップロード(ドラッグアンドドロップ出来ます)", type="csv")
df = pd.DataFrame()
dfview = st.dataframe(df)



if upfile:
    df = pd.read_csv(upfile)
    dfview.dataframe(df, on_select=handle_selection, selection_mode="single-row")
