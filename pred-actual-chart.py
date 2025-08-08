import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
import ta
import datetime
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from PIL import Image
#自作ライブラリー
from mystock_lib import *

#株式コードと会社名を取得する関数
def get_code_company_name(code):
    coname = yf.Ticker(checkTicker(code)).info["shortName"]
    if code[0]=="^":
        return coname
    else:
        return f"{code} {coname}"

#株価データを取得する関数
@st.cache_data
def get_stock_data(ticker):
    df = yf.download(ticker)
    df.columns = [col[0] for col in df.columns]
    return df

#テクニカル指標をデータセットに追加する関数
@st.cache_data
def get_addtec_data(df, rsif, rsis, bspan, bsigma=2):
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    
    df["MABB"] = ta.trend.sma_indicator(df['Close'], window=bspan)
    df["std"] = df["Close"].rolling(window=bspan).std()
    #ボリンジャーバンド
    df["LBand"] = df["MABB"] - bsigma*df["std"]
    df["UBand"] = df["MABB"] + bsigma*df["std"]
    #Z値
    df["siguma"] = (df["Close"] - df["MABB"]) / df["std"]
    #HMA
    df["HMA5"] = hma(df["Close"], 5)
    #MACD
    MACD = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    
    df['MACD'] = MACD.macd()
    df['MACD_Signal'] = MACD.macd_signal()
    df['MACD_Hist'] = MACD.macd_diff()
    
    #RSI
    df['RSIFast'] = ta.momentum.RSIIndicator(df['Close'], window=rsif).rsi()
    df['RSISlow'] = ta.momentum.RSIIndicator(df['Close'], window=rsis).rsi()
    df['70'], df['30'] = [70 for _ in df["Close"]], [30 for _ in df["Close"]]
    return df

#予測データをデータセットに追加する関数
@st.cache_data
def get_addpred_data(df):
    lowband = df["LBand"]
    Uperband = df["UBand"]
    z = df["siguma"]
     
    overbandL = lowband < df["Close"]
    overbandLshift = overbandL.shift(1)
    #前日の終値が-２σの外にあって今日は中にある、またはＺ値が-３σ以下
    tmpbuysig = (overbandL != overbandLshift) & (overbandL==True) | (z < -3) 
    
    df["BUYP"] = [m if d == True else np.nan for d, m in zip(tmpbuysig, df["Low"])]
    
    overbandU = Uperband <= df["High"]
    overbandUshift = overbandU.shift(1)
    #前日の終値が-２σの外にあって今日は中にある、またはＺ値が+３σ以上
    tmpsellsig = (overbandU != overbandUshift) & (overbandU==True) | (z > 3) 
    
    df["SELLP"] = [m if d == True else np.nan for d, m in zip(tmpsellsig, df["High"])]
    
    #売り条件RSI短期も長期も70以下でt短期がデッドクロス
    # tmprsicrs = (df['RSIFast'] < df['RSIFast'].shift(1)) & (df['RSIFast'].shift(1) >= 70) & (df['RSISlow'] < df['RSISlow'].shift(1)) & (df['RSISlow'].shift(1) >= 70) & (df['RSISlow'] >= df['RSIFast'])
    # df["SELRSI"] = [h if rsi == True else np.nan for rsi, h in zip(tmprsicrs, df["High"])] 
    tmprsidc = (df['RSIFast'] < df['RSIFast'].shift(1)) & (df['RSIFast'].shift(1) >= 70) & (df['RSISlow'] < df['RSISlow'].shift(1)) & (df['RSISlow'].shift(1) >= 70) & (df['RSISlow'] >= df['RSIFast'])
    df["SELRSI"] = [h if rsi == True else np.nan for rsi, h in zip(tmprsidc, df["High"])]
    
    tmprsigc = (df['RSIFast'] > df['RSIFast'].shift(1)) & (df['RSIFast'].shift(1) <= 30) & (df['RSISlow'] > df['RSISlow'].shift(1)) & (df['RSISlow'].shift(1) <= 30) & (df['RSISlow'] <= df['RSIFast'])
    df["BUYRSI"] = [h if rsi == True else np.nan for rsi, h in zip(tmprsigc, df["Low"])]
    return df

#スイングポイントをデータセットに追加する関数
@st.cache_data
def get_addswing_data(df):
    #短期売スイング
    df["ssw"] = (df["High"] > df["High"].shift(1)) & (df["High"] >= df["High"].shift(-1))
    df["sswprice"] = [h if s == True else np.nan for s, h in zip(df["ssw"], df["High"])]
    #短期買スイング
    df["lsw"]= (df["Low"] <= df["Low"].shift(1)) & (df["Low"] < df["Low"].shift(-1))
    df["lswprice"] = [l if s == True else np.nan for s, l in zip(df["lsw"], df["Low"])]
    
    #中期の買スイング
    mlswprice = df[df["lsw"] == True][["lswprice"]]
    mlswprice["mlsw"] = (mlswprice <= mlswprice.shift(1)) & (mlswprice < mlswprice.shift(-1))
    mlswprice["中期買スイング"] = [l if s == True else np.nan for s, l in zip(mlswprice["mlsw"], mlswprice["lswprice"])]
    # インデックスを基準にデータフレームを結合
    df = pd.concat([df, mlswprice[["中期買スイング"]]], axis=1)
    #中期の売スイング
    msswprice = df[df["ssw"] == True][["sswprice"]]
    msswprice["mssw"] = (msswprice >= msswprice.shift(1)) & (msswprice >= msswprice.shift(-1))
    msswprice["中期売スイング"] = [l if s == True else np.nan for s, l in zip(msswprice["mssw"], msswprice["sswprice"])]
    df = pd.concat([df, msswprice[["中期売スイング"]]], axis=1)
    return df

#スイングポイントのチャートを表示する関数
def plot_swing(df, title=""):
    #インデックスを文字列型に（休日の抜けを無くす）
    df.index = pd.to_datetime(df.index).strftime('%m-%d-%Y')

    #チャートのレイアウト
    layout = {
        "height":1000,
        "title":{"text": "{}".format(title), "x": 0.5},
        "xaxis":{"title": "日付", "rangeslider":{"visible":False}},
        "yaxis1":{"domain":[.46, 1.0], "title": "価格（円）", "side": "left", "tickformat": ","},
        "yaxis2":{"domain":[.40,.46]},
        "yaxis3":{"domain":[.20,.295], "title":"出来高", "side":"right"},
        "yaxis4":{"domain":[.10,.195], "title":"MACD", "side":"right"},
        "yaxis5":{"domain":[.00,.095], "title":"RSI", "side":"right"},
        "plot_bgcolor":"light blue"
    }
    
    data = [
        go.Candlestick(yaxis="y1",x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                       increasing_line_color="red",
                       increasing_line_width=1.0,
                       increasing_fillcolor="red",
                       decreasing_line_color="blue",
                       decreasing_line_width=1.0,
                       decreasing_fillcolor="blue"
                       ),
        #移動平均
        go.Scatter(yaxis="y1",x=df.index, y=df['MA5'], name="MA5",
                   line={"color": "royalblue", "width":1.2}),
        go.Scatter(yaxis="y1",x=df.index, y=df['MA20'], name="MA20",
                   line={"color": "lightseagreen", "width":1.2}),
        go.Scatter(yaxis="y1",x=df.index, y=df['MA60'], name="MA60",
                   line={"color": "darkred", "width":1.0}),
        
        #短期スイングのマーキング
        #ショートのマーキング
        go.Scatter(yaxis="y1",x=df.index, y=df['sswprice'], name="短期の売スイング",
                   mode="markers", opacity=0.8,
                   marker={"size": 15, "color": "pink", "symbol": "x"}),
        go.Scatter(yaxis="y1",x=df.index, y=df['中期売スイング'], name="中期の売スイング",
                   mode="markers", opacity=0.8,
                   marker={"size": 15, "color": "red", "symbol": "x"}),    
        #ロングのマーキング
        go.Scatter(yaxis="y1",x=df.index, y=df['lswprice'], name="短期の買スイング",
                   mode="markers", opacity=0.5,
                   marker={"size": 15, "color": "blue"}),
        go.Scatter(yaxis="y1",x=df.index, y=df['中期買スイング'], name="中期の買スイング",
                   mode="markers", opacity=0.8,
                   marker={"size": 15, "color": "green"}),    
        #ボリンジャーバンド
        go.Scatter(yaxis="y1",x=df.index, y=df['UBand'], name="",
                   line={"color":"lavender", "width":0}),
        go.Scatter(yaxis="y1",x=df.index, y=df['LBand'], name="BB",
                   line={"color":"lavender", "width":0},
                   fill="tonexty", fillcolor="rgba(170,170,170,.2)"),
        #出来高
        go.Bar(yaxis="y3", x=df.index, y=df['Volume'], name="Volume",marker={"color":"orange"}),
        #MACD
        go.Scatter(yaxis="y4", x=df.index, y=df['MACD'], name="MACD",
                   line={"color":"magenta", "width":1}),
        go.Scatter(yaxis="y4", x=df.index, y=df['MACD_Signal'], name="MACDSIG",
                   line={"color":"green", "width":1}),
        go.Bar(yaxis="y4", x=df.index, y=df['MACD_Hist'], name="MACDHIST",marker={"color":"blue"}),
        #RSI
        go.Scatter(yaxis="y5", x=df.index, y=df['RSIFast'], name="RSIFast",
                   line={"color":"magenta", "width":1}),
        go.Scatter(yaxis="y5", x=df.index, y=df['RSISlow'], name="RSISlow",
                   line={"color":"green", "width":1}),
        go.Scatter(yaxis="y5", x=df.index, y=df['30'], name="30",
                   line={"color":"black", "width":0.5}),
        go.Scatter(yaxis="y5", x=df.index, y=df['70'], name="70",
                   line={"color":"black", "width":0.5}),
    
    ]
    
    fig = go.Figure(data = data, layout = go.Layout(layout))   
    return fig

#予測チャートを表示する関数
def plot_pred_swing(df, title=""):
    #インデックスを文字列型に（休日の抜けを無くす）
    df.index = pd.to_datetime(df.index).strftime('%m-%d-%Y')
    
    layout = {
        "height":1000,
        "title":{"text": "{}".format(title), "x": 0.5},
        "xaxis":{"title": "日付", "rangeslider":{"visible":False}},
        "yaxis1":{"domain":[.46, 1.0], "title": "価格（円）", "side": "left", "tickformat": ","},
        "yaxis2":{"domain":[.40,.46]},
        "yaxis3":{"domain":[.20,.295], "title":"出来高", "side":"right"},
        "yaxis4":{"domain":[.10,.195], "title":"MACD", "side":"right"},
        "yaxis5":{"domain":[.00,.095], "title":"RSI", "side":"right"},
        "plot_bgcolor":"light blue"
    }
    
    data = [
        go.Candlestick(yaxis="y1",x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                       increasing_line_color="red",
                       increasing_line_width=1.0,
                       increasing_fillcolor="red",
                       decreasing_line_color="blue",
                       decreasing_line_width=1.0,
                       decreasing_fillcolor="blue"
                       ),
        #移動平均
        go.Scatter(yaxis="y1",x=df.index, y=df['HMA5'], name="HMA5",
                   line={"color": "royalblue", "width":1.2}),
        go.Scatter(yaxis="y1",x=df.index, y=df['MA20'], name="MA20",
                   line={"color": "lightseagreen", "width":1.2}),
        go.Scatter(yaxis="y1",x=df.index, y=df['MA60'], name="MA60",
                   line={"color": "darkred", "width":1.0}),
        
        #短期スイングのマーキング
        #ショートのマーキング
        go.Scatter(x=df.index, y=df['BUYP'], name="買スイング",
                   mode="markers", opacity=0.8,
                   marker={"size": 15, "color": "lime"}),
        go.Scatter(x=df.index, y=df['SELLP'], name="売スイング",
                   mode="markers", opacity=0.8,
                   marker={"size": 15, "color": "red", "symbol": "x"}),
        go.Scatter(x=df.index, y=df['SELRSI'], name="RSIの売シグナル",
                   mode="markers", opacity=0.8,
                   marker={"size": 15, "color": "orange", "symbol": "x"}), 
        go.Scatter(x=df.index, y=df['BUYRSI'], name="RSIの買シグナル",
                   mode="markers", opacity=0.8,
                   marker={"size": 15, "color": "aqua"}),
        
        #ボリンジャーバンド
        go.Scatter(yaxis="y1",x=df.index, y=df['UBand'], name="",
                   line={"color":"lavender", "width":0}),
        go.Scatter(yaxis="y1",x=df.index, y=df['LBand'], name="BB",
                   line={"color":"lavender", "width":0},
                   fill="tonexty", fillcolor="rgba(170,170,170,.2)"),
        #出来高
        go.Bar(yaxis="y3", x=df.index, y=df['Volume'], name="Volume",marker={"color":"orange"}),
        #MACD
        go.Scatter(yaxis="y4", x=df.index, y=df['MACD'], name="MACD",
                   line={"color":"magenta", "width":1}),
        go.Scatter(yaxis="y4", x=df.index, y=df['MACD_Signal'], name="MACDSIG",
                   line={"color":"green", "width":1}),
        go.Bar(yaxis="y4", x=df.index, y=df['MACD_Hist'], name="MACDHIST",marker={"color":"blue"}),
        #RSI
        go.Scatter(yaxis="y5", x=df.index, y=df['RSIFast'], name="RSIFast",
                   line={"color":"magenta", "width":1}),
        go.Scatter(yaxis="y5", x=df.index, y=df['RSISlow'], name="RSISlow",
                   line={"color":"green", "width":1}),
        go.Scatter(yaxis="y5", x=df.index, y=df['30'], name="30",
                   line={"color":"black", "width":0.5}),
        go.Scatter(yaxis="y5", x=df.index, y=df['70'], name="70",
                   line={"color":"black", "width":0.5}),
    
    ]
    
    fig = go.Figure(data = data, layout = go.Layout(layout))
    return fig

########################################################
# ストリームリットのインターフェイス
########################################################
st.title("予測チャートと実際のチャート")
st.text("予測と実際の売買タイミングを表示")
image = Image.open("./images/headerpredactual.png")
st.image(image)
st.caption("ティッカーコードを入力")


col1, col2 = st.columns(2)
with col1:
    #ティッカー（初期値は、金/ドル）
    ticker = st.text_input("コードを入力:", value="GC=F")
    rsif = st.number_input("RSI短期", value=3)
    bbspan = st.number_input("ボリンジャーバンド期間", value=20)
    
with col2:
    #HMA
    dispan = st.number_input("表示期間", value=60)
    rsis = st.number_input("RSI長期", value=5)
    bbsigma = st.number_input("ボリンジャーバンド標準偏差", value=2)

st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 30%;
        background-color: #6a5acd;  /* 背景色 */
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
btnplot = st.button("描画")

if btnplot:
    df = get_stock_data(checkTicker(ticker))
    dftec = get_addtec_data(df, rsif, rsis, bbspan, bbsigma)
    
    dfpred = get_addpred_data(dftec)
    figpred = plot_pred_swing(dfpred.tail(dispan))
    st.subheader(get_code_company_name(ticker) + "予想スイングポイント")
    st.plotly_chart(figpred)
    
    dfswing = get_addswing_data(dftec)
    figswing = plot_swing(dfswing.tail(dispan))
    st.subheader(get_code_company_name(ticker) + "実際のスイングポイント")
    st.plotly_chart(figswing)

