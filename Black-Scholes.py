import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import yfinance as yf
from scipy.optimize import fsolve
from PIL import Image

#
kokusai10 = 0
hv = 0
#
#コードを引数に前日終値を求める関数
def getuastprice(code):
    ticker = yf.Ticker(code)
    return ticker.info["regularMarketOpen"]
    
#
def getkinri(type):
    if type=="アメリカン":
        kokusai10 = 0.04478
    else:
        kokusai10 = 0.013
    return kokusai10
    
#HVの計算
def calculate_volatility(data, period=20):
    """
    ヒストリカル・ボラティリティ（年率）を計算する関数

    Args:
        data: 価格データ（pandas.Series）
        period: ボラティリティ計算期間（日、デフォルトは20日）

    Returns:
        年率ボラティリティ
    """

    # 価格変動率を計算
    returns = data.pct_change()
    # ボラティリティ（標準偏差）を計算
    volatility = returns.rolling(window=period).std()
    # 年率換算
    annual_volatility = volatility * np.sqrt(250)  # 250は年間取引日数
    
    return annual_volatility

#データを取得HVを返す
def getHV(code):
    df = yf.download(code, progress=False)
    df.columns = [col[0] for col in df.columns]
    price_data = df["Close"] 
    return calculate_volatility(price_data, period=20).iloc[-1]

#コールの理論値
def black_scholes_call(S, K, sigma, r, t):
    """
    ブラックショールズモデルによるヨーロピアンコールオプション価格の算出

    Args:
        S: 原資産価格（日経225指数）
        K: 権利行使価格
        sigma: 原資産価格のボラティリティ（年率）
        r: 無リスク金利（年率）
        t: 満期までの期間（年）

    Returns:
        コールオプション価格
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    C = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    return C

#プットの理論値
def black_scholes_put(S, K, sigma, r, t):
    """
    ブラックショールズモデルによるヨーロピアンプットオプション価格の算出

    Args:
        S: 原資産価格（日経225指数）
        K: 権利行使価格
        sigma: 原資産価格のボラティリティ（年率）
        r: 無リスク金利（年率）
        t: 満期までの期間（年）

    Returns:
        プットオプション価格
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    P = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return P

#アメリカンオプション理論価格
def american_option_price(S, K, sigma, r, t, option_type="call"):
    """
    アメリカンオプションの価格を算出する関数（近似解）

    Args:
        S: 原資産価格
        K: 権利行使価格
        sigma: 原資産価格のボラティリティ
        r: 無リスク金利
        t: 満期までの期間
        option_type: "call" または "put" (デフォルトは "call")

    Returns:
        アメリカンオプションの価格
    """
    # ブラックショールズモデルによるヨーロピアンオプション価格
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type == "call":
        C_european = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        intrinsic_value = max(0, S - K)
        american_price = max(C_european, intrinsic_value)

    elif option_type == "put":
        P_european = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
        intrinsic_value = max(0, K - S)
        american_price = max(P_european, intrinsic_value)

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return american_price


def implied_volatility(market_price, S, K, T, r):
    '''
    インプライド・ボラティリティの計算
    '''
    def equation(sigma):
        return black_scholes_call(S, K, T, r, sigma) - market_price
    
    # 初期値として適当なボラティリティを設定
    initial_guess = 0.5 
    iv = fsolve(equation, initial_guess)[0]
    return iv

#
st.title("オプションの理論価格")
st.text("ブラックショールズ方程式で理論価格を算出します")
image = Image.open("header.png")
st.image(image)
st.caption("原資産価格、権利行使価格、残存日数、金利（10年国債）、コール現在価格、プット現在価格")
st.caption("を入力してください。")


selected_option = st.selectbox('オプション種類（選択してください）:', ["ヨーロピアン", "アメリカン"])
ticker = st.text_input("ティッカーコード：", value="^N225")
uastprice = st.text_input("原資産価格：" , value=getuastprice(ticker))
exeprice = st.text_input("権利行使価格：", value=getuastprice(ticker))
ndr = st.text_input("残存日数：", value=20)
hv = st.text_input("ボラティリティー(HV)：", value=getHV(ticker))
kinri = st.text_input("無リスク金利（長期国債金利）：", value=getkinri(selected_option))
col1, col2 = st.columns(2)
with col1:
    marketpriceC = st.text_input("コールオプション現在値：", value="")

with col2:
    marketpriceP = st.text_input("プットオプション現在値：", value="")

st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #4CAF50;  /* 背景色 */
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
btncal = st.button('理論価格計算')

#####イベント処理
if btncal:
    S = float(uastprice)
    K = float(exeprice)
    sigma = float(hv)
    r = float(kinri)
    t = float(int(ndr) / 365)
    
    if selected_option=="アメリカン":
        #コールオプション
        C_american = american_option_price(S, K, sigma, r, t, option_type="call")
        # プットオプション
        P_american = american_option_price(S, K, sigma, r, t, option_type="put")
        st.write("コールオプション理論価格:", round(C_american, 2))
        st.write("プットオプション理論価格:", round(P_american, 2)) 
    else:
        #ヨーロピアンの場合
        call_price = black_scholes_call(S, K, sigma, r, t)
        put_price = black_scholes_put(S, K, sigma, r, t)
        st.write("コールオプション理論価格:", round(call_price, 2))
        st.write("プットオプション理論価格:", round(put_price, 2))
        if marketpriceC != "" and marketpriceP != "" :
            marketpriceC = float(marketpriceC)
            marketpriceP = float(marketpriceP)
            iv = implied_volatility(marketpriceC, S, K, t, r)
            st.write("インポライドボラティリティー(IV):", iv)
            st.write("コールー理論価格:", marketpriceC - call_price )
            st.write("プットー理論価格:", marketpriceP - put_price )
        elif marketpriceC != "":
            marketpriceC = float(marketpriceC)
            iv = implied_volatility(marketpriceC, S, K, t, r)
            st.write("インポライドボラティリティー(IV):", iv)
        else:
            st.write("コールとプットの現在価格を入力してください")
            
    




