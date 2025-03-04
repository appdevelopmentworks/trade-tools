import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
from PIL import Image


def black_scholes_european(S, K, sigma, r, T, option_type="call"):
    """
    ブラック・ショールズモデルによるヨーロピアンオプションの価格計算
    Args:
        S: 原資産価格
        K: 権利行使価格
        sigma: ボラティリティ
        r: 無リスク金利
        T: 満期までの期間（年）
        option_type: "call"または"put"
    Returns:
        ヨーロピアンオプションの価格
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def american_option_binomial(S, K, sigma, r, T, option_type="call", n=100, dividend_yield=0):
    """
    二項ツリーモデルによるアメリカンオプションの価格計算
    Args:
        S: 原資産価格
        K: 権利行使価格
        sigma: ボラティリティ
        r: 無リスク金利
        T: 満期までの期間（年）
        option_type: "call"または"put"
        n: ステップ数（デフォルト: 100）
        dividend_yield: 配当利回り（デフォルト: 0）
    Returns:
        アメリカンオプションの価格
    """
    # タイムステップの計算
    dt = T / n
    
    # アップとダウンの係数
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # リスク中立確率
    p = (np.exp((r - dividend_yield) * dt) - d) / (u - d)
    
    # 割引率
    discount = np.exp(-r * dt)
    
    # 価格ツリーの初期化
    price_tree = np.zeros((n+1, n+1))
    
    # 満期時点の価格を計算
    for i in range(n+1):
        # 満期時の原資産価格
        ST = S * (u ** (n - i)) * (d ** i)
        
        # 満期時点のオプション価値（ペイオフ）
        if option_type == "call":
            price_tree[i, n] = max(0, ST - K)
        else:  # put
            price_tree[i, n] = max(0, K - ST)
    
    # バックワードインダクション（後ろから前に計算）
    for j in range(n-1, -1, -1):
        for i in range(j+1):
            # 現時点の原資産価格
            St = S * (u ** (j - i)) * (d ** i)
            
            # 保有継続価値（次の時点の期待値）
            continuation_value = discount * (p * price_tree[i, j+1] + (1-p) * price_tree[i+1, j+1])
            
            # 即時行使価値
            if option_type == "call":
                exercise_value = max(0, St - K)
            else:  # put
                exercise_value = max(0, K - St)
            
            # アメリカンオプションでは、継続価値と即時行使価値の大きい方を選択
            price_tree[i, j] = max(continuation_value, exercise_value)
    
    # オプション価格（ツリーの根）を返す
    return price_tree[0, 0]

def bsm_american_option_price(S, K, sigma, r, T, option_type="call", n=100, dividend_yield=0):
    """
    アメリカンオプション価格を計算する関数
    Args:
        S: 原資産価格
        K: 権利行使価格
        sigma: ボラティリティ
        r: 無リスク金利
        T: 満期までの期間（年）
        option_type: "call"または"put"
        n: 二項ツリーのステップ数（デフォルト: 100）
        dividend_yield: 配当利回り（デフォルト: 0）
    Returns:
        アメリカンオプションの価格
    """
    # ヨーロピアンオプション価格（比較用）
    european_price = black_scholes_european(S, K, sigma, r, T, option_type)
    
    # 無配当コールの場合は早期行使が最適でないため、ヨーロピアン価格を返す
    if option_type == "call" and dividend_yield == 0:
        return european_price
    
    # その他の場合は二項ツリーモデルで計算
    american_price = american_option_binomial(S, K, sigma, r, T, option_type, n, dividend_yield)
    
    return american_price

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
    date = datetime.now() - timedelta(days=35)
    df = yf.download(code,start= date.date(), progress=False)
    df.columns = [col[0] for col in df.columns]
    price_data = df["Close"] 
    return calculate_volatility(price_data, period=20).iloc[-1]
#####################################################################


st.title("海外オプション理論価格ツール")
image = Image.open("headeroption.png")
st.image(image)
st.caption("ティッカーコードを入力し納会日、コールかプットを選択")

col1, col2 = st.columns(2)

with col1:
    tcode = st.text_input("ティッカーコードを入力", value="SPY")
    # Tickerオブジェクトの作成
    ticker = yf.Ticker(tcode)
    #
    optionT = st.radio("オプションタイプ:", ("call", "put"), horizontal=True)
with col2:
    gengetu = st.selectbox("限月を選択:", ticker.options)
    redays = datetime.strptime(gengetu, "%Y-%m-%d") - datetime.now()
    zandays = st.text_input("残存日数:", value=redays.days + 1)
    

col3, col4, col5 = st.columns(3)

with col3:
    gensiprice =  st.text_input("原資産価格:", value=ticker.info["regularMarketPrice"])
with col4:
    haitou = st.text_input("配当利回り:", value=ticker.info["dividendYield"])
with col5:
    txtKinri = st.text_input("無リスク金利:", value=0.04478)
    

#オプションのデータを取得
option_chain = ticker.option_chain(gengetu) 
calls = option_chain.calls
callsja = calls[['strike','lastPrice','bid','ask','change','percentChange','volume','openInterest','impliedVolatility','inTheMoney', 'lastTradeDate']]
callsja = callsja.set_axis(['権利行使価格', '終値', '売気配', "買気配", "変化", "変化率", "出来高", "契約未履行数", "IV", "ITM", "最終取引成立日"], axis=1)
puts = option_chain.puts
putsja = puts[['strike','lastPrice','bid','ask','change','percentChange','volume','openInterest','impliedVolatility','inTheMoney', 'lastTradeDate']]
putsja = putsja.set_axis(['権利行使価格', '終値', '売気配', "買気配", "変化", "変化率", "出来高", "契約未履行数", "IV", "ITM", "最終取引成立日"], axis=1)

#無リスク金利
r = float(txtKinri)
#S: 原資産価格
S = ticker.info["regularMarketPrice"]
#配当利回り
dividend_yield = ticker.info["dividendYield"] / 100
#残存日数
redays = datetime.strptime(gengetu, "%Y-%m-%d") - datetime.now()
T = (redays.days + 1) / 365
#ヒストリカルボラティリティー
sigma = getHV(tcode)


#配列初期化
rironchiC = []
rironchiP = []
#
for row in callsja["権利行使価格"]:
    rp = bsm_american_option_price(S, row, sigma, r, T, option_type="call", n=100, dividend_yield=dividend_yield)
    rironchiC.append(round(rp, 2))
    
for row in putsja["権利行使価格"]:
    rp = bsm_american_option_price(S, row, sigma, r, T, option_type="put", n=100, dividend_yield=dividend_yield)
    rironchiP.append(round(rp, 2))
    
callsja.insert(2, '理論価格', rironchiC)
callsja.insert(3, '終値-理論値', callsja["終値"] - callsja["理論価格"])
putsja.insert(2, '理論価格', rironchiP)
putsja.insert(3, '終値-理論値', putsja["終値"] - putsja["理論価格"])

st.write("HV(ヒストリカルボラティリティー)",round(sigma, 2))
# st.write("S",S)
# st.write("T",T)
# st.write("dividend_yield",dividend_yield)


if optionT=="call":
    st.dataframe(callsja)
else:
    st.dataframe(putsja)

