import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

#自作ライブラリー
from mystock_lib import *

# --- App Basic Settings ---
st.set_page_config(page_title="エントリー支援アプリ", layout="wide")

st.title("エントリー支援アプリ 📈")

image = Image.open("./images/headerbuyentry.png")
st.image(image)
# Added caption
st.caption("本アプリは買いのみのエントリーポイントを表示しています、また中・長期線が下向きの時は精度が下がるので注意してください！")


# --- Sidebar (Input Controls) ---
st.sidebar.header("設定")

ticker_code = checkTicker(st.sidebar.text_input("ティッカーコード", value="^N225"))
ema_period = st.sidebar.number_input("中期EMA期間", min_value=1, max_value=200, value=18)
sma_period = st.sidebar.number_input("長期SMA期間", min_value=1, max_value=200, value=60)

run_button = st.sidebar.button("実行")

# --- Main Logic ---
if run_button:
    # 1. Data Fetching
    try:
        df = yf.download(ticker_code, start='2022-01-01', end=None, progress=False)
        if df.empty:
            st.error("ティッカーコードが見つからないか、データがありません。")
            st.stop()
    except Exception as e:
        st.error(f"データ取得中にエラーが発生しました: {e}")
        st.stop()

    # Convert hierarchical column names to simple names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Technical Indicator Calculations
    df['EMA_mid'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['SMA_long'] = df['Close'].rolling(window=sma_period).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 3. Entry Condition Detection
    condition1 = df['EMA_mid'] > df['EMA_mid'].shift(1)
    condition2 = df['Close'].shift(1) < df['EMA_mid'].shift(1)
    condition3 = df['Close'] > df['EMA_mid']
    final_mask = condition1 & condition2 & condition3
    entry_signals = df[final_mask].copy()

    # 4. Chart Creation
    # Temporarily convert date index to string for chart display
    display_df = df.copy()
    display_df.index = pd.to_datetime(display_df.index).strftime('%m-%d-%Y')
    display_entry_signals = entry_signals.copy()
    display_entry_signals.index = pd.to_datetime(display_entry_signals.index).strftime('%m-%d-%Y')

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=display_df.index,
                    open=display_df['Open'], high=display_df['High'], low=display_df['Low'], close=display_df['Close'],
                    name=ticker_code))

    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['EMA_mid'], mode='lines', line=dict(color='cyan', width=1.5), name="中期EMA"))
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['SMA_long'], mode='lines', line=dict(color='tomato', width=1.5), name="長期SMA"))
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['SMA_200'], mode='lines', line=dict(color='purple', width=1.5, dash='dot'), name="200日移動平均"))

    entry_dates = display_entry_signals.index
    entry_prices = display_entry_signals['Low'] * 0.98
    fig.add_trace(go.Scatter(x=entry_dates, y=entry_prices, mode='markers',
                             marker=dict(color='blue', size=10, symbol='triangle-up'),
                             name='エントリーポイント'))

    fig.update_layout(
        title=f"{ticker_code}のエントリーポイント",
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 5. Display
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("検出されたエントリーポイント")
    if not entry_signals.empty:
        # Prepare data for display (select 'Open' and 'Close' columns)
        display_data = entry_signals[['Open', 'Close']].copy()
        display_data.index = pd.to_datetime(display_data.index).strftime('%Y-%m-%d')
        display_data.rename(columns={'Open': '当日の始値', 'Close': '当日の終値'}, inplace=True)

        # Display in st.dataframe and format numbers to one decimal place
        st.dataframe(display_data.style.format('{:.1f}'))
    else:
        st.info("条件に合致するエントリーポイントは見つかりませんでした。")