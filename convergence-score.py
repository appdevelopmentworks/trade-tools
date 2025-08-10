# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from PIL import Image
import warnings

# --- アプリケーションの基本設定 ---
st.set_page_config(
    page_title="株価収束度分析アプリ",
    page_icon="📈",
    layout="wide"
)

# yfinanceに関する将来的な仕様変更の警告を非表示にします
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance.utils')


# --- バックエンド処理 (分析関数) ---
# この関数はキャッシュされ、同じ銘柄と期間での再計算を高速化します
@st.cache_data
def analyze_price_convergence(ticker, period="3mo", window_size=5):
    """
    株価の始値、終値、5日移動平均線の収束度を分析する関数。
    """
    stock_data = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    if stock_data.empty or len(stock_data) < window_size * 2:
        return None
        
    stock_data['SMA5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data.dropna(inplace=True)
    
    if len(stock_data) < 2:
        return None
        
    cv_list = []
    for _, row in stock_data.iterrows():
        values = np.array([row['Open'], row['Close'], row['SMA5']])
        mean = np.mean(values)
        std = np.std(values)
        cv = std / mean if mean > 0 else 0
        cv_list.append(cv)
        
    stock_data['Convergence_CV'] = cv_list
    stock_data['Convergence_Score'] = stock_data['Convergence_CV'].rolling(window=window_size).mean()
    stock_data.dropna(inplace=True)
    
    return stock_data


# --- UI (ユーザーインターフェース) 部分 ---

st.title('📈 株価収束ランクキング')
image = Image.open("./images/headerconvergenceranking.png")
st.image(image)

st.info('CSVファイルをアップロードし、サイドバーで条件を設定して「分析を実行」ボタンを押してください。')

# --- サイドバー (入力コントロール) ---
st.sidebar.header('⚙️ 設定')

# 1. ファイルアップロード
uploaded_file = st.sidebar.file_uploader(
    "銘柄リストのCSVファイルをアップロード", type=['csv']
)
st.sidebar.caption("※1列目にティッカーコード、2列目に銘柄名があるCSVファイルを選択してください。")

# 2. Convergence_Scoreの範囲指定
st.sidebar.subheader("絞り込み条件")
min_score = st.sidebar.number_input(
    'Convergence_Score (最小値)',
    min_value=0.0, max_value=0.5, value=0.002, step=0.001, format="%.4f"
)
max_score = st.sidebar.number_input(
    'Convergence_Score (最大値)',
    min_value=0.0, max_value=0.5, value=0.01, step=0.001, format="%.4f"
)

# 3. 実行ボタン
run_button = st.sidebar.button('分析を実行')

# --- メイン画面 (結果表示) ---
if run_button:
    if uploaded_file is not None:
        if min_score > max_score:
            st.error("エラー: 最小値が最大値を超えています。")
        else:
            with st.spinner('分析を実行中です...'):
                try:
                    # usecols=[0, 1] でCSVの最初の2列のみを読み込む
                    screener_df = pd.read_csv(uploaded_file, usecols=[0, 1], encoding='cp932')
                    screener_df.columns = ['コード', '銘柄名']
                except Exception:
                    # cp932で失敗した場合、utf-8で再試行
                    uploaded_file.seek(0) # ファイルポインタを先頭に戻す
                    screener_df = pd.read_csv(uploaded_file, usecols=[0, 1], encoding='utf-8')
                    screener_df.columns = ['コード', '銘柄名']

                screener_df['コード'] = screener_df['コード'].astype(str)
                results_list = []
                
                # Streamlit用のプログレスバー
                progress_bar = st.progress(0, text="分析開始")
                total_rows = len(screener_df)

                for index, row in screener_df.iterrows():
                    ticker_code = row['コード']
                    company_name = row['銘柄名']
                    yf_ticker = f"{ticker_code}.T"
                    
                    convergence_data = analyze_price_convergence(yf_ticker)
                    
                    if convergence_data is not None and len(convergence_data) >= 2:
                        latest_close = convergence_data['Close'].iloc[-1]
                        previous_close = convergence_data['Close'].iloc[-2]
                        change_percentage = ((latest_close - previous_close) / previous_close) * 100 if previous_close > 0 else 0
                        latest_data_row = convergence_data.iloc[-1]
                        
                        results_list.append({
                            'コード': ticker_code, '銘柄名': company_name, '現在値': latest_close,
                            '前日比(%)': change_percentage, 'Open': latest_data_row['Open'],
                            'Close': latest_data_row['Close'], 'SMA5': latest_data_row['SMA5'],
                            'Volume': latest_data_row['Volume'], 'Convergence_Score': latest_data_row['Convergence_Score']
                        })
                    
                    # プログレスバーを更新
                    progress_bar.progress((index + 1) / total_rows, text=f"分析中: {company_name} ({ticker_code})")

                progress_bar.empty() # プログレスバーを非表示に

            if results_list:
                final_df = pd.DataFrame(results_list)
                sorted_df = final_df.sort_values(by='Convergence_Score', ascending=True)

                # サイドバーで指定された範囲でフィルタリング
                filtered_df = sorted_df[
                    (sorted_df['Convergence_Score'] >= min_score) &
                    (sorted_df['Convergence_Score'] <= max_score)
                ]

                st.success(f'分析が完了しました。{len(filtered_df)}件の銘柄が見つかりました。')
                
                # Streamlitのデータフレーム表示機能で、各列の書式を設定
                st.dataframe(filtered_df, use_container_width=True, hide_index=True,
                             column_config={
                                 "現在値": st.column_config.NumberColumn(format="%.2f"),
                                 "前日比(%)": st.column_config.NumberColumn(format="%.2f%%"),
                                 "Open": st.column_config.NumberColumn(format="%.2f"),
                                 "Close": st.column_config.NumberColumn(format="%.2f"),
                                 "SMA5": st.column_config.NumberColumn(format="%.2f"),
                                 "Volume": st.column_config.NumberColumn(format=""),
                                 "Convergence_Score": st.column_config.NumberColumn(format="%.5f"),
                             })

                # 結果をCSVでダウンロードするボタン
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8-sig')

                csv_data = convert_df_to_csv(filtered_df)
                st.download_button(
                    label="📂 結果をCSVでダウンロード",
                    data=csv_data,
                    file_name=f'convergence_ranking_{min_score:.4f}_to_{max_score:.4f}.csv',
                    mime='text/csv',
                )

            else:
                st.warning('分析可能なデータが見つかりませんでした。')
    else:
        st.warning('⚠️ CSVファイルをアップロードしてください。')