import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib  # 日本語文字化け対応
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
#自作ライブラリー
from mystock_lib import *


# ページ設定
st.set_page_config(
    page_title="株価収束検出システム",
    page_icon="📊",
    layout="wide"
)

# 関数定義
def calculate_convergence_metrics(df, window=10):
    """
    株価の収束度を計算する関数
    """
    # 5日移動平均線を計算
    df['MA5'] = df['Close'].rolling(window=5).mean()
    
    # 1. 変動係数（Coefficient of Variation）
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['CV'] = (rolling_std / rolling_mean) * 100
    
    # 2. レンジ比率（高値-安値を終値で正規化）
    df['Range'] = df['High'] - df['Low']
    df['Range_Ratio'] = (df['Range'] / df['Close']) * 100
    df['Range_Ratio_MA'] = df['Range_Ratio'].rolling(window=window).mean()
    
    # 3. ATR（Average True Range）を終値で正規化
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=window).mean()
    df['ATR_Ratio'] = (df['ATR'] / df['Close']) * 100
    
    # 4. ボリンジャーバンド幅
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / rolling_mean) * 100
    
    # 5. 価格変動の標準偏差（対数リターン）
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Return'].rolling(window=window).std() * 100
    
    # 6. 収束スコア（複合指標）
    df['Convergence_Score'] = (
        (1 / (1 + df['CV'].fillna(100))) * 25 +
        (1 / (1 + df['Range_Ratio_MA'].fillna(100))) * 25 +
        (1 / (1 + df['ATR_Ratio'].fillna(100))) * 25 +
        (1 / (1 + df['Volatility'].fillna(100))) * 25
    )
    
    # 収束フラグ（閾値以下を収束と判定）
    cv_threshold = 2.0  # CV 2%以下
    range_threshold = 2.0  # レンジ比率 2%以下
    df['Is_Converged'] = (
        (df['CV'] < cv_threshold) & 
        (df['Range_Ratio_MA'] < range_threshold)
    ).astype(int)
    
    return df

def detect_convergence_periods(df, min_days=5):
    """
    連続した収束期間を検出
    """
    convergence_periods = []
    
    # 収束フラグが1の連続期間を検出
    df['Group'] = (df['Is_Converged'] != df['Is_Converged'].shift()).cumsum()
    
    for group_id in df[df['Is_Converged'] == 1]['Group'].unique():
        period_df = df[(df['Group'] == group_id) & (df['Is_Converged'] == 1)]
        
        if len(period_df) >= min_days:
            start_date = period_df.index[0]
            end_date = period_df.index[-1]
            duration = len(period_df)
            avg_cv = period_df['CV'].mean()
            
            convergence_periods.append({
                'start': start_date,
                'end': end_date,
                'duration': duration,
                'avg_cv': avg_cv
            })
    
    return convergence_periods

def create_convergence_plot(df, ticker, convergence_periods):
    """
    収束分析の可視化（Streamlit用）
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 1. 株価とMA5
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='終値', color='black', linewidth=1)
    ax1.plot(df.index, df['MA5'], label='5日移動平均', color='blue', alpha=0.7)
    
    # 収束期間をハイライト
    for period in convergence_periods:
        ax1.axvspan(period['start'], period['end'], alpha=0.2, color='red', 
                   label='収束期間' if period == convergence_periods[0] else "")
    
    ax1.set_ylabel('株価')
    ax1.set_title(f'{ticker} - 株価収束分析')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 変動係数（CV）
    ax2 = axes[1]
    ax2.plot(df.index, df['CV'], label='変動係数 (CV)', color='red', linewidth=1)
    ax2.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='閾値 (2%)')
    ax2.set_ylabel('CV (%)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. レンジ比率とATR比率
    ax3 = axes[2]
    ax3.plot(df.index, df['Range_Ratio_MA'], label='レンジ比率 (移動平均)', 
             color='green', linewidth=1)
    ax3.plot(df.index, df['ATR_Ratio'], label='ATR比率', 
             color='orange', linewidth=1, alpha=0.7)
    ax3.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('比率 (%)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. 収束スコア
    ax4 = axes[3]
    ax4.fill_between(df.index, 0, df['Convergence_Score'] * 100, 
                     alpha=0.5, color='purple', label='収束スコア')
    ax4.set_ylabel('収束スコア')
    ax4.set_xlabel('日付')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # X軸の日付フォーマット
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# ==================== メインアプリケーション ====================

st.title("📊 株価収束検出システム")
image = Image.open("./images/headerconvergence.png")
st.image(image)
st.markdown("---")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # ティッカーコード入力
    ticker_input = checkTicker(st.text_input(
        "ティッカーコード",
        value="8151",
        help="例: 8151 (日本株), AAPL (米国株)"
    ))
    
    # 期間選択
    period_options = {
        "1ヶ月": "1mo",
        "3ヶ月": "3mo",
        "6ヶ月": "6mo",
        "1年": "1y",
        "2年": "2y"
    }
    selected_period_label = st.selectbox(
        "分析期間",
        options=list(period_options.keys()),
        index=2  # デフォルト6ヶ月
    )
    period = period_options[selected_period_label]
    
    # パラメータ設定
    st.subheader("詳細パラメータ")
    window = st.slider("計算ウィンドウ（日）", 5, 20, 10)
    cv_threshold = st.slider("CV閾値（%）", 1.0, 5.0, 2.0, 0.1)
    range_threshold = st.slider("レンジ比率閾値（%）", 1.0, 5.0, 2.0, 0.1)
    min_days = st.slider("最小連続日数", 3, 10, 5)
    
    # 検出ボタン
    detect_button = st.button("🔍 収束検出実行", type="primary", use_container_width=True)

# メインコンテンツ
if detect_button:
    if not ticker_input:
        st.error("ティッカーコードを入力してください")
    else:
        try:
            # プログレスバー表示
            progress_bar = st.progress(0, text="データを取得中...")
            
            # データ取得
            stock = yf.Ticker(ticker_input)
            df = stock.history(period=period, interval='1d')
            
            if df.empty:
                st.error(f"エラー: {ticker_input} のデータを取得できませんでした")
            else:
                progress_bar.progress(33, text="収束指標を計算中...")
                
                # 収束指標を計算
                df = calculate_convergence_metrics(df, window=window)
                
                # 閾値を適用
                df['Is_Converged'] = (
                    (df['CV'] < cv_threshold) & 
                    (df['Range_Ratio_MA'] < range_threshold)
                ).astype(int)
                
                progress_bar.progress(66, text="収束期間を検出中...")
                
                # 収束期間を検出
                convergence_periods = detect_convergence_periods(df, min_days=min_days)
                
                progress_bar.progress(100, text="完了！")
                progress_bar.empty()
                
                # 結果表示
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("分析期間", f"{len(df)}日")
                with col2:
                    st.metric("収束日数", f"{df['Is_Converged'].sum()}日")
                with col3:
                    st.metric("収束率", f"{df['Is_Converged'].sum() / len(df) * 100:.1f}%")
                with col4:
                    st.metric("検出期間数", f"{len(convergence_periods)}件")
                
                st.markdown("---")
                
                # タブで結果を整理
                tab1, tab2, tab3, tab4 = st.tabs(["📈 グラフ", "📊 収束期間", "🔢 データセット", "📋 統計サマリー"])
                
                with tab1:
                    st.subheader("収束分析グラフ")
                    fig = create_convergence_plot(df, ticker_input, convergence_periods)
                    st.pyplot(fig)
                
                with tab2:
                    st.subheader("検出された収束期間")
                    if convergence_periods:
                        periods_df = pd.DataFrame(convergence_periods)
                        periods_df['start'] = pd.to_datetime(periods_df['start'])
                        periods_df['end'] = pd.to_datetime(periods_df['end'])
                        periods_df.index = range(1, len(periods_df) + 1)
                        periods_df.columns = ['開始日', '終了日', '継続日数', '平均CV(%)']
                        st.dataframe(periods_df, use_container_width=True)
                        
                        # 詳細情報
                        st.write("### 詳細情報")
                        for i, period in enumerate(convergence_periods, 1):
                            st.write(f"**期間{i}**: {period['start'].strftime('%Y-%m-%d')} ～ "
                                    f"{period['end'].strftime('%Y-%m-%d')} "
                                    f"({period['duration']}日間, 平均CV: {period['avg_cv']:.2f}%)")
                    else:
                        st.info("指定された条件で収束期間は検出されませんでした")
                
                with tab3:
                    st.subheader("データセット（直近20日）")
                    
                    # 表示する列を選択
                    display_columns = ['Close', 'MA5', 'CV', 'Range_Ratio_MA', 
                                     'ATR_Ratio', 'Volatility', 'Convergence_Score', 'Is_Converged']
                    
                    # データフレームを整形
                    display_df = df[display_columns].tail(20).round(2)
                    display_df.columns = ['終値', '5日MA', 'CV(%)', 'レンジ比率(%)', 
                                         'ATR比率(%)', 'ボラティリティ(%)', '収束スコア', '収束フラグ']
                    
                    # スタイリング付きで表示
                    st.dataframe(
                        display_df.style.background_gradient(subset=['CV(%)', 'レンジ比率(%)']),
                        use_container_width=True
                    )
                    
                    # CSV ダウンロードボタン
                    csv = df.to_csv(encoding='utf-8-sig')
                    st.download_button(
                        label="📥 全データをCSVでダウンロード",
                        data=csv,
                        file_name=f"{ticker_input}_convergence_analysis.csv",
                        mime="text/csv"
                    )
                
                with tab4:
                    st.subheader("統計サマリー")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**基本統計量**")
                        st.write(f"- 平均CV: {df['CV'].mean():.2f}%")
                        st.write(f"- 最小CV: {df['CV'].min():.2f}%")
                        st.write(f"- 最大CV: {df['CV'].max():.2f}%")
                        st.write(f"- CV中央値: {df['CV'].median():.2f}%")
                    
                    with col2:
                        st.write("**収束統計**")
                        st.write(f"- 総収束日数: {df['Is_Converged'].sum()}日")
                        st.write(f"- 収束率: {df['Is_Converged'].sum() / len(df) * 100:.1f}%")
                        st.write(f"- 最長収束期間: {max([p['duration'] for p in convergence_periods], default=0)}日")
                        st.write(f"- 平均収束スコア: {df['Convergence_Score'].mean() * 100:.1f}")
                    
                    # 期間別サマリー
                    st.write("**期間別収束率**")
                    monthly_convergence = df.groupby(pd.Grouper(freq='M'))['Is_Converged'].agg(['sum', 'count'])
                    monthly_convergence['rate'] = (monthly_convergence['sum'] / monthly_convergence['count'] * 100).round(1)
                    monthly_convergence.columns = ['収束日数', '総日数', '収束率(%)']
                    st.dataframe(monthly_convergence, use_container_width=True)
                    
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            st.info("ティッカーコードが正しいか確認してください。日本株の場合は '.T' を付けてください（例: 5247.T）")

else:
    # 初期画面の説明
    st.info("👈 左側のサイドバーでティッカーコードを入力し、「収束検出実行」ボタンをクリックしてください")
    
    with st.expander("📖 使い方"):
        st.markdown("""
        ### このアプリケーションについて
        株価の収束（レンジ相場）を自動的に検出し、数値化して分析します。
        
        ### 収束の判定基準
        - **変動係数（CV）**: 価格のばらつきを平均値で正規化した指標
        - **レンジ比率**: 日中の値幅を終値で正規化した指標
        - 両指標が閾値以下の期間を「収束」と判定
        
        ### ティッカーコードの例
        - 日本株: 5247.T, 7203.T (トヨタ), 9984.T (ソフトバンクG)
        - 米国株: AAPL (Apple), GOOGL (Google), TSLA (Tesla)
        
        ### 特徴
        - 異なる価格帯の銘柄でも同じ基準で比較可能
        - 収束期間を自動検出してハイライト表示
        - CSVエクスポート機能付き
        """)