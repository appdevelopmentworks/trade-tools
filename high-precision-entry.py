import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import warnings
#自作ライブラリー
from mystock_lib import *

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="高精度早期エントリーポイント検出システム",
    page_icon="📈",
    layout="wide"
)

# カスタムCSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5px 15px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

class StreamlitEntryDetector:
    """Streamlit用エントリーポイント検出システム"""
    
    def __init__(self):
        self.df = None
        self.actual_symbol = None
        self.company_name = None  # 企業名を追加
        self.entry_points = {
            'aggressive': [],
            'standard': [],
            'conservative': []
        }
        
    def fetch_data(self, symbol, period='2y'):
        """データ取得と企業名取得"""
        try:
            ticker = yf.Ticker(symbol)
            self.df = ticker.history(period=period, interval='1d')
            
            if not self.df.empty and len(self.df) > 100:
                self.actual_symbol = symbol
                
                # 企業名（shortName）を取得
                try:
                    ticker_info = ticker.info
                    # shortNameがない場合はlongNameを試す
                    self.company_name = ticker_info.get('shortName') or ticker_info.get('longName') or symbol
                    # 企業名が長すぎる場合は短縮
                    if len(self.company_name) > 50:
                        self.company_name = self.company_name[:47] + "..."
                except Exception as e:
                    # 取得できない場合はティッカーコードを使用
                    self.company_name = symbol
                    print(f"企業名取得エラー: {e}")
                
                return True
            return False
        except Exception as e:
            print(f"データ取得エラー: {e}")
            return False
    
    def calculate_indicators(self, short_ema=9, mid_ema=18, long_ema=60, rsi_period=14):
        """カスタマイズ可能なテクニカル指標を計算"""
        
        # カスタム EMA
        self.df[f'EMA_{short_ema}'] = self.df['Close'].ewm(span=short_ema, adjust=False).mean()
        self.df[f'EMA_{mid_ema}'] = self.df['Close'].ewm(span=mid_ema, adjust=False).mean()
        self.df[f'EMA_{long_ema}'] = self.df['Close'].ewm(span=long_ema, adjust=False).mean()
        
        # 200日EMA（固定）
        self.df['EMA_200'] = self.df['Close'].ewm(span=200, adjust=False).mean()
        
        # EMAの傾き
        self.df[f'EMA_{short_ema}_slope'] = self.df[f'EMA_{short_ema}'].pct_change(3) * 100
        self.df[f'EMA_{mid_ema}_slope'] = self.df[f'EMA_{mid_ema}'].pct_change(5) * 100
        
        # RSI（カスタム期間）
        self.df[f'RSI_{rsi_period}'] = self.calculate_rsi(self.df['Close'], rsi_period)
        self.df['RSI_7'] = self.calculate_rsi(self.df['Close'], 7)  # 短期RSI（固定）
        
        # Stochastic RSI
        self.calculate_stochastic_rsi(rsi_period)
        
        # MACD（標準）
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_hist'] = self.df['MACD'] - self.df['Signal']
        
        # 高速MACD
        fast_exp1 = self.df['Close'].ewm(span=5, adjust=False).mean()
        fast_exp2 = self.df['Close'].ewm(span=13, adjust=False).mean()
        self.df['Fast_MACD'] = fast_exp1 - fast_exp2
        self.df['Fast_Signal'] = self.df['Fast_MACD'].ewm(span=6, adjust=False).mean()
        self.df['Fast_MACD_hist'] = self.df['Fast_MACD'] - self.df['Fast_Signal']
        
        # ATR
        self.calculate_atr()
        
        # ボリンジャーバンド
        self.df['BB20_middle'] = self.df['Close'].rolling(window=20).mean()
        bb_std = self.df['Close'].rolling(window=20).std()
        self.df['BB20_upper'] = self.df['BB20_middle'] + (bb_std * 2)
        self.df['BB20_lower'] = self.df['BB20_middle'] - (bb_std * 2)
        self.df['BB_width'] = (self.df['BB20_upper'] - self.df['BB20_lower']) / self.df['BB20_middle'] * 100
        self.df['BB_percent'] = (self.df['Close'] - self.df['BB20_lower']) / (self.df['BB20_upper'] - self.df['BB20_lower'])
        
        # 出来高分析
        self.df['Volume_MA20'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Volume_ratio'] = self.df['Volume'] / self.df['Volume_MA20']
        
        # OBV
        self.calculate_obv()
        
        # サポート・レジスタンス
        self.identify_support_resistance()
        
    def calculate_rsi(self, prices, period=14):
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic_rsi(self, rsi_period):
        """Stochastic RSI計算"""
        rsi = self.df[f'RSI_{rsi_period}']
        rsi_min = rsi.rolling(window=14).min()
        rsi_max = rsi.rolling(window=14).max()
        self.df['StochRSI'] = (rsi - rsi_min) / (rsi_max - rsi_min + 0.00001)
        self.df['StochRSI_K'] = self.df['StochRSI'].rolling(window=3).mean()
        self.df['StochRSI_D'] = self.df['StochRSI_K'].rolling(window=3).mean()
    
    def calculate_atr(self, period=14):
        """ATR計算"""
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df['ATR'] = true_range.rolling(period).mean()
        
    def calculate_obv(self):
        """OBV計算"""
        obv = []
        obv_value = 0
        
        for i in range(len(self.df)):
            if i == 0:
                obv.append(0)
            else:
                if self.df['Close'].iloc[i] > self.df['Close'].iloc[i-1]:
                    obv_value += self.df['Volume'].iloc[i]
                elif self.df['Close'].iloc[i] < self.df['Close'].iloc[i-1]:
                    obv_value -= self.df['Volume'].iloc[i]
                obv.append(obv_value)
        
        self.df['OBV'] = obv
        self.df['OBV_MA'] = self.df['OBV'].rolling(window=20).mean()
        
    def identify_support_resistance(self):
        """サポート・レジスタンスレベル"""
        recent_highs = self.df['High'].rolling(window=20).max()
        recent_lows = self.df['Low'].rolling(window=20).min()
        self.df['Resistance'] = recent_highs
        self.df['Support'] = recent_lows
        
    def detect_entry_points(self, short_ema, mid_ema, long_ema, rsi_period):
        """エントリーポイント検出（カスタムパラメータ対応）"""
        
        for i in range(50, len(self.df)):
            current = self.df.iloc[i]
            prev1 = self.df.iloc[i-1]
            prev2 = self.df.iloc[i-2]
            prev5 = self.df.iloc[i-5]
            
            # NaN値のチェック
            if pd.isna(current['RSI_7']) or pd.isna(current['MACD']):
                continue
            
            # === 積極的エントリー ===
            aggressive_conditions = {
                'oversold_bounce': (
                    current['RSI_7'] > 30 and prev1['RSI_7'] < 30 and
                    current['Close'] > prev1['Close']
                ),
                'stoch_rsi_cross': (
                    not pd.isna(current['StochRSI_K']) and
                    current['StochRSI_K'] > current['StochRSI_D'] and
                    prev1['StochRSI_K'] <= prev1['StochRSI_D'] and
                    current['StochRSI_K'] < 0.3
                ),
                'fast_macd_turn': (
                    current['Fast_MACD_hist'] > prev1['Fast_MACD_hist'] and
                    prev2['Fast_MACD_hist'] < prev1['Fast_MACD_hist']
                ),
                'volume_spike': (
                    current['Volume_ratio'] > 1.5 and
                    current['Close'] > current['Open']
                ),
                'bb_squeeze_release': (
                    not pd.isna(current['BB_width']) and
                    current['BB_width'] > prev5['BB_width'] and
                    current['BB_percent'] > 0.2 and prev1['BB_percent'] < 0.2
                )
            }
            
            aggressive_score = sum([
                aggressive_conditions['oversold_bounce'] * 2,
                aggressive_conditions['stoch_rsi_cross'] * 2,
                aggressive_conditions['fast_macd_turn'] * 1.5,
                aggressive_conditions['volume_spike'] * 1,
                aggressive_conditions['bb_squeeze_release'] * 1.5
            ])
            
            if aggressive_score >= 3:
                self.entry_points['aggressive'].append({
                    'date': self.df.index[i],
                    'price': current['Close'],
                    'type': 'aggressive',
                    'score': aggressive_score,
                    'conditions': aggressive_conditions
                })
            
            # === 標準エントリー ===
            standard_conditions = {
                'ema_support': (
                    current['Low'] <= current[f'EMA_{mid_ema}'] * 1.005 and
                    current['Close'] > current[f'EMA_{mid_ema}'] and
                    current[f'EMA_{mid_ema}_slope'] > -0.5
                ),
                'rsi_goldilocks': (
                    35 < current[f'RSI_{rsi_period}'] < 55 and
                    current[f'RSI_{rsi_period}'] > prev1[f'RSI_{rsi_period}']
                ),
                'macd_cross': (
                    current['MACD'] > current['Signal'] and
                    prev1['MACD'] <= prev1['Signal']
                ),
                'trend_alignment': (
                    current[f'EMA_{short_ema}'] > current[f'EMA_{mid_ema}'] and
                    current['Close'] > current[f'EMA_{short_ema}']
                ),
                'obv_confirmation': (
                    not pd.isna(current['OBV']) and
                    current['OBV'] > current['OBV_MA']
                )
            }
            
            standard_score = sum([
                standard_conditions['ema_support'] * 2,
                standard_conditions['rsi_goldilocks'] * 1.5,
                standard_conditions['macd_cross'] * 2,
                standard_conditions['trend_alignment'] * 1.5,
                standard_conditions['obv_confirmation'] * 1
            ])
            
            if standard_score >= 4:
                self.entry_points['standard'].append({
                    'date': self.df.index[i],
                    'price': current['Close'],
                    'type': 'standard',
                    'score': standard_score,
                    'conditions': standard_conditions
                })
            
            # === 保守的エントリー ===
            conservative_conditions = {
                'strong_trend': (
                    current[f'EMA_{short_ema}'] > current[f'EMA_{mid_ema}'] > current[f'EMA_{long_ema}'] and
                    current[f'EMA_{mid_ema}_slope'] > 0.5
                ),
                'pullback_complete': (
                    prev2['Close'] < prev2[f'EMA_{mid_ema}'] and
                    prev1['Close'] < prev1[f'EMA_{mid_ema}'] and
                    current['Close'] > current[f'EMA_{mid_ema}'] and
                    current[f'RSI_{rsi_period}'] > 45
                ),
                'volume_confirmation': (
                    current['Volume_ratio'] > 1.2 and
                    current['Volume'] > prev1['Volume']
                ),
                'momentum_strong': (
                    current['MACD_hist'] > 0 and
                    current['MACD_hist'] > prev1['MACD_hist']
                ),
                'above_support': (
                    not pd.isna(current['Support']) and
                    current['Close'] > current['Support'] * 1.02
                )
            }
            
            conservative_score = sum([
                conservative_conditions['strong_trend'] * 2,
                conservative_conditions['pullback_complete'] * 2.5,
                conservative_conditions['volume_confirmation'] * 1,
                conservative_conditions['momentum_strong'] * 1.5,
                conservative_conditions['above_support'] * 1
            ])
            
            if conservative_score >= 5:
                self.entry_points['conservative'].append({
                    'date': self.df.index[i],
                    'price': current['Close'],
                    'type': 'conservative',
                    'score': conservative_score,
                    'conditions': conservative_conditions
                })
    
    def create_chart(self, short_ema, mid_ema, long_ema, rsi_period, selected_entry_types):
        """チャート作成（選択されたエントリータイプのみ表示）"""
        
        # データフレームのコピーを作成して、インデックスを文字列に変換（休日の歯抜けを防ぐ）
        df_copy = self.df.copy()
        df_copy.index = pd.to_datetime(df_copy.index).strftime('%m-%d-%Y')
        
        # サブプロット作成
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
            subplot_titles=(f'{self.company_name or self.actual_symbol}の高精度エントリーポイント',
                          'Volume & OBV',
                          f'RSI({rsi_period}) & StochRSI',
                          'MACD',
                          'ATR & BB Width'),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": True}],   # 2段目: Volume & OBV
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": True}]]   # 5段目: ATR & BB Width
        )
        
        # 1. メインチャート
        fig.add_trace(
            go.Candlestick(
                x=df_copy.index,
                open=df_copy['Open'],
                high=df_copy['High'],
                low=df_copy['Low'],
                close=df_copy['Close'],
                name=self.company_name or self.actual_symbol,
                increasing_line_color='red',
                decreasing_line_color='blue',
                xaxis='x',  # 明示的にx軸を指定
                yaxis='y'   # 明示的にy軸を指定
            ),
            row=1, col=1
        )
        
        # EMAライン（カスタム期間）
        ema_configs = [
            (f'EMA_{short_ema}', 'orange', 1, '短期EMA'),
            (f'EMA_{mid_ema}', 'green', 2, '中期EMA'),
            (f'EMA_{long_ema}', 'purple', 1, '長期EMA'),
            ('EMA_200', 'gray', 1, 'EMA200')
        ]
        
        for ema_col, color, width, label in ema_configs:
            if ema_col in df_copy.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_copy.index,
                        y=df_copy[ema_col],
                        name=label,
                        line=dict(color=color, width=width),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # ボリンジャーバンド
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['BB20_upper'],
                name='BB Upper',
                line=dict(color='lightgray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['BB20_lower'],
                name='BB Lower',
                line=dict(color='lightgray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(200,200,200,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # エントリーポイント（選択されたタイプのみ表示）
        entry_configs = [
            ('aggressive', 'red', 'star', 20, '積極的'),
            ('standard', 'orange', 'triangle-up', 15, '標準'),
            ('conservative', 'green', 'diamond', 12, '保守的')
        ]
        
        # 日本語から英語キーへのマッピング
        type_mapping = {
            '積極的': 'aggressive',
            '標準': 'standard',
            '保守的': 'conservative'
        }
        
        # 選択されたタイプの英語キーリストを作成
        selected_types = [type_mapping[t] for t in selected_entry_types if t in type_mapping]
        
        for entry_type, color, symbol, size, label in entry_configs:
            # 選択されたタイプのみ表示
            if entry_type in selected_types and self.entry_points[entry_type]:
                dates = [pd.to_datetime(ep['date']).strftime('%m-%d-%Y') for ep in self.entry_points[entry_type]]
                prices = [ep['price'] for ep in self.entry_points[entry_type]]
                scores = [ep['score'] for ep in self.entry_points[entry_type]]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=prices,
                        mode='markers',
                        name=f'{label}エントリー',
                        marker=dict(
                            size=size,
                            color=color,
                            symbol=symbol,
                            line=dict(color='white', width=1)
                        ),
                        text=[f"Score: {s:.1f}" for s in scores],
                        hovertemplate='%{x}<br>価格: %{y:.2f}<br>%{text}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 2. 出来高とOBV
        colors = ['red' if close >= open else 'blue' 
                 for close, open in zip(df_copy['Close'], df_copy['Open'])]
        
        # 出来高バーチャート（プライマリY軸）
        fig.add_trace(
            go.Bar(
                x=df_copy.index,
                y=df_copy['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5,
                showlegend=True,
                xaxis='x2',  # 明示的にx2軸を指定
                yaxis='y2'   # 明示的にy2軸を指定
            ),
            row=2, col=1,
            secondary_y=False
        )
        
        # OBVライン（セカンダリY軸）
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['OBV'],
                name='OBV',
                line=dict(color='purple', width=1.5),
                showlegend=True
            ),
            row=2, col=1,
            secondary_y=True
        )
        
        # OBV移動平均線（セカンダリY軸）
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['OBV_MA'],
                name='OBV MA(20)',
                line=dict(color='magenta', width=1, dash='dash'),
                showlegend=True
            ),
            row=2, col=1,
            secondary_y=True
        )
        
        # 3. RSI
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy[f'RSI_{rsi_period}'],
                name=f'RSI({rsi_period})',
                line=dict(color='blue', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['StochRSI_K'] * 100,
                name='StochRSI',
                line=dict(color='orange', width=1, dash='dot')
            ),
            row=3, col=1
        )
        
        # RSI基準線
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, line_width=0.5)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1, line_width=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, line_width=0.5)
        
        # 4. MACD
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['MACD'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['Signal'],
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=4, col=1
        )
        
        # MACDヒストグラム
        colors = ['green' if val >= 0 else 'red' for val in df_copy['MACD_hist']]
        fig.add_trace(
            go.Bar(
                x=df_copy.index,
                y=df_copy['MACD_hist'],
                name='MACD Hist',
                marker_color=colors,
                opacity=0.3,
                showlegend=False
            ),
            row=4, col=1
        )
        
        # 5. ATR
        # ATR（プライマリY軸）
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['ATR'],
                name='ATR',
                line=dict(color='orange', width=1),
                showlegend=True
            ),
            row=5, col=1,
            secondary_y=False
        )
        
        # BB Width（セカンダリY軸）
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['BB_width'],
                name='BB Width (%)',
                line=dict(color='purple', width=1),
                showlegend=True
            ),
            row=5, col=1,
            secondary_y=True
        )
        
        # レイアウト設定
        fig.update_layout(
            title={
                'text': f'{self.company_name or self.actual_symbol} ({self.actual_symbol}) - 高精度早期エントリーポイント検出システム',
                'font': {'size': 20}
            },
            xaxis_rangeslider_visible=False,  # 1段目のレンジスライダーを非表示
            xaxis2_rangeslider_visible=False,  # 2段目のレンジスライダーを明示的に非表示
            xaxis3_rangeslider_visible=False,  # 3段目のレンジスライダーを明示的に非表示
            xaxis4_rangeslider_visible=False,  # 4段目のレンジスライダーを明示的に非表示
            xaxis5_rangeslider_visible=False,  # 5段目のレンジスライダーを明示的に非表示
            height=900,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # X軸設定（全ての段で統一）
        for i in range(1, 6):
            fig.update_xaxes(
                type='category',
                tickmode='linear',
                tick0=0,
                dtick=20,
                tickangle=45,
                row=i, col=1,
                rangeslider_visible=False  # 各段でレンジスライダーを非表示
            )
        
        # Y軸ラベル
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="OBV", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="RSI/StochRSI", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        fig.update_yaxes(title_text="ATR", row=5, col=1, secondary_y=False)
        fig.update_yaxes(title_text="BB Width (%)", row=5, col=1, secondary_y=True)
        
        return fig
    
    def analyze_performance(self, selected_entry_types):
        """パフォーマンス分析（選択されたタイプのみ）"""
        performance_stats = {}
        
        # 日本語から英語キーへのマッピング
        type_mapping = {
            '積極的': 'aggressive',
            '標準': 'standard',
            '保守的': 'conservative'
        }
        
        # 選択されたタイプの英語キーリストを作成
        selected_types = [type_mapping[t] for t in selected_entry_types if t in type_mapping]
        
        for entry_type in selected_types:
            if not self.entry_points[entry_type]:
                continue
                
            profits = []
            for ep in self.entry_points[entry_type]:
                try:
                    entry_idx = self.df.index.get_loc(ep['date'])
                    if entry_idx + 20 < len(self.df):
                        entry_price = ep['price']
                        future_prices = self.df['High'].iloc[entry_idx+1:entry_idx+21]
                        max_profit = ((future_prices.max() - entry_price) / entry_price) * 100
                        profits.append(max_profit)
                except:
                    continue
            
            if profits:
                performance_stats[entry_type] = {
                    'count': len(self.entry_points[entry_type]),
                    'avg_profit': np.mean(profits),
                    'max_profit': np.max(profits),
                    'min_profit': np.min(profits),
                    'win_rate': len([p for p in profits if p > 0]) / len(profits) * 100
                }
        
        return performance_stats

# メインアプリケーション
# タイトル
st.title("📈 高精度早期エントリーポイント検出システム")
st.markdown("---")

# サイドバーに入力コントロール
with st.sidebar:
    st.header("⚙️ パラメータ設定")
    
    # ティッカーコード入力
    ticker = checkTicker(st.text_input(
        "ティッカーコード",
        value="^N225",
        help="例: ^N225, AAPL, TSLA, 7203"
    ))
    
    # 期間選択
    period = st.selectbox(
        "データ期間",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=4,
        help="分析対象期間を選択"
    )
    
    st.markdown("### 📊 移動平均線設定")
    
    # 移動平均期間入力
    col1, col2 = st.columns(2)
    with col1:
        short_ema = st.number_input(
            "短期EMA",
            min_value=5,
            max_value=50,
            value=9,
            step=1,
            help="短期の指数移動平均期間"
        )
        long_ema = st.number_input(
            "長期EMA",
            min_value=20,
            max_value=100,
            value=60,
            step=1,
            help="長期の指数移動平均期間"
        )
    
    with col2:
        mid_ema = st.number_input(
            "中期EMA",
            min_value=10,
            max_value=50,
            value=18,
            step=1,
            help="中期の指数移動平均期間"
        )
        rsi_period = st.number_input(
            "RSI期間",
            min_value=7,
            max_value=21,
            value=14,
            step=1,
            help="RSIの計算期間"
        )
    
    st.markdown("### 📋 その他の設定")
    
    # エントリータイプ選択
    entry_types = st.multiselect(
        "表示するエントリータイプ",
        options=["積極的", "標準", "保守的"],
        default=["積極的", "標準", "保守的"],
        help="チャートに表示するエントリーポイントのタイプ"
    )
    
    # 実行ボタン
    st.markdown("---")
    execute = st.button("🚀 分析実行", use_container_width=True)

# メインエリア
if execute:
    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # インスタンス作成
    detector = StreamlitEntryDetector()
    
    # データ取得
    status_text.text(f"{ticker}のデータを取得中...")
    progress_bar.progress(20)
    
    if detector.fetch_data(ticker, period):
        # 企業名を取得できた場合は表示
        if detector.company_name and detector.company_name != ticker:
            status_text.text(f"{detector.company_name} ({ticker})のテクニカル指標を計算中...")
        else:
            status_text.text(f"{ticker}のテクニカル指標を計算中...")
        progress_bar.progress(40)
        detector.calculate_indicators(short_ema, mid_ema, long_ema, rsi_period)
        
        # エントリーポイント検出
        status_text.text("エントリーポイントを検出中...")
        progress_bar.progress(60)
        detector.detect_entry_points(short_ema, mid_ema, long_ema, rsi_period)
        
        # チャート作成（選択されたエントリータイプを渡す）
        status_text.text("チャートを作成中...")
        progress_bar.progress(80)
        fig = detector.create_chart(short_ema, mid_ema, long_ema, rsi_period, entry_types)
        
        # パフォーマンス分析（選択されたタイプのみ）
        status_text.text("パフォーマンスを分析中...")
        progress_bar.progress(90)
        performance = detector.analyze_performance(entry_types)
        
        # 完了
        progress_bar.progress(100)
        if detector.company_name and detector.company_name != ticker:
            status_text.text(f"{detector.company_name} ({ticker}) の分析完了！")
        else:
            status_text.text(f"{ticker} の分析完了！")
        
        # 基本情報表示
        col1, col2, col3, col4 = st.columns(4)
        
        latest_price = detector.df['Close'].iloc[-1]
        prev_close = detector.df['Close'].iloc[-2]
        price_change = latest_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        with col1:
            # 企業名がある場合は表示
            display_name = detector.company_name if detector.company_name else ticker
            st.metric(
                f"{display_name}",
                f"{latest_price:,.2f}",
                f"{price_change:+,.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            # 選択されたタイプのエントリー数をカウント
            type_mapping = {
                '積極的': 'aggressive',
                '標準': 'standard',
                '保守的': 'conservative'
            }
            selected_types = [type_mapping[t] for t in entry_types if t in type_mapping]
            total_entries = sum(len(detector.entry_points[t]) for t in selected_types if t in detector.entry_points)
            st.metric("選択タイプのエントリー数", total_entries)
        
        with col3:
            # 積極的タイプが選択されている場合のみ表示
            if '積極的' in entry_types and detector.entry_points['aggressive']:
                last_aggressive = detector.entry_points['aggressive'][-1]['date']
                st.metric("直近の積極的エントリー", last_aggressive.strftime('%Y-%m-%d'))
            elif '積極的' in entry_types:
                st.metric("直近の積極的エントリー", "なし")
            else:
                st.metric("直近の積極的エントリー", "未選択")
        
        with col4:
            current_rsi = detector.df[f'RSI_{rsi_period}'].iloc[-1]
            st.metric(f"現在のRSI({rsi_period})", f"{current_rsi:.2f}")
        
        # チャート表示
        st.plotly_chart(fig, use_container_width=True)
        
        # 結果レポート
        st.markdown("---")
        st.header("📊 分析結果レポート")
        
        # パフォーマンスサマリー（選択されたタイプのみ）
        if performance:
            st.subheader("📈 エントリーポイント別パフォーマンス")
            
            # 日本語ラベルを作成
            type_labels = {
                'aggressive': '積極的',
                'standard': '標準',
                'conservative': '保守的'
            }
            
            perf_df = pd.DataFrame(performance).T
            perf_df.index = [type_labels[idx] for idx in perf_df.index if idx in type_labels]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    perf_df[['count', 'win_rate', 'avg_profit']].round(2),
                    use_container_width=True
                )
            
            with col2:
                # パフォーマンスチャート
                colors = {
                    '積極的': 'red',
                    '標準': 'orange',
                    '保守的': 'green'
                }
                bar_colors = [colors.get(idx, 'gray') for idx in perf_df.index]
                
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Bar(
                    x=perf_df.index,
                    y=perf_df['avg_profit'],
                    text=perf_df['avg_profit'].round(2),
                    textposition='auto',
                    marker_color=bar_colors
                ))
                fig_perf.update_layout(
                    title="平均利益率 (%)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_perf, use_container_width=True)
        
        # 直近のエントリーポイント（選択されたタイプのみ）
        st.subheader("📍 直近のエントリーポイント (上位10件)")
        
        # 日本語から英語キーへのマッピング
        type_mapping = {
            '積極的': 'aggressive',
            '標準': 'standard',
            '保守的': 'conservative'
        }
        
        # 選択されたタイプの英語キーリストを作成
        selected_types = [type_mapping[t] for t in entry_types if t in type_mapping]
        
        all_entries = []
        for entry_type in selected_types:
            if entry_type in detector.entry_points:
                for ep in detector.entry_points[entry_type]:
                    ep['entry_type'] = entry_type
                    all_entries.append(ep)
        
        if all_entries:
            all_entries.sort(key=lambda x: x['date'], reverse=True)
            
            recent_entries = []
            for ep in all_entries[:10]:
                recent_entries.append({
                    '日付': ep['date'].strftime('%Y-%m-%d'),
                    'タイプ': {'aggressive': '積極的', 'standard': '標準', 'conservative': '保守的'}[ep['entry_type']],
                    '価格': f"{ep['price']:,.2f}",
                    'スコア': f"{ep['score']:.1f}"
                })
            
            st.dataframe(pd.DataFrame(recent_entries), use_container_width=True)
        else:
            st.info("選択されたタイプのエントリーポイントが検出されませんでした。")
        
        # 現在の市場状況
        st.subheader("🔍 現在の市場状況分析")
        
        current_data = detector.df.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**トレンド状況**")
            if current_data[f'EMA_{short_ema}'] > current_data[f'EMA_{mid_ema}'] > current_data[f'EMA_{long_ema}']:
                st.success("上昇トレンド ↗️")
            elif current_data[f'EMA_{short_ema}'] < current_data[f'EMA_{mid_ema}'] < current_data[f'EMA_{long_ema}']:
                st.error("下降トレンド ↘️")
            else:
                st.warning("横ばい →")
        
        with col2:
            st.markdown("**RSI状態**")
            if current_rsi > 70:
                st.error("買われ過ぎ")
            elif current_rsi < 30:
                st.success("売られ過ぎ")
            else:
                st.info("中立")
        
        with col3:
            st.markdown("**ボラティリティ**")
            current_atr = detector.df['ATR'].iloc[-1]
            avg_atr = detector.df['ATR'].mean()
            if current_atr > avg_atr * 1.2:
                st.warning("高ボラティリティ")
            else:
                st.info("通常")
        
        # エクスポート機能
        st.markdown("---")
        st.subheader("💾 データエクスポート")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSVダウンロード
            csv = detector.df.to_csv()
            # ファイル名に企業名を使用（スペースや特殊文字を除去）
            safe_name = (detector.company_name or ticker).replace(' ', '_').replace('/', '_').replace('.', '')
            st.download_button(
                label="📥 価格データをCSVでダウンロード",
                data=csv,
                file_name=f"{safe_name}_price_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # エントリーポイントのダウンロード（選択されたタイプのみ）
            if all_entries:
                entry_df = pd.DataFrame([{
                    '日付': ep['date'],
                    'タイプ': {'aggressive': '積極的', 'standard': '標準', 'conservative': '保守的'}[ep['entry_type']],
                    '価格': ep['price'],
                    'スコア': ep['score']
                } for ep in all_entries])
                
                csv_entries = entry_df.to_csv(index=False)
                # ファイル名に企業名を使用（スペースや特殊文字を除去）
                safe_name = (detector.company_name or ticker).replace(' ', '_').replace('/', '_').replace('.', '')
                st.download_button(
                    label="📥 選択タイプのエントリーポイントをCSVでダウンロード",
                    data=csv_entries,
                    file_name=f"{safe_name}_selected_entry_points.csv",
                    mime="text/csv"
                )
    else:
        st.error(f"❌ {ticker} のデータ取得に失敗しました。ティッカーコードを確認してください。")
        st.info("💡 ヒント: 日経225の場合は ^N225、米国株の場合は AAPL や TSLA などを入力してください。")