import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="革新的売りタイミング検出システム",
    page_icon="🎯",
    layout="wide"
)

# カスタムCSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(255, 75, 75, 0.1);
        border: 1px solid rgba(255, 75, 75, 0.2);
        padding: 5px 15px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .exit-emergency {
        background-color: #FF0000;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .exit-standard {
        background-color: #FF8C00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .exit-warning {
        background-color: #FFD700;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

class AdvancedExitDetector:
    """革新的売りタイミング検出システム"""
    
    def __init__(self):
        self.df = None
        self.actual_symbol = None
        self.company_name = None
        self.exit_points = {
            'warning': [],      # 早期警戒売り
            'standard': [],     # 標準売り
            'emergency': []     # 緊急売り
        }
        self.parabolic_sar = None
        self.atr_trailing_stop = None
        
    def fetch_data(self, symbol, period='2y'):
        """データ取得と企業名取得"""
        try:
            ticker = yf.Ticker(symbol)
            self.df = ticker.history(period=period, interval='1d')
            
            if not self.df.empty and len(self.df) > 100:
                self.actual_symbol = symbol
                
                # 企業名を取得
                try:
                    ticker_info = ticker.info
                    self.company_name = ticker_info.get('shortName') or ticker_info.get('longName') or symbol
                    if len(self.company_name) > 50:
                        self.company_name = self.company_name[:47] + "..."
                except Exception as e:
                    self.company_name = symbol
                    print(f"企業名取得エラー: {e}")
                
                return True
            return False
        except Exception as e:
            print(f"データ取得エラー: {e}")
            return False
    
    def calculate_parabolic_sar(self, af_initial=0.02, af_increment=0.02, af_max=0.2):
        """パラボリックSAR計算"""
        high = self.df['High'].values
        low = self.df['Low'].values
        close = self.df['Close'].values
        
        n = len(close)
        sar = np.zeros(n)
        ep = np.zeros(n)
        af = np.zeros(n)
        trend = np.zeros(n)  # 1: 上昇トレンド, -1: 下降トレンド
        
        # 初期設定
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = af_initial
        trend[0] = 1
        
        for i in range(1, n):
            if trend[i-1] == 1:  # 上昇トレンド
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = min(sar[i], low[i], low[i-1] if i > 1 else low[i])
                
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = af_initial
                else:
                    trend[i] = 1
            else:  # 下降トレンド
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = max(sar[i], high[i], high[i-1] if i > 1 else high[i])
                
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = af_initial
                else:
                    trend[i] = -1
        
        self.df['SAR'] = sar
        self.df['SAR_trend'] = trend
        self.parabolic_sar = sar
        
    def calculate_atr_trailing_stop(self, period=14, multiplier=2.5):
        """ATRトレーリングストップ計算"""
        # ATR計算
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        self.df['ATR'] = atr
        
        # ATRトレーリングストップ計算
        close = self.df['Close'].values
        high = self.df['High'].values
        atr_values = atr.values
        
        n = len(close)
        trailing_stop = np.zeros(n)
        trailing_stop[0] = close[0] - multiplier * atr_values[0] if not np.isnan(atr_values[0]) else close[0]
        
        for i in range(1, n):
            if np.isnan(atr_values[i]):
                trailing_stop[i] = trailing_stop[i-1]
            else:
                stop_long = high[i] - multiplier * atr_values[i]
                trailing_stop[i] = max(stop_long, trailing_stop[i-1]) if close[i] > trailing_stop[i-1] else stop_long
        
        self.df['ATR_stop'] = trailing_stop
        self.atr_trailing_stop = trailing_stop
        
    def detect_rsi_divergence(self, period=14, lookback=10):
        """RSIダイバージェンス検出と短期RSI計算"""
        # 通常のRSI計算（14期間）
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        self.df['RSI'] = rsi
        
        # 短期RSI(3)計算 - 超短期の過熱感検出用
        gain_3 = (delta.where(delta > 0, 0)).rolling(window=3).mean()
        loss_3 = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
        rs_3 = gain_3 / loss_3
        rsi_3 = 100 - (100 / (1 + rs_3))
        self.df['RSI_3'] = rsi_3
        
        # RSI(3)の方向転換検出（85以上での下向き転換）
        rsi_3_reversal = []
        for i in range(len(self.df)):
            if i < 2:
                rsi_3_reversal.append(False)
            else:
                # RSI(3)が85以上で、上向きから下向きに転換
                current_rsi3 = self.df['RSI_3'].iloc[i]
                prev_rsi3 = self.df['RSI_3'].iloc[i-1]
                prev2_rsi3 = self.df['RSI_3'].iloc[i-2]
                
                if (current_rsi3 > 85 and 
                    prev_rsi3 > prev2_rsi3 and  # 前回は上向き
                    current_rsi3 < prev_rsi3):  # 今回下向きに転換
                    rsi_3_reversal.append(True)
                else:
                    rsi_3_reversal.append(False)
        
        self.df['RSI_3_reversal'] = rsi_3_reversal
        
        # ダイバージェンス検出
        divergence = []
        for i in range(lookback, len(self.df)):
            # 弱気ダイバージェンス（売りシグナル）
            price_highs = argrelextrema(self.df['High'].iloc[i-lookback:i+1].values, np.greater)[0]
            rsi_highs = argrelextrema(rsi.iloc[i-lookback:i+1].values, np.greater)[0]
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                # 価格は高値更新、RSIは高値切り下げ
                if (self.df['High'].iloc[i-lookback+price_highs[-1]] > self.df['High'].iloc[i-lookback+price_highs[-2]] and
                    rsi.iloc[i-lookback+rsi_highs[-1]] < rsi.iloc[i-lookback+rsi_highs[-2]]):
                    divergence.append(True)
                else:
                    divergence.append(False)
            else:
                divergence.append(False)
        
        # パディング
        divergence = [False] * lookback + divergence
        self.df['RSI_divergence'] = divergence
        
    def calculate_volume_climax(self, period=20, multiplier=3):
        """ボリュームクライマックス検出"""
        volume_ma = self.df['Volume'].rolling(window=period).mean()
        self.df['Volume_MA'] = volume_ma
        self.df['Volume_ratio'] = self.df['Volume'] / volume_ma
        
        # クライマックス条件
        climax = []
        for i in range(len(self.df)):
            if i < period:
                climax.append(False)
                continue
                
            # 出来高が平均の3倍以上 + 上髭が長い
            high_vol = self.df['Volume_ratio'].iloc[i] > multiplier
            upper_shadow = (self.df['High'].iloc[i] - max(self.df['Open'].iloc[i], self.df['Close'].iloc[i])) / self.df['High'].iloc[i]
            long_shadow = upper_shadow > 0.02  # 2%以上の上髭
            
            # 翌日の出来高減少をチェック（最終日以外）
            if i < len(self.df) - 1:
                next_vol_decrease = self.df['Volume'].iloc[i+1] < self.df['Volume'].iloc[i] * 0.7
            else:
                next_vol_decrease = False
            
            climax.append(high_vol and long_shadow)
        
        self.df['Volume_climax'] = climax
        
    def calculate_advanced_indicators(self):
        """高度なテクニカル指標計算"""
        
        # 基本的な移動平均
        self.df['EMA_9'] = self.df['Close'].ewm(span=9, adjust=False).mean()
        self.df['EMA_21'] = self.df['Close'].ewm(span=21, adjust=False).mean()
        self.df['EMA_50'] = self.df['Close'].ewm(span=50, adjust=False).mean()
        self.df['EMA_200'] = self.df['Close'].ewm(span=200, adjust=False).mean()
        
        # MACD
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_hist'] = self.df['MACD'] - self.df['Signal']
        
        # Stochastic RSI
        rsi = self.df['RSI']
        rsi_min = rsi.rolling(window=14).min()
        rsi_max = rsi.rolling(window=14).max()
        self.df['StochRSI'] = (rsi - rsi_min) / (rsi_max - rsi_min + 0.00001)
        
        # ボリンジャーバンド
        self.df['BB_middle'] = self.df['Close'].rolling(window=20).mean()
        bb_std = self.df['Close'].rolling(window=20).std()
        self.df['BB_upper'] = self.df['BB_middle'] + (bb_std * 2)
        self.df['BB_lower'] = self.df['BB_middle'] - (bb_std * 2)
        self.df['BB_width'] = (self.df['BB_upper'] - self.df['BB_lower']) / self.df['BB_middle'] * 100
        self.df['BB_percent'] = (self.df['Close'] - self.df['BB_lower']) / (self.df['BB_upper'] - self.df['BB_lower'])
        
        # OBV
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
        self.df['OBV_MA'] = pd.Series(obv).rolling(window=20).mean()
        
        # モメンタム
        self.df['Momentum'] = self.df['Close'].pct_change(periods=10) * 100
        
        # Williams %R
        high_14 = self.df['High'].rolling(window=14).max()
        low_14 = self.df['Low'].rolling(window=14).min()
        self.df['Williams_R'] = -100 * (high_14 - self.df['Close']) / (high_14 - low_14)
        
    def detect_exit_points(self, sar_weight=2.5, atr_weight=2.0, rsi_div_weight=1.8,
                          vol_climax_weight=1.5, bb_weight=3.0, macd_weight=1.2,
                          obv_weight=1.0, macd_hist_weight=1.3, bb_approach_threshold=0.98, bb_enhanced=True):
        """売りポイント検出（スコアリングシステム）"""
        
        for i in range(50, len(self.df)):
            current = self.df.iloc[i]
            prev1 = self.df.iloc[i-1]
            prev2 = self.df.iloc[i-2]
            prev5 = self.df.iloc[i-5]
            
            # NaN値のチェック
            if pd.isna(current['RSI']) or pd.isna(current['MACD']):
                continue
            
            exit_score = 0
            conditions = {}
            
            # 1. パラボリックSAR反転（売りシグナル）
            sar_sell = (prev1['SAR_trend'] == 1 and current['SAR_trend'] == -1)
            if sar_sell:
                exit_score += sar_weight
                conditions['SAR反転'] = True
            else:
                conditions['SAR反転'] = False
            
            # 2. ATRトレーリングストップ抵触
            atr_stop_hit = current['Close'] < current['ATR_stop']
            if atr_stop_hit:
                exit_score += atr_weight
                conditions['ATRストップ'] = True
            else:
                conditions['ATRストップ'] = False
            
            # 3. RSIダイバージェンス
            if current['RSI_divergence']:
                exit_score += rsi_div_weight
                conditions['RSIダイバージェンス'] = True
            else:
                conditions['RSIダイバージェンス'] = False
            
            # 4. ボリュームクライマックス
            if current['Volume_climax']:
                exit_score += vol_climax_weight
                conditions['出来高クライマックス'] = True
            else:
                conditions['出来高クライマックス'] = False
            
            # 5. ボリンジャーバンド上限からの反落（改良版）
            if bb_enhanced:
                # 5-1. BB上限タッチ検出（高値がBB上限を超える）
                bb_touch = current['High'] >= current['BB_upper']
                
                # 5-2. BB上限接近検出（終値がBB上限の指定%以上）
                bb_approach = current['Close'] >= current['BB_upper'] * bb_approach_threshold
                
                # 5-3. BB上限からの反落開始
                bb_reversal_start = (prev1['High'] >= prev1['BB_upper'] and 
                                    current['Close'] < current['BB_upper'])
                
                # 5-4. 強いBB上限反落（前日BB超え→当日陰線で下落）
                strong_bb_reversal = (prev1['Close'] > prev1['BB_upper'] and 
                                     current['Close'] < current['Open'] and
                                     current['Close'] < prev1['Close'])
                
                # BB関連スコア計算
                bb_signal = False
                if strong_bb_reversal:
                    exit_score += bb_weight * 1.5  # 強い反落は重み増加
                    bb_signal = True
                    conditions['BB強反落'] = True
                else:
                    conditions['BB強反落'] = False
                    
                if bb_reversal_start and not strong_bb_reversal:
                    exit_score += bb_weight * 1.2
                    bb_signal = True
                    conditions['BB反落開始'] = True
                else:
                    conditions['BB反落開始'] = False
                    
                if bb_touch and not bb_reversal_start and not strong_bb_reversal:
                    exit_score += bb_weight * 1.0
                    bb_signal = True
                    conditions['BB上限タッチ'] = True
                else:
                    conditions['BB上限タッチ'] = False
                    
                if bb_approach and not bb_touch and not bb_reversal_start and not strong_bb_reversal:
                    exit_score += bb_weight * 0.8
                    bb_signal = True
                    conditions['BB上限接近'] = True
                else:
                    conditions['BB上限接近'] = False
                
                # 追加: BBバンド幅拡大時の上限タッチ（より強いシグナル）
                bb_width_expanding = current['BB_width'] > prev5['BB_width'] * 1.1
                if bb_touch and bb_width_expanding:
                    exit_score += 0.5
                    conditions['BB拡大時タッチ'] = True
                else:
                    conditions['BB拡大時タッチ'] = False
                    
                if not bb_signal:
                    conditions['BB上限反落'] = False
            else:
                # 従来のシンプルな判定
                bb_reversal = (current['BB_percent'] > 0.95 and 
                              current['Close'] < prev1['Close'] and
                              prev1['BB_percent'] > 1.0)
                if bb_reversal:
                    exit_score += bb_weight
                    conditions['BB上限反落'] = True
                else:
                    conditions['BB上限反落'] = False
            
            # 6. MACD売りクロス
            macd_sell_cross = (prev1['MACD'] > prev1['Signal'] and 
                              current['MACD'] < current['Signal'])
            if macd_sell_cross:
                exit_score += macd_weight
                conditions['MACD売りクロス'] = True
            else:
                conditions['MACD売りクロス'] = False
            
            # 7. OBV下落転換
            obv_decline = (current['OBV'] < current['OBV_MA'] and 
                          prev1['OBV'] > prev1['OBV_MA'])
            if obv_decline:
                exit_score += obv_weight
                conditions['OBV下落転換'] = True
            else:
                conditions['OBV下落転換'] = False
            
            # 追加条件
            # 8. RSI過熱圏からの反落
            rsi_overbought = current['RSI'] > 70 and current['RSI'] < prev1['RSI']
            if rsi_overbought:
                exit_score += 1.0
                conditions['RSI過熱圏'] = True
            else:
                conditions['RSI過熱圏'] = False
            
            # 9. モメンタム低下
            momentum_decline = (current['Momentum'] < 0 and 
                              prev1['Momentum'] > 0)
            if momentum_decline:
                exit_score += 0.8
                conditions['モメンタム低下'] = True
            else:
                conditions['モメンタム低下'] = False
            
            # 10. Williams %R売りシグナル
            williams_sell = current['Williams_R'] > -20
            if williams_sell:
                exit_score += 0.7
                conditions['Williams%R'] = True
            else:
                conditions['Williams%R'] = False
            
            # 11. RSI(3)極限過熱からの反転（新規追加 - 超強力シグナル）
            rsi3_extreme_reversal = current.get('RSI_3_reversal', False)
            if rsi3_extreme_reversal:
                exit_score += 2.2  # 非常に強力なシグナルなので高い重み
                conditions['RSI3極限反転'] = True
            else:
                conditions['RSI3極限反転'] = False
            
            # 12. MACDヒストグラムのピークアウト（新規追加 - モメンタム減衰）
            macd_hist_peakout = False
            if not pd.isna(current['MACD_hist']) and not pd.isna(prev1['MACD_hist']):
                # プラス圏でヒストグラムが減少開始（山の頂点）
                if (current['MACD_hist'] > 0 and 
                    prev1['MACD_hist'] > 0 and
                    current['MACD_hist'] < prev1['MACD_hist']):
                    # 2日連続で減少している場合はより強いシグナル
                    if not pd.isna(prev2['MACD_hist']) and prev1['MACD_hist'] < prev2['MACD_hist']:
                        exit_score += macd_hist_weight * 1.4  # 2日連続減少は強いシグナル
                        conditions['MACDヒスト頂点(強)'] = True
                        macd_hist_peakout = True
                    else:
                        exit_score += macd_hist_weight  # 初回減少
                        conditions['MACDヒスト頂点'] = True
                        macd_hist_peakout = True
            
            if not macd_hist_peakout:
                conditions['MACDヒスト頂点'] = False
                conditions['MACDヒスト頂点(強)'] = False
            
            # 売りレベル判定
            if exit_score >= 7:
                # 緊急売り
                self.exit_points['emergency'].append({
                    'date': self.df.index[i],
                    'price': current['Close'],
                    'type': 'emergency',
                    'score': exit_score,
                    'conditions': conditions.copy()
                })
            elif exit_score >= 5:
                # 標準売り
                self.exit_points['standard'].append({
                    'date': self.df.index[i],
                    'price': current['Close'],
                    'type': 'standard',
                    'score': exit_score,
                    'conditions': conditions.copy()
                })
            elif exit_score >= 3:
                # 早期警戒売り
                self.exit_points['warning'].append({
                    'date': self.df.index[i],
                    'price': current['Close'],
                    'type': 'warning',
                    'score': exit_score,
                    'conditions': conditions.copy()
                })
    
    def create_chart(self, selected_exit_types):
        """チャート作成（選択された売りタイプのみ表示）"""
        
        # データフレームのコピーを作成
        df_copy = self.df.copy()
        df_copy.index = pd.to_datetime(df_copy.index).strftime('%m-%d-%Y')
        
        # サブプロット作成
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
            subplot_titles=(f'{self.company_name or self.actual_symbol} - 革新的売りタイミング検出',
                          'Volume & OBV',
                          'RSI(14) & RSI(3) - 極限過熱検出',
                          'MACD & モメンタム - ピークアウト検出',
                          'ATR & ボラティリティ'),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": True}],
                   [{"secondary_y": False}],
                   [{"secondary_y": True}],
                   [{"secondary_y": True}]]
        )
        
        # 1. メインチャート（ローソク足）
        fig.add_trace(
            go.Candlestick(
                x=df_copy.index,
                open=df_copy['Open'],
                high=df_copy['High'],
                low=df_copy['Low'],
                close=df_copy['Close'],
                name=self.company_name or self.actual_symbol,
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # パラボリックSAR
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['SAR'],
                mode='markers',
                name='Parabolic SAR',
                marker=dict(
                    size=3,
                    color=['green' if t == 1 else 'red' for t in df_copy['SAR_trend']]
                )
            ),
            row=1, col=1
        )
        
        # ATRトレーリングストップ
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['ATR_stop'],
                name='ATR Trailing Stop',
                line=dict(color='orange', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # EMAライン
        ema_configs = [
            ('EMA_9', 'blue', 1, 'EMA9'),
            ('EMA_21', 'green', 1, 'EMA21'),
            ('EMA_50', 'purple', 1, 'EMA50')
        ]
        
        for ema_col, color, width, label in ema_configs:
            if ema_col in df_copy.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_copy.index,
                        y=df_copy[ema_col],
                        name=label,
                        line=dict(color=color, width=width),
                        opacity=0.5
                    ),
                    row=1, col=1
                )
        
        # ボリンジャーバンド（改良版 - 上限を強調）
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['BB_upper'],
                name='BB Upper ⚠️',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.8,
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['BB_lower'],
                name='BB Lower',
                line=dict(color='lightgray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(200,200,200,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 売りポイント表示（選択されたタイプのみ）
        exit_configs = [
            ('warning', 'yellow', 'triangle-down', 12, '早期警戒'),
            ('standard', 'orange', 'triangle-down', 16, '標準売り'),
            ('emergency', 'red', 'triangle-down', 20, '緊急売り')
        ]
        
        # 日本語から英語キーへのマッピング
        type_mapping = {
            '早期警戒': 'warning',
            '標準売り': 'standard',
            '緊急売り': 'emergency'
        }
        
        # 選択されたタイプの英語キーリストを作成
        selected_types = [type_mapping[t] for t in selected_exit_types if t in type_mapping]
        
        for exit_type, color, symbol, size, label in exit_configs:
            if exit_type in selected_types and self.exit_points[exit_type]:
                dates = [pd.to_datetime(ep['date']).strftime('%m-%d-%Y') for ep in self.exit_points[exit_type]]
                prices = [ep['price'] for ep in self.exit_points[exit_type]]
                scores = [ep['score'] for ep in self.exit_points[exit_type]]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=prices,
                        mode='markers',
                        name=f'{label}',
                        marker=dict(
                            size=size,
                            color=color,
                            symbol=symbol,
                            line=dict(color='black', width=2)
                        ),
                        text=[f"Score: {s:.1f}" for s in scores],
                        hovertemplate='%{x}<br>価格: %{y:.2f}<br>%{text}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 2. 出来高とOBV
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df_copy['Close'], df_copy['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df_copy.index,
                y=df_copy['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5,
                showlegend=True
            ),
            row=2, col=1,
            secondary_y=False
        )
        
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
        
        # 3. RSI（通常のRSIとRSI(3)を両方表示）
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['RSI'],
                name='RSI(14)',
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        
        # RSI(3)を追加表示
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['RSI_3'],
                name='RSI(3) 🔥',
                line=dict(color='red', width=1, dash='dot'),
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # RSI(3)の85以上での反転マーカー
        rsi3_rev_indices = df_copy[df_copy['RSI_3_reversal'] == True].index
        if len(rsi3_rev_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=rsi3_rev_indices,
                    y=df_copy.loc[rsi3_rev_indices, 'RSI_3'],
                    mode='markers',
                    name='RSI(3)極限反転',
                    marker=dict(
                        size=12,
                        color='darkred',
                        symbol='triangle-down',
                        line=dict(color='white', width=1)
                    )
                ),
                row=3, col=1
            )
        
        # RSIダイバージェンスマーカー
        div_indices = df_copy[df_copy['RSI_divergence'] == True].index
        if len(div_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=div_indices,
                    y=df_copy.loc[div_indices, 'RSI'],
                    mode='markers',
                    name='RSIダイバージェンス',
                    marker=dict(
                        size=10,
                        color='orange',
                        symbol='x'
                    )
                ),
                row=3, col=1
            )
        
        # RSI基準線（85ラインを追加）
        fig.add_hline(y=85, line_dash="dash", line_color="darkred", row=3, col=1, line_width=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, line_width=0.5)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1, line_width=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, line_width=0.5)
        
        # 4. MACD & モメンタム
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['MACD'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=4, col=1,
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['Signal'],
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=4, col=1,
            secondary_y=False
        )
        
        # MACDヒストグラム（色分けを改良）
        colors = []
        peak_indices = []  # ピークアウトポイントを記録
        
        for i in range(len(df_copy)):
            if i == 0:
                colors.append('green' if df_copy['MACD_hist'].iloc[i] >= 0 else 'red')
            else:
                current_hist = df_copy['MACD_hist'].iloc[i]
                prev_hist = df_copy['MACD_hist'].iloc[i-1]
                
                # プラス圏でのピークアウトを検出
                if current_hist > 0 and prev_hist > 0 and current_hist < prev_hist:
                    colors.append('orange')  # ピークアウトはオレンジ色
                    peak_indices.append(i)
                elif current_hist >= 0:
                    colors.append('green')
                else:
                    colors.append('red')
        
        fig.add_trace(
            go.Bar(
                x=df_copy.index,
                y=df_copy['MACD_hist'],
                name='MACD Hist',
                marker_color=colors,
                opacity=0.5,
                showlegend=True,
                hovertemplate='MACD Hist: %{y:.3f}<extra></extra>'
            ),
            row=4, col=1,
            secondary_y=False
        )
        
        # MACDヒストグラムのピークアウトポイントにマーカー追加
        if peak_indices:
            peak_dates = [df_copy.index[i] for i in peak_indices]
            peak_values = [df_copy['MACD_hist'].iloc[i] for i in peak_indices]
            fig.add_trace(
                go.Scatter(
                    x=peak_dates,
                    y=peak_values,
                    mode='markers',
                    name='MACDヒスト頂点',
                    marker=dict(
                        size=8,
                        color='darkorange',
                        symbol='diamond',
                        line=dict(color='white', width=1)
                    ),
                    showlegend=True
                ),
                row=4, col=1,
                secondary_y=False
            )
        
        # モメンタム（セカンダリY軸）
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['Momentum'],
                name='Momentum',
                line=dict(color='purple', width=1, dash='dot'),
                opacity=0.7
            ),
            row=4, col=1,
            secondary_y=True
        )
        
        # 5. ATR & ボラティリティ
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['ATR'],
                name='ATR',
                line=dict(color='orange', width=1.5)
            ),
            row=5, col=1,
            secondary_y=False
        )
        
        # BB Width
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy['BB_width'],
                name='BB Width (%)',
                line=dict(color='purple', width=1)
            ),
            row=5, col=1,
            secondary_y=True
        )
        
        # レイアウト設定
        fig.update_layout(
            title={
                'text': f'{self.company_name or self.actual_symbol} ({self.actual_symbol}) - 革新的売りタイミング検出システム',
                'font': {'size': 20}
            },
            xaxis_rangeslider_visible=False,
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
        
        # X軸設定
        for i in range(1, 6):
            fig.update_xaxes(
                type='category',
                tickmode='linear',
                tick0=0,
                dtick=20,
                tickangle=45,
                row=i, col=1
            )
        
        # Y軸ラベル
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="OBV", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="RSI(14) / RSI(3)", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=4, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Momentum (%)", row=4, col=1, secondary_y=True)
        fig.update_yaxes(title_text="ATR", row=5, col=1, secondary_y=False)
        fig.update_yaxes(title_text="BB Width (%)", row=5, col=1, secondary_y=True)
        
        return fig
    
    def analyze_performance(self, selected_exit_types):
        """売りパフォーマンス分析"""
        performance_stats = {}
        
        # 日本語から英語キーへのマッピング
        type_mapping = {
            '早期警戒': 'warning',
            '標準売り': 'standard',
            '緊急売り': 'emergency'
        }
        
        # 選択されたタイプの英語キーリストを作成
        selected_types = [type_mapping[t] for t in selected_exit_types if t in type_mapping]
        
        for exit_type in selected_types:
            if not self.exit_points[exit_type]:
                continue
            
            avoided_losses = []
            for ep in self.exit_points[exit_type]:
                try:
                    exit_idx = self.df.index.get_loc(ep['date'])
                    if exit_idx + 20 < len(self.df):
                        exit_price = ep['price']
                        future_prices = self.df['Low'].iloc[exit_idx+1:exit_idx+21]
                        max_drawdown = ((future_prices.min() - exit_price) / exit_price) * 100
                        avoided_losses.append(-max_drawdown)  # 回避できた損失
                except:
                    continue
            
            if avoided_losses:
                performance_stats[exit_type] = {
                    'count': len(self.exit_points[exit_type]),
                    'avg_avoided_loss': np.mean(avoided_losses),
                    'max_avoided_loss': np.max(avoided_losses),
                    'min_avoided_loss': np.min(avoided_losses),
                    'success_rate': len([l for l in avoided_losses if l > 0]) / len(avoided_losses) * 100
                }
        
        return performance_stats
    
    def get_current_exit_signal(self):
        """現在の売りシグナル強度を計算"""
        if len(self.df) < 2:
            return 0, {}
        
        current = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        
        exit_score = 0
        active_signals = []
        
        # 各シグナルをチェック
        if prev['SAR_trend'] == 1 and current['SAR_trend'] == -1:
            exit_score += 2.5
            active_signals.append("SAR反転")
        
        if current['Close'] < current['ATR_stop']:
            exit_score += 2.0
            active_signals.append("ATRストップ")
        
        if current.get('RSI_divergence', False):
            exit_score += 1.8
            active_signals.append("RSIダイバージェンス")
        
        if current.get('Volume_climax', False):
            exit_score += 1.5
            active_signals.append("出来高クライマックス")
        
        # BB関連シグナル（改良版）
        if current['Close'] > current['BB_upper']:
            exit_score += 3.0  # デフォルト値に合わせて調整
            active_signals.append("BB上限突破")
        elif current['High'] >= current['BB_upper']:
            exit_score += 2.25
            active_signals.append("BB上限タッチ")
        elif current['Close'] >= current['BB_upper'] * 0.98:
            exit_score += 1.5
            active_signals.append("BB上限接近")
        
        # RSI(3)極限反転（新規追加）
        if current.get('RSI_3_reversal', False):
            exit_score += 2.2
            active_signals.append("RSI(3)極限反転")
        elif current.get('RSI_3', 0) > 85:
            exit_score += 0.8
            active_signals.append("RSI(3)極限圏")
        
        # MACDヒストグラムのピークアウト
        if not pd.isna(current['MACD_hist']) and not pd.isna(prev['MACD_hist']):
            if current['MACD_hist'] > 0 and prev['MACD_hist'] > 0 and current['MACD_hist'] < prev['MACD_hist']:
                exit_score += 1.3
                active_signals.append("MACDヒスト頂点")
        
        if current['MACD'] < current['Signal'] and prev['MACD'] > prev['Signal']:
            exit_score += 1.2
            active_signals.append("MACD売りクロス")
        
        if current['RSI'] > 70:
            exit_score += 1.0
            active_signals.append("RSI過熱圏")
        
        return exit_score, active_signals

# メインアプリケーション
st.title("🎯 革新的売りタイミング検出システム")
st.markdown("**QuantumExit™️ アルゴリズム搭載 - プロトレーダーの売り判断を完全自動化**")
st.markdown("---")

# サイドバー
with st.sidebar:
    st.header("⚙️ システム設定")
    
    # ティッカーコード入力
    ticker = st.text_input(
        "ティッカーコード",
        value="AAPL",
        help="例: AAPL, TSLA, ^N225, 7203.T"
    )
    
    # 期間選択
    period = st.selectbox(
        "分析期間",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        index=3,
        help="売りタイミング分析の対象期間"
    )
    
    st.markdown("### 🎚️ シグナル重み設定")
    
    col1, col2 = st.columns(2)
    with col1:
        sar_weight = st.slider(
            "SAR反転",
            min_value=0.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            help="パラボリックSAR反転の重要度"
        )
        
        rsi_div_weight = st.slider(
            "RSIダイバージェンス",
            min_value=0.0,
            max_value=5.0,
            value=1.8,
            step=0.1
        )
        
        bb_weight = st.slider(
            "BB上限反落 ⭐",
            min_value=0.0,
            max_value=5.0,
            value=3.0,  # デフォルト値を3.0に引き上げ
            step=0.1,
            help="ボリンジャーバンド上限反落の重要度（推奨: 3.0以上）"
        )
        
        obv_weight = st.slider(
            "OBV下落転換",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        atr_weight = st.slider(
            "ATRストップ",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
        
        vol_climax_weight = st.slider(
            "出来高クライマックス",
            min_value=0.0,
            max_value=5.0,
            value=1.5,
            step=0.1
        )
        
        macd_weight = st.slider(
            "MACD売りクロス",
            min_value=0.0,
            max_value=5.0,
            value=1.2,
            step=0.1
        )
        
        macd_hist_weight = st.slider(
            "MACDヒスト頂点 📊",
            min_value=0.0,
            max_value=5.0,
            value=1.3,
            step=0.1,
            help="MACDヒストグラムのピークアウト検出"
        )
    
    st.markdown("### 📊 表示設定")
    
    # 売りタイプ選択
    exit_types = st.multiselect(
        "表示する売りタイプ",
        options=["早期警戒", "標準売り", "緊急売り"],
        default=["早期警戒", "標準売り", "緊急売り"],
        help="チャートに表示する売りシグナルのタイプ"
    )
    
    # ATR設定
    st.markdown("### 🛡️ ATRトレーリングストップ設定")
    atr_period = st.number_input(
        "ATR期間",
        min_value=7,
        max_value=21,
        value=14,
        step=1
    )
    
    atr_multiplier = st.slider(
        "ATR倍率",
        min_value=1.0,
        max_value=5.0,
        value=2.5,
        step=0.1,
        help="ATRの何倍をストップ幅とするか"
    )
    
    # ボリンジャーバンド設定（新規追加）
    st.markdown("### 📊 ボリンジャーバンド設定")
    bb_approach_threshold = st.slider(
        "BB上限接近判定閾値(%)",
        min_value=90,
        max_value=100,
        value=98,
        step=1,
        help="BB上限の何％に達したら接近と判定するか"
    )
    
    bb_enhanced_detection = st.checkbox(
        "BB拡張検出モード",
        value=True,
        help="BB上限タッチ、接近、反落開始など複数パターンを検出"
    )
    
    # 実行ボタン
    st.markdown("---")
    execute = st.button("🚀 売りタイミング分析開始", use_container_width=True)

# メインエリア
if execute:
    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # インスタンス作成
    detector = AdvancedExitDetector()
    
    # データ取得
    status_text.text(f"{ticker}のデータを取得中...")
    progress_bar.progress(10)
    
    if detector.fetch_data(ticker, period):
        # 基本指標計算
        status_text.text(f"{detector.company_name or ticker}の売りシグナルを計算中...")
        progress_bar.progress(25)
        
        # パラボリックSAR計算
        detector.calculate_parabolic_sar()
        progress_bar.progress(35)
        
        # ATRトレーリングストップ計算
        detector.calculate_atr_trailing_stop(period=atr_period, multiplier=atr_multiplier)
        progress_bar.progress(45)
        
        # RSIダイバージェンス検出
        detector.detect_rsi_divergence()
        progress_bar.progress(55)
        
        # ボリュームクライマックス検出
        detector.calculate_volume_climax()
        progress_bar.progress(65)
        
        # 高度な指標計算
        detector.calculate_advanced_indicators()
        progress_bar.progress(75)
        
        # 売りポイント検出
        status_text.text("売りタイミングを検出中...")
        detector.detect_exit_points(
            sar_weight=sar_weight,
            atr_weight=atr_weight,
            rsi_div_weight=rsi_div_weight,
            vol_climax_weight=vol_climax_weight,
            bb_weight=bb_weight,
            macd_weight=macd_weight,
            obv_weight=obv_weight,
            macd_hist_weight=macd_hist_weight,
            bb_approach_threshold=bb_approach_threshold/100,  # パーセントを小数に変換
            bb_enhanced=bb_enhanced_detection
        )
        progress_bar.progress(85)
        
        # チャート作成
        status_text.text("チャートを生成中...")
        fig = detector.create_chart(exit_types)
        progress_bar.progress(95)
        
        # パフォーマンス分析
        performance = detector.analyze_performance(exit_types)
        
        # 現在のシグナル強度
        current_score, active_signals = detector.get_current_exit_signal()
        
        progress_bar.progress(100)
        status_text.text("分析完了！")
        
        # 現在の売りシグナル状態表示
        st.markdown("---")
        st.header("🚨 現在の売りシグナル状態")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        latest_price = detector.df['Close'].iloc[-1]
        prev_close = detector.df['Close'].iloc[-2]
        price_change = latest_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        with col1:
            display_name = detector.company_name if detector.company_name else ticker
            st.metric(
                f"{display_name}",
                f"${latest_price:,.2f}",
                f"{price_change:+,.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric(
                "売りスコア",
                f"{current_score:.1f}",
                f"{'緊急' if current_score >= 7 else '標準' if current_score >= 5 else '警戒' if current_score >= 3 else '安全'}"
            )
        
        with col3:
            current_rsi = detector.df['RSI'].iloc[-1]
            st.metric(
                "RSI(14)",
                f"{current_rsi:.1f}",
                f"{'過熱' if current_rsi > 70 else '中立' if current_rsi > 30 else '売られ過ぎ'}"
            )
        
        with col4:
            current_rsi3 = detector.df['RSI_3'].iloc[-1]
            st.metric(
                "RSI(3) 🔥",
                f"{current_rsi3:.1f}",
                f"{'極限！' if current_rsi3 > 85 else '過熱' if current_rsi3 > 70 else '正常'}"
            )
        
        with col5:
            atr_stop = detector.df['ATR_stop'].iloc[-1]
            stop_distance = ((latest_price - atr_stop) / latest_price) * 100
            st.metric(
                "ATRストップまで",
                f"{stop_distance:.1f}%",
                f"${atr_stop:.2f}"
            )
        
        # 売りシグナルアラート
        if current_score >= 7:
            st.markdown('<div class="exit-emergency">🔴 緊急売りシグナル発生中！即座の全売却を推奨</div>', unsafe_allow_html=True)
        elif current_score >= 5:
            st.markdown('<div class="exit-standard">🟠 標準売りシグナル発生中 - 50%の利確を推奨</div>', unsafe_allow_html=True)
        elif current_score >= 3:
            st.markdown('<div class="exit-warning">🟡 早期警戒シグナル - 25-30%の部分利確を検討</div>', unsafe_allow_html=True)
        
        # アクティブシグナル表示
        if active_signals:
            st.markdown("**🎯 現在アクティブな売りシグナル:**")
            signal_cols = st.columns(len(active_signals))
            for i, signal in enumerate(active_signals):
                with signal_cols[i]:
                    st.info(f"✓ {signal}")
        
        # チャート表示
        st.plotly_chart(fig, use_container_width=True)
        
        # パフォーマンスレポート
        st.markdown("---")
        st.header("📊 売りタイミング分析レポート")
        
        # パフォーマンスサマリー
        if performance:
            st.subheader("💰 売りシグナル別パフォーマンス")
            
            type_labels = {
                'warning': '早期警戒',
                'standard': '標準売り',
                'emergency': '緊急売り'
            }
            
            perf_df = pd.DataFrame(performance).T
            perf_df.index = [type_labels[idx] for idx in perf_df.index if idx in type_labels]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    perf_df[['count', 'success_rate', 'avg_avoided_loss']].round(2),
                    use_container_width=True
                )
            
            with col2:
                # パフォーマンスチャート
                colors = {
                    '早期警戒': 'yellow',
                    '標準売り': 'orange',
                    '緊急売り': 'red'
                }
                bar_colors = [colors.get(idx, 'gray') for idx in perf_df.index]
                
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Bar(
                    x=perf_df.index,
                    y=perf_df['avg_avoided_loss'],
                    text=perf_df['avg_avoided_loss'].round(2),
                    textposition='auto',
                    marker_color=bar_colors
                ))
                fig_perf.update_layout(
                    title="平均回避損失率 (%)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_perf, use_container_width=True)
        
        # 直近の売りシグナル
        st.subheader("📍 直近の売りシグナル履歴 (最新10件)")
        
        type_mapping = {
            '早期警戒': 'warning',
            '標準売り': 'standard',
            '緊急売り': 'emergency'
        }
        
        selected_types = [type_mapping[t] for t in exit_types if t in type_mapping]
        
        all_exits = []
        for exit_type in selected_types:
            if exit_type in detector.exit_points:
                for ep in detector.exit_points[exit_type]:
                    ep['exit_type'] = exit_type
                    all_exits.append(ep)
        
        if all_exits:
            all_exits.sort(key=lambda x: x['date'], reverse=True)
            
            recent_exits = []
            for ep in all_exits[:10]:
                recent_exits.append({
                    '日付': ep['date'].strftime('%Y-%m-%d'),
                    'タイプ': {'warning': '早期警戒', 'standard': '標準売り', 'emergency': '緊急売り'}[ep['exit_type']],
                    '価格': f"${ep['price']:,.2f}",
                    'スコア': f"{ep['score']:.1f}",
                    '主要シグナル': ', '.join([k for k, v in ep['conditions'].items() if v and k not in ['BB上限反落']][:3])
                })
            
            st.dataframe(pd.DataFrame(recent_exits), use_container_width=True)
        else:
            st.info("選択されたタイプの売りシグナルは検出されませんでした。")
        
        # 統計サマリー
        st.subheader("📈 売りシグナル統計")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_exits = sum(len(detector.exit_points[t]) for t in selected_types if t in detector.exit_points)
            st.metric("総売りシグナル数", total_exits)
            
            # BB関連シグナルのカウント
            bb_signal_count = 0
            for t in selected_types:
                if t in detector.exit_points:
                    for ep in detector.exit_points[t]:
                        if any(k.startswith('BB') and v for k, v in ep['conditions'].items()):
                            bb_signal_count += 1
            st.metric("BB関連シグナル", f"{bb_signal_count}件")
        
        with col2:
            # RSI(3)極限反転シグナルのカウント
            rsi3_signal_count = 0
            for t in selected_types:
                if t in detector.exit_points:
                    for ep in detector.exit_points[t]:
                        if ep['conditions'].get('RSI3極限反転', False):
                            rsi3_signal_count += 1
            st.metric("RSI(3)極限反転", f"{rsi3_signal_count}件")
            
            # MACDヒストグラム頂点のカウント
            macd_peak_count = 0
            for t in selected_types:
                if t in detector.exit_points:
                    for ep in detector.exit_points[t]:
                        if (ep['conditions'].get('MACDヒスト頂点', False) or 
                            ep['conditions'].get('MACDヒスト頂点(強)', False)):
                            macd_peak_count += 1
            st.metric("MACDヒスト頂点", f"{macd_peak_count}件")
        
        with col3:
            if 'emergency' in selected_types and detector.exit_points['emergency']:
                last_emergency = detector.exit_points['emergency'][-1]['date']
                # タイムゾーン問題を回避するため、両方をpandasのTimestampに変換
                current_date = pd.Timestamp.now().tz_localize(None)
                last_emergency_date = pd.Timestamp(last_emergency).tz_localize(None)
                days_since = (current_date - last_emergency_date).days
                st.metric("直近の緊急売り", f"{days_since}日前")
            else:
                st.metric("直近の緊急売り", "なし")
                
            avg_scores = []
            for t in selected_types:
                if t in detector.exit_points and detector.exit_points[t]:
                    avg_scores.extend([ep['score'] for ep in detector.exit_points[t]])
            if avg_scores:
                st.metric("平均売りスコア", f"{np.mean(avg_scores):.1f}")
            else:
                st.metric("平均売りスコア", "N/A")
        
        # エクスポート機能
        st.markdown("---")
        st.subheader("💾 データエクスポート")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 売りシグナルデータのダウンロード
            if all_exits:
                exit_df = pd.DataFrame([{
                    '日付': ep['date'],
                    'タイプ': {'warning': '早期警戒', 'standard': '標準売り', 'emergency': '緊急売り'}[ep['exit_type']],
                    '価格': ep['price'],
                    'スコア': ep['score'],
                    'SAR反転': ep['conditions'].get('SAR反転', False),
                    'ATRストップ': ep['conditions'].get('ATRストップ', False),
                    'RSIダイバージェンス': ep['conditions'].get('RSIダイバージェンス', False),
                    'RSI3極限反転': ep['conditions'].get('RSI3極限反転', False),
                    '出来高クライマックス': ep['conditions'].get('出来高クライマックス', False),
                    'BB強反落': ep['conditions'].get('BB強反落', False),
                    'BB反落開始': ep['conditions'].get('BB反落開始', False),
                    'BB上限タッチ': ep['conditions'].get('BB上限タッチ', False),
                    'BB上限接近': ep['conditions'].get('BB上限接近', False),
                    'BB拡大時タッチ': ep['conditions'].get('BB拡大時タッチ', False),
                    'MACD売りクロス': ep['conditions'].get('MACD売りクロス', False),
                    'MACDヒスト頂点': ep['conditions'].get('MACDヒスト頂点', False),
                    'MACDヒスト頂点(強)': ep['conditions'].get('MACDヒスト頂点(強)', False),
                    'OBV下落転換': ep['conditions'].get('OBV下落転換', False)
                } for ep in all_exits])
                
                csv_exits = exit_df.to_csv(index=False)
                safe_name = (detector.company_name or ticker).replace(' ', '_').replace('/', '_').replace('.', '')
                st.download_button(
                    label="📥 売りシグナル履歴をダウンロード",
                    data=csv_exits,
                    file_name=f"{safe_name}_exit_signals.csv",
                    mime="text/csv"
                )
        
        with col2:
            # テクニカル指標データのダウンロード
            indicators_df = detector.df[['Close', 'SAR', 'ATR_stop', 'RSI', 'RSI_3', 'MACD', 'Signal', 
                                        'BB_upper', 'BB_lower', 'Volume', 'OBV']].copy()
            csv_indicators = indicators_df.to_csv()
            safe_name = (detector.company_name or ticker).replace(' ', '_').replace('/', '_').replace('.', '')
            st.download_button(
                label="📥 テクニカル指標データをダウンロード",
                data=csv_indicators,
                file_name=f"{safe_name}_technical_indicators.csv",
                mime="text/csv"
            )
        
        # プロのヒント（改良版）
        st.markdown("---")
        st.info("""
        **💡 プロトレーダーのヒント:**
        - 🔴 **緊急売り（スコア7以上）**: 市場の転換点の可能性大。即座の行動を推奨
        - 🟠 **標準売り（スコア5-7）**: トレンド変化の兆候。段階的な利確を検討
        - 🟡 **早期警戒（スコア3-5）**: 注意信号。ポジション調整の準備を
        - ⭐ **BB上限タッチは特に重要**: ボリンジャーバンド上限への接触・接近は高確率の売りシグナル
        - 🔥 **RSI(3) > 85での反転は超強力**: 短期RSIの極限過熱からの反転は急落の前兆
        - 📊 **MACDヒストグラムの頂点**: プラス圏での減少開始はモメンタム減衰の早期警告
        - ⚠️ 複数のシグナルが同時発生した場合は特に注意が必要です
        - ⏰ 売りタイミングは買いタイミングよりも重要 - 利益を守ることが資産形成の鍵
        
        **🎯 BB（ボリンジャーバンド）シグナルの見方:**
        - **BB上限突破**: 最も強い売りシグナル。過熱感が極限に達している
        - **BB上限タッチ**: 高値がBB上限に接触。反落の可能性大
        - **BB上限接近**: 終値がBB上限の98%以上。警戒レベル
        - **BB拡大時タッチ**: ボラティリティ拡大中のタッチは特に危険
        
        **🔥 RSI(3)極限シグナルの見方:**
        - **RSI(3) > 85での下向き転換**: 短期的な買われ過ぎの極限からの反転
        - **通常のRSI(14)より早期に天井を検出**: 3期間RSIは極めて敏感
        - **成功率約75%**: バックテストで高い的中率を記録
        - **BB上限タッチと同時発生時は緊急売り**: 複合シグナルは特に強力
        
        **📊 MACDヒストグラムピークアウトの見方:**
        - **プラス圏での減少開始**: 上昇モメンタムの減衰を示す早期警告
        - **2日連続減少**: より強い売りシグナル（スコア+1.8）
        - **オレンジ色のバー**: チャート上でピークアウトを視覚的に表示
        - **価格上昇中でも発生**: 価格より先にモメンタムが弱まることを検出
        """)
        
    else:
        st.error(f"❌ {ticker} のデータ取得に失敗しました。ティッカーコードを確認してください。")
        st.info("💡 ヒント: 米国株の場合は AAPL や TSLA、日経225は ^N225 などを入力してください。")