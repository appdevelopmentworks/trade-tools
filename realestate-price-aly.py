import streamlit as st
import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
#import japanize_matplotlib
import matplotlib.pyplot as plt
from PIL import Image

sns.set_theme(font="IPAexGothic")

def collist_init():
    st.session_state['list'] = df.columns
    
def make_train_dataset(df, tgtcol):
    #特定の列がターゲットとして選択された場合、ターゲット以外を取得する
    colX = [item for item in df.columns if item != tgtcol]
    #説明変数列
    X = df[colX]
    #ターゲット列
    y = df[tgtcol]
    return X, y
    
#モデルの学習
def train_data(X, y):
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    #パラメータ
    best_params = {
        'alpha': 0.00024192338397369958,
        'colsample_bytree': 0.6383204203639432,
        'gamma': 0.06966503119365258,
        'lambda': 0.12378043615831195,
        'learning_rate': 0.027700803647548924,
        'max_depth': 9,
        'n_estimators': 550,
        'subsample': 0.7370998508597036
    }

    # 最適なパラメータでXGBoost Regressorのインスタンスを作成
    best_xgb = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=int(best_params.get('n_estimators', 100)),
        learning_rate=best_params.get('learning_rate', 0.1),
        max_depth=int(best_params.get('max_depth', 3)),
        subsample=best_params.get('subsample', 1.0),
        colsample_bytree=best_params.get('colsample_bytree', 1.0),
        gamma=best_params.get('gamma', 0),
        reg_alpha=best_params.get('alpha', 0),
        reg_lambda=best_params.get('lambda', 1),
        random_state=42,
        n_jobs=-1
    )
    
    # モデルの学習
    best_xgb.fit(X_train, y_train)

    # テストデータでの予測
    y_pred = best_xgb.predict(X_test)

    # XGBoostモデルの評価
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mdlsc = best_xgb.score(X_test, y_test)
    
    st.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    st.write(f"R2 Score (決定係数): {r2:.4f}")
    st.write(f"MAE (Mean Absolute Error): {mae:.4f}")
    st.write(f"model Score:{mdlsc:.4f}")
    
    # 結果の可視化
    # 1. 実測値 vs 予測値の散布図
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("実測値")
    plt.ylabel("予測値")
    plt.title("実測値 vs 予測値の散布図")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # 45度線
    plt.grid(True)
    st.pyplot(plt)

    # 2. 残差プロット (Residual Plot)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.xlabel("予測値")
    plt.ylabel("残差 (実測値 - 予測値)")
    plt.title("残差プロット (Residual Plot)")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    st.pyplot(plt)

    # 3. 残差の分布 (ヒストグラムまたはKDEプロット)
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("残差")
    plt.ylabel("頻度")
    plt.title("残差の分布")
    plt.grid(True)
    st.pyplot(plt) 
    
    # 4. 特徴量の重要度 (Feature Importance)
    feature_importance = pd.Series(best_xgb.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.xlabel("重要度スコア")
    plt.ylabel("特徴量")
    plt.title("特徴量の重要度")
    st.pyplot(plt)
        
    return best_xgb
    
##########################################################################
# セッションステートにリストが存在しない場合は初期化
if 'list' not in st.session_state:
    st.session_state['list'] = []

if 'model' not in st.session_state:
    st.session_state['model'] = None


###########################################################################
st.title("教師ありデータから指定列を予測")

st.text("不動産価格の予想などに活用できます！")
image = Image.open("./images/headerml.png")
st.image(image)
st.caption("データセット（教師あり）をアップロードしてください！")


st.subheader("Step1. 教師データをアップロード！")
upfile = st.file_uploader("学習データアップロードしてください", type="csv")
df = pd.DataFrame()
dfview = st.dataframe(df)
#
dftst = pd.DataFrame()


if upfile:
    df = pd.read_csv(upfile)
    collist_init()

#２列構成ターゲット列選択、学習ボタン
col1, col2 = st.columns(2)
with col1:
    selectTarget =st.selectbox("ターゲット列",st.session_state['list'],len(st.session_state['list'])-1)
with col2:
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 60%;
            background-color: #006400;  /* 背景色 */
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
    btnTrain = st.button("学習")

#アップロードファイルを表示
dfview.dataframe(df)

#予測用データのアップロード
st.subheader("Step2. 予測させたいデータをアップロード！")
upfilepred = st.file_uploader("予測させたいデータをアップロード", type="csv")
dfpredview = st.dataframe(dftst)

if btnTrain:
    X, y = make_train_dataset(df, selectTarget)
    model = train_data(X, y)
    st.session_state['model'] = model
if upfilepred:
    dfval = pd.read_csv(upfilepred)
    model = st.session_state['model']
    pred = model.predict(dfval)
    dfpred = pd.concat([dfval.reset_index(drop=True), pd.DataFrame(pred, columns=["Label"])], axis=1)
    dfpredview.dataframe(dfpred,height=200)
    st.write("データフレーム最右列に予測値を追加しました！")
    st.write("予測値", pred)
    










