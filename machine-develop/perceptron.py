from sklearn.datasets import load_breast_cancer
from pandas import DataFrame 
from matplotlib import pyplot
pyplot.rcParams["font.family"] = "IPAGothic"


breast_canser = load_breast_cancer() 
X = breast_canser.data[:,:10] #特徴量
y = breast_canser.target #目的変数 
"""
入力に対して適切な出力をするようんな重みwを適切に求める => perceptron
"""
columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元'] 

df = DataFrame(data=X[:,:10],columns=columns) #dataとcolumns名（列）を格納する
df['目的変数'] = y

X = df[["面積","へこみ"]].values
y = df["目的変数"].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
#random_state パラメータに分割方法のシードを与える=>再現性のある分割方法にする。

# ＜データの前処理＞ 標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #インスタンス化
sc.fit(X_train) #fit() 渡されたデータの最大値、最小値、平均、標準偏差、傾き...などの統計を取得して、内部メモリに保存する 必ず訓練データで行う

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#fit()で取得した統計情報を使って、渡されたデータを実際に書き換える。

train_mean = X_train_std.mean(axis=0) #列に沿って

#<学習>
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=1000,random_state=42)
#max_iter = 重みの最大更新回数 random_state = 実行の度に値が変わるのを防ぐ 
ppn.fit(X_train_std,y_train)

#<予測>
pred = ppn.predict(X_test_std) #テストデータの予測

#<評価>
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

#<決定領域>
import numpy as np
import matplotlib.pyplot as plt
pyplot.rcParams['font.family'] = 'IPAPGothic'
from mlxtend.plotting import plot_decision_regions

# すべてのデータをプロットするとデータが多すぎるので制限する
N = 100

# 訓練データとテストデータからN個ずつのサンプルを先頭から取ってくる
sampled_X = np.vstack((X_train_std[:N], X_test_std[:N]))
#vstack=行方向に連結  ,hstack=列方向に連結
sampled_y = np.hstack((y_train[:N], y_test[:N]))

plt.figure(figsize=(12,12))
plt.xlabel("面積")
plt.ylabel("へこみ")
plt.title("パーセプトロンの決定領域")
plot_decision_regions(sampled_X, sampled_y, clf=ppn, legend=2,  X_highlight=X_test_std[:N])

plt.show()
