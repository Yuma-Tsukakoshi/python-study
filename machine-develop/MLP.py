"""
多層パーセプトロン：活性化関数=>tanh関数 、ReLU関数
誤差逆伝播法；微分を効率よく計算し、重みを学習する
出力層から入力層に向けて計算する計算を 逆伝播
一本の直線では表現できないような決定境界を学習できる
"""
from pandas import DataFrame
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X = breast_cancer.data[:,:10]
y = breast_cancer.target

columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

df = DataFrame(data=X[:, :10], columns=columns)
df['目的変数'] = y

X = df[["面積","へこみ"]].values
y = df["目的変数"].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#<学習>
from sklearn.neural_network import MLPClassifier
# 多層パーセプトロンで使われるライブラリ
"""
中間層のユニット_1 = 6、中間層のユニット_2 = 2とする（hidden_layer_sizes=(6, 2)） 
活性化関数 =  ReLU 関数（activation='relu'）
solver = 'sgd' 確率的勾配降下法による学習
学習率 = 0.01 正則化パラメータ =  0.0001 
"""
mlp = MLPClassifier(hidden_layer_sizes=(6,2),activation="relu",solver="sgd",learning_rate_init=0.01, alpha=0.0001, max_iter=1000, random_state=42)

mlp.fit(X_train_std, y_train)

#<予測>
pred = mlp.predict(X_test_std)
# 確率（パーセプトロンでは取得できない）
proba = mlp.predict_proba(X_test_std)

#<評価>
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)

#<決定領域>
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
#決定境界の可視化

N = 100
sampled_X = np.vstack((X_train_std[:N], X_test_std[:N]))
sampled_y = np.hstack((y_train[:N], y_test[:N]))

plt.figure(figsize=(12,12))
plt.xlabel("面積",fontname='MS Gothic')
plt.ylabel("へこみ",fontname='MS Gothic')
plt.title("多層パーセプトロンの決定領域",fontname='MS Gothic')

plot_decision_regions(sampled_X, sampled_y, clf=mlp, legend=2,X_highlight=X_test_std[:N])
"""
第１引数
スケーリングした訓練データの説明変数
第２引数
訓練データの目的変数
第３引数(clf)
構築したロジスティック回帰モデル
legend = 凡例の位置 右上から半時計周りに1,2,3,4
X_highlight =〇で囲って強調する
"""
plt.show()