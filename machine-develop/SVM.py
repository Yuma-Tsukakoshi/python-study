"""
サポートベクターマシン(SVM):線形分離不可能な問題を解ける
学習に必要なデータ数も比較的少なくても良いという利点あり
ゴール=>マージンを最大化する決定境界を学習により求める
ソフトマージン分類,スラック変数 
"""
"""
カーネル法:関数 ϕ を通して高次元空間に非線形に射影し、射影先の空間で線形分離する
=>カーネル法によるSVMの学習は計算コストが高いため、カーネルトリック技術で計算を効率化。 
=>カーネルトリックは、射影先の空間での内積値を カーネル関数 を使って定義する。 
つまり、射影先で ϕ(x) を計算することなく、サンプル間の距離がわかる。
"""
#<データの読み込み>
from pandas import DataFrame
from sklearn.datasets import load_breast_cancer

breast_canser =load_breast_cancer()
X = breast_canser.data[:,:10]
y = breast_canser.target

columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

df = DataFrame(data=X[:,:10],columns=columns)
df["目的変数"] = y

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
from sklearn.svm import SVC
#サポートベクターマシンの実装である SVC を import 
svc = SVC(kernel="rbf",C=25,probability=True,random_state=42)
#カーネル関数：RBFカーネル。
#C ソフトマージンの厳しさを表すパラメータ 
# probability=True 後の確率推定に用いる
svc.fit(X_train_std,y_train)

#<予測>
pred = svc.predict(X_test_std)
proba = svc.predict_proba(X_test_std)

#<評価>
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)

#<決定領域>
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

N = 100
sampled_X = np.vstack((X_train_std[:N],X_test_std[:N]))
sampled_y = np.hstack((y_train[:N],y_test[:N]))

plt.figure(figsize=(12,12))
plt.xlabel("面積",fontname='MS Gothic')
plt.ylabel("へこみ",fontname='MS Gothic')
plt.title("サポートベクターマシンの決定領域",fontname='MS Gothic')

plot_decision_regions(sampled_X,sampled_y,clf=svc,legend=2,X_highlight=X_test_std[:N])

plt.show()
