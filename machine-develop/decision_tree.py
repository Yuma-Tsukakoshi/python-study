"""
出力に至る過程を木構造として視覚的に捉えることができる
メリット
1: 欠損値を欠損値のまま扱うことができる
2: 標準化のような特徴量のスケーリング処理が不要
3: 木構造を辿ることで予測の根拠を簡単に得ることができる
4: 学習した予測器から特徴量の重要度を知ることができる

デメリット
木の深さを深くし過ぎると過学習を起こしやすい

ゴール=>データのばらつきの減少を表す 情報利得 の最大化 
データをもっとも“綺麗に”分割するように学習。
"""
from pandas import DataFrame
from  sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X = breast_cancer.data[:,:10]
y = breast_cancer.target

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
"""
決定木の実装 => DecisionTreeClassifier 
max_depth = 木の深さの最大値
max_features=None 特徴量が2つしかないので、全ての特徴量を利用。
criterion = 情報利得の計算 ジニ指数を指定。
"""
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="gini",max_depth=2,max_features=None,random_state=42)
clf.fit(X_train_std,y_train)

pred = clf.predict(X_test_std)
proba = clf.predict_proba(X_test_std)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

N = 100
sampled_X = np.vstack((X_train_std[:N],X_test_std[:N]))
sampled_y = np.hstack((y_train[:N],y_test[:N]))

plt.figure(figsize=(12,12))
plt.xlabel("面積",fontname='MS Gothic')
plt.ylabel("へこみ",fontname='MS Gothic')
plt.title("決定木の決定領域",fontname='MS Gothic')

plot_decision_regions(sampled_X,sampled_y,clf=clf,legend=2,X_highlight=X_test_std[:N])

plt.show()

