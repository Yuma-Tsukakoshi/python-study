"""
k近傍法 = 訓練データを覚えておき、予測したいサンプルと“距離的”に近い k 個のサンプルで多数決をとり、最も多かったクラスラベルを予測ラベルとする
訓練データの評価が予測時まで行われない学習 = 怠惰学習
kNN のメリット:
訓練データの追加が即時反映される
が挙げられます。
デメリット:
全ての訓練データを予測時に使用するので、それら記憶するための記憶容量が必要となる
訓練データが増加するにつれて、予測に時間がかかるようになる

ミンコフスキー距離 (Minkowski distance) と呼ばれる距離がデフォルトで実装されています。

"""
from pandas import DataFrame
from sklearn import metrics
from sklearn.datasets import load_breast_cancer

breast_cancer =  load_breast_cancer()
X = breast_cancer.data[:,:10]
y = breast_cancer.target

columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

df = DataFrame(data=X[:,:10],columns=columns)
df["目的変数"] = y

X = df[["面積","へこみ"]].values
y = df['目的変数'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#<学習>
from sklearn.neighbors import KNeighborsClassifier
"""
kNN の実装 => KNeighborsClassifier 
KNeighborsClassifier のインスタンス 
n_neighbors : k の数を 16 
ミンコフスキー距離 : p=2 (距離指標)
"""
knn = KNeighborsClassifier(n_neighbors=16,p=2,metric='minkowski')

knn.fit(X_train_std,y_train)

#<予測>
pred = knn.predict(X_test_std)
proba = knn.predict_proba(X_test_std)

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
plt.title("k近傍法の決定領域",fontname='MS Gothic')

plot_decision_regions(sampled_X, sampled_y, clf=knn, legend=2,  X_highlight=X_test_std[:N])

plt.show()