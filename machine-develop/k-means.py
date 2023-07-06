"""
1.ランダムに K 個のクラスタの中心（セントロイド）μ(1),⋯,μ(K) を決める
2.各サンプル（データ）を最も近いセントロイドに割り当てる
3.割り当てられたサンプルの中心となるようにセントロイドを更新する
4.サンプル点へのクラスタの割り当てが変化しなくなるか、または一定回数、手順2と3を繰り返す
"""

import numpy as np
import pandas

from  matplotlib  import pyplot
from sklearn.datasets import make_blobs

X ,y = make_blobs(n_samples=200, centers=4, n_features=2, cluster_std=2.0, random_state=42)

#正解のプロット
K = 4  # クラスタ数

pyplot.figure(figsize=(12, 12))

for i in range(K):
    idx = (y == i)
    pyplot.scatter(X[idx, 0],
                   X[idx, 1],
                   s=100,
                   label=f"cluster {i}")

pyplot.xlabel(f"$x_{0}$", fontsize=16)
pyplot.ylabel(f"$x_{1}$", fontsize=16)

# pyplot.grid()
# pyplot.legend()
# pyplot.show()

#k-means
from sklearn.cluster import KMeans
K = 4
kmeans = KMeans(n_clusters=K , init='random', n_init=1, random_state=42)

y_pred = kmeans.fit_predict(X)
#print(y_pred)

kmeans.inertia_

pyplot.figure(figsize=(12, 12))

# 予測のプロット
for i in range(K):
    idx = (y_pred == i)
    pyplot.scatter(X[idx, 0],
                   X[idx, 1],
                   s=50,
                   label=f"cluster {i}")

# セントロイドのプロット
pyplot.scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               s=200,
               marker='*',
               label='centroids')
    
pyplot.xlabel(f"$x_{0}$", fontsize=16)
pyplot.ylabel(f"$x_{1}$", fontsize=16)
pyplot.grid()
pyplot.legend()
# pyplot.show()

kmeans.cluster_centers_
# クラスタ数を3にする
K = 3
# クラスタ数以外のパラメータは同じ
kmeans = KMeans(n_clusters=3, init='random', random_state=42)
# 学習と予測
y_pred = kmeans.fit_predict(X)
kmeans.inertia_

pyplot.figure(figsize=(12, 12))

for i in range(K):
    idx = (y_pred == i)
    pyplot.scatter(X[idx, 0],
                   X[idx, 1],
                   s=50,
                   label=f"cluster {i}")

pyplot.scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               s=200,
               marker='*',
               label='centroids')
    
pyplot.xlabel(f"$x_{0}$", fontsize=16)
pyplot.ylabel(f"$x_{1}$", fontsize=16)
pyplot.grid()
pyplot.legend()
pyplot.show()