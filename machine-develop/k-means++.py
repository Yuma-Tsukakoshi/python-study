"""
1.データからランダムに1点、セントロイドとして選ぶ
2.各データについて、最も近いセントロイドとの距離 D(x) を計算する
3.D(x) の比率で重み付けされた分布から新しいセントロイドをサンプリングする
4.K 個のセントロイドが選ばれるまで 2 と 3 を繰り返す
5.K 個のセントロイドを用いて、通常の k-means を適用する
"""

import numpy 
import pandas

from matplotlib import pyplot
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=4, n_features=2, cluster_std=2.0, random_state=42)

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

pyplot.grid()
pyplot.legend()
pyplot.show()

from sklearn.cluster import KMeans

K = 4
kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42)

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

kmeans.cluster_centers_