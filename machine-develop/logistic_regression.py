"""
活性化関数：シグモイド関数(重みの学習が複雑になる)
尤度を最大化するｗを求める=>ロジスティクス回帰の目的
＜尤度が1に近いほどよいモデルであると解釈できる＞
対数尤度を最大化する＝コスト関数を最小化する E(w)[コスト関数] = -l(w)[-対数尤度]
"""
"""
過学習：overfitting
学習不足：underfitting
"""

from random import sample
from pandas import DataFrame
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X = breast_cancer.data[:,:10]
y = breast_cancer.target
columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

df = DataFrame(data=X[:,:10],columns=columns)
df["目的変数"] = y

X = df[['面積', 'へこみ']].values
y = df["目的変数"].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#<学習>
from sklearn.linear_model import SGDClassifier
lr = SGDClassifier(loss='log_loss',alpha=0.0001, learning_rate='constant', eta0=0.1, shuffle=True, max_iter=1000, random_state=42)
"""
loss='log' =>ロジスティック回帰としてモデルを作成
alpha  λ に相当する正則化パラメータlearning_rate='constant' と eta0=0.1 を与え、
η を 0.1 にセット
shuffle パラメータは確率的勾配降下法においては特に重要な役割を果たすパラメータで、True を与えることで、学習の各ステップ毎に学習データをシャッフルします。これにより、ランダム性のある勾配学習を行うことができます（逆にシャッフルしないと、毎回同じ順番で勾配を計算します）
"""
lr.fit(X_train_std,y_train)

#<予測>
pred = lr.predict(X_test_std)
proba = lr.predict_proba(X_test_std) #各クラスに所属する確率

#<評価>
from sklearn.metrics import accuracy_score

#<決定領域>
import numpy
from matplotlib import pyplot
from mlxtend.plotting import plot_decision_regions

N = 100
sampled_X = numpy.vstack((X_train_std[:N],X_test_std[:N])) 
sampled_y = numpy.hstack((y_train[:N],y_test[:N]))

pyplot.figure(figsize=(12,12))
pyplot.xlabel("面積",fontname='MS Gothic')
pyplot.ylabel("へこみ",fontname='MS Gothic')
pyplot.title("ロジスティック回帰の決定領域",fontname='MS Gothic')

plot_decision_regions(sampled_X, sampled_y, clf=lr, legend=2,  X_highlight=X_test_std[:N])

pyplot.show()