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
from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
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
