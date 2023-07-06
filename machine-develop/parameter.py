from pandas import DataFrame
from  sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X = breast_cancer.data[:, :10]
y = breast_cancer.target

columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

df = DataFrame(data=X[:, :10], columns=columns)
df['目的変数'] = y

X = df[['面積', 'へこみ']].values
y = df['目的変数'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold #層状K分割

#グリッドサーチの利用
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [1, 2],
    'n_estimators': [10, 15, 20, 25, 30]
}

gs = GridSearchCV(
  estimator=RandomForestClassifier(criterion="gini",random_state=42),param_grid=param_grid,scoring="accuracy",
  cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=42),return_train_score=True)
gs.fit(X_train,y_train)

#探索した結果、もっとも性能の良かったパラメータは best_params_ 属性に格納
gs.best_params_
gs.best_score_
df_grid_result = DataFrame(gs.cv_results_)
df_grid_result[['param_max_depth', 'param_n_estimators', 'mean_train_score', 'mean_test_score']]

# もっとも良かった機械学習モデルを取り出す
clf = gs.best_estimator_
# scoreメソッドを利用して、正解率の計算する
clf.score(X_test, y_test)

#ランダムサーチ
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'max_depth': randint(1, 3),
    'n_estimators': randint(10, 31)
}

rs = RandomizedSearchCV(
    # ランダムフォレスト
    estimator=RandomForestClassifier(criterion='gini', random_state=42),
    # 上で定義したパラメータの分布
    param_distributions=param_dist,
    scoring='accuracy',
    # 交差検証に StratifiedKFold を利用する
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    # 探索回数は10回
    n_iter=10,
    return_train_score=True,
    random_state=42)

rs.fit(X_train,y_train)
rs.best_params_
rs.best_score_

df_random_result = DataFrame(rs.cv_results_)
df_random_result[['param_max_depth', 'param_n_estimators', 'mean_train_score', 'mean_test_score']]

clf2 = rs.best_estimator_
clf2.score(X_test, y_test)