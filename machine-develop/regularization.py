"""
正則化手法を利用する線形回帰アルゴリズムには、3 つの有名なアルゴリズムが存在します：

Lasso 0<λ1，λ2=0
Ridge regression λ1=0 ，0<λ2
Elastic Net 0<λ1，0<λ2
"""
import numpy 
from pandas import DataFrame
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ames = fetch_openml(name="house_prices", as_frame=True)
X = ames.data
y = ames.target 
feature_names = [
    'LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
] # 特徴量名

df = DataFrame(data=X, columns=feature_names)

df['SalePrice'] = y  # 目的変数

X = df[['1stFlrSF', 'YearBuilt']].values
y = df["SalePrice"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


X_train0 = X_train[:, 0].reshape(-1, 1)  # 訓練データの 1stFlrSF
X_train1 = X_train[:, 1].reshape(-1, 1)  # 訓練データの YearBuilt
X_test0 = X_test[:, 0].reshape(-1, 1)  # テストデータの 1stFlrSF
X_test1 = X_test[:, 1].reshape(-1, 1)  # テストデータの YearBuilt

from sklearn.preprocessing import MinMaxScaler,StandardScaler
standerd_scaler = StandardScaler()
X_train_scaled0 = standerd_scaler.fit_transform(X_train0)
X_test_scaled0 = standerd_scaler.transform(X_test0)

min_max_scaler = MinMaxScaler()
X_train_scaled1 = min_max_scaler.fit_transform(X_train1)
X_test_scaled1 = min_max_scaler.transform(X_test1)

X_train_scaled = numpy.zeros(X_train.shape)
X_train_scaled[:, 0] = X_train_scaled0.reshape(-1)
X_train_scaled[:, 1] = X_train_scaled1.reshape(-1)

X_test_scaled = numpy.zeros(X_test.shape)
X_test_scaled[:, 0] = standerd_scaler.transform(X_test0).reshape(-1)
X_test_scaled[:, 1] = min_max_scaler.transform(X_test1).reshape(-1)

#k分割交差検証を利用して、正則化パラメータ（λ1 と λ2）を求める
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Lassoの実装
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=kf)
lasso.fit(X_train_scaled,y_train)
lasso.score(X_train_scaled, y_train) #決定係数の計算
lasso.score(X_test_scaled, y_test)
lasso.alpha_ #λ1

#Ridge回帰の実装
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(cv=kf)
ridge.fit(X_train_scaled, y_train)
ridge.score(X_train_scaled, y_train)
ridge.score(X_test_scaled, y_test)
ridge.alpha_#λ2

#Elastic Netの実装
from sklearn.linear_model import ElasticNetCV
elasticnet = ElasticNetCV(cv=kf, l1_ratio=0.5)
elasticnet.fit(X_train_scaled, y_train)
elasticnet.score(X_train_scaled, y_train)
elasticnet.score(X_test_scaled, y_test)
elasticnet.alpha_
elasticnet.l1_ratio_