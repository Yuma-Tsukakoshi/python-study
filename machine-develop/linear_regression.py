"""
残差平方和 = RSS
総平方和 = TSS
決定係数:  R^2 = 1-RRS/TSS

平均二乗誤差 = MSE
R^2 = 1-MSE/Var(y)
"""

#データの読み込み
from turtle import color
import numpy 
from pandas import DataFrame
from pyparsing import alphas
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ames = fetch_openml(name="house_prices",as_frame=True)
X = ames.data
y = ames.target
feature_names = [
    'LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
] # 特徴量名
df = DataFrame(data=X,columns=feature_names)
df["SalePrice"] = y

X = df[['1stFlrSF', 'YearBuilt']].values
y = df['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#単回帰のために2つの特徴量を別々の変数として用意
X_train0 = X_train[:, 0].reshape(-1, 1)  # 訓練データの 1stFlrSF
X_train1 = X_train[:, 1].reshape(-1, 1)  # 訓練データの YearBuilt
X_test0 = X_test[:, 0].reshape(-1, 1)  # テストデータの 1stFlrSF
X_test1 = X_test[:, 1].reshape(-1, 1)  # テストデータの YearBuilt
#各行で1サンプルの特徴量を入力データとしたいためreshape ※-1:もとの形状からの推測

#前処理 1stFlrSF=>標準化 1階の面積
from sklearn.preprocessing import MinMaxScaler,StandardScaler
standerd_scaler = StandardScaler()
X_train_scaled0 = standerd_scaler.fit_transform(X_train0)
X_test_scaled0 = standerd_scaler.transform(X_test0)

#前処理 YearBuilt=>正規化 最初に建設された年
min_max_scaler = MinMaxScaler()
X_train_scaled1 = min_max_scaler.fit_transform(X_train1)
X_test_scaled1 = min_max_scaler.transform(X_test1)

# X_train と同じ形状の配列を作る
X_train_scaled = numpy.zeros(X_train.shape)
# 1列目にスケール済みの 1stFlrSF を代入する
X_train_scaled[:, 0] = X_train_scaled0.reshape(-1)
# 2列目にスケール済みの YearBuilt を代入する
X_train_scaled[:, 1] = X_train_scaled1.reshape(-1)

# X_test と同じ形状の配列を作る
X_test_scaled = numpy.zeros(X_test.shape)
# 1列目にスケール済みの 1stFlrSF を代入する
X_test_scaled[:, 0] = X_test_scaled0.reshape(-1)
# 2列目にスケール済みの YearBuilt を代入する
X_test_scaled[:, 1] = X_test_scaled1.reshape(-1)

#  ↑  データの準備が完了  ↑  #

#単回帰
from sklearn.linear_model import SGDRegressor
reg0 = SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3, penalty='none', random_state=42)
"""
loss='squared_error' => コスト関数として二乗誤差を利用max_iter=1000 => 重みの最大更新回数を1000回
tol=1e-3 => コスト関数の減少が 0.001 以下になったら学習を止める
penalty='none' => 正則化をどうするかというパラメータ
"""
reg0.fit(X_train_scaled0,y_train)
reg0.score(X_train_scaled0,y_train) #決定係数の計算
reg0.score(X_test_scaled0,y_test)

import matplotlib.pyplot as plt
y_pred = reg0.predict(X_test_scaled0)

fig = plt.figure(figsize = (12,12))
plt.scatter(X_test0,y_test,color="black",alpha=0.55)
plt.plot(X_test0,y_pred,linewidth=3,color="blue")

plt.xlabel("1stFlrSF", fontsize=18,fontname='MS Gothic')
plt.ylabel("SalePrice", fontsize=18,fontname='MS Gothic')

reg1 = SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3, penalty='none', random_state=42)
reg1.fit(X_train_scaled1,y_train)
reg1.score(X_train_scaled1,y_train)
reg1.score(X_test_scaled1,y_test)

y_pred = reg1.predict(X_test_scaled1)

fig = plt.figure(figsize=(12,12))
plt.scatter(X_test1,y_test,color="black",alpha=0.55)
plt.plot(X_test1,y_pred,linewidth=3,color="blue")

plt.xlabel("YearBuilt", fontsize=18,fontname='MS Gothic')
plt.ylabel("SalePrice", fontsize=18,fontname='MS Gothic')

#重回帰
reg = SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3, penalty='none', random_state=42)
reg.fit(X_train_scaled,y_train)

reg.score(X_train_scaled,y_train)
reg.score(X_test_scaled,y_test)
