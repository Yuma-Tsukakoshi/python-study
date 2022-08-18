import numpy 
from pandas import DataFrame
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ames = fetch_openml(name="house_prices", as_frame=True)
X = ames.data  # 特徴量
y = ames.target # 目的変数
feature_names = [
    'LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
] # 特徴量名

df = DataFrame(data=X, columns=feature_names)

df['SalePrice'] = y  # 目的変数

X = df[['1stFlrSF', 'YearBuilt']].values
y = df['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train0 = X_train[:, 0].reshape(-1, 1)  # 訓練データの 1stFlrSF
X_train1 = X_train[:, 1].reshape(-1, 1)  # 訓練データの YearBuilt
X_test0 = X_test[:, 0].reshape(-1, 1)  # テストデータの 1stFlrSF
X_test1 = X_test[:, 1].reshape(-1, 1)  # テストデータの YearBuilt

from sklearn.ensemble import RandomForestRegressor
rf0 = RandomForestRegressor(criterion='squared_error', random_state=42)

rf0.fit(X_train0,y_train)
rf0.score(X_train0, y_train)
rf0.score(X_test0, y_test)

from matplotlib import pyplot
y_pred = rf0.predict(X_test0)

idx = numpy.argsort(X_test0[:, 0])
#X_test0[:,0] 一つのリストの形にしている
fig = pyplot.figure(figsize=(12, 12))

# テストデータのプロット（散布図）
pyplot.scatter(X_test0, y_test, color='black', alpha=0.55)

# 予測値のプロット（直線）
# このとき、データを 1stFlrSF の昇順にプロットする！
pyplot.plot(X_test0[:, 0][idx], y_pred[idx], linewidth=2, color='blue')

pyplot.xlabel("1stFlrSF", fontsize=18)
pyplot.ylabel("SalePrice", fontsize=18)

rf1 = RandomForestRegressor(criterion='squared_error', random_state=42)
rf1.fit(X_train1, y_train)
rf1.score(X_train1, y_train)
rf1.score(X_test1, y_test)
y_pred = rf1.predict(X_test1)
idx = numpy.argsort(X_test1[:, 0])

fig = pyplot.figure(figsize=(12, 12))

# テストデータのプロット（散布図）
pyplot.scatter(X_test1, y_test, color='black', alpha=0.55)

# 予測値のプロット（直線）
# このとき、データを YearBuilt の昇順にプロットする！
pyplot.plot(X_test1[:, 0][idx], y_pred[idx], linewidth=2, color='blue')

pyplot.xlabel("YearBuilt", fontsize=18)
pyplot.ylabel("SalePrice", fontsize=18)

rf = RandomForestRegressor(criterion='squared_error', random_state=42)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
rf.score(X_test, y_test)