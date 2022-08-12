#標準化
#平均値: 1.850371707708594e-17=0  標準偏差: 1.0
import numpy
import pandas

data = {'name': ['Ryo', 'Kaori', 'Hideyuki', 'Hayato', 'Miki', 'Saeko'],  # 名前
        'gender': ['M', 'F', 'M', 'M', 'F', 'F'],  # 性別
        'height': [186, 168, 175, 210, 160, 163],  # 身長
        'weight': [72, 47, 62, 90, None, numpy.NaN],  # 体重
        'age': [30, 20, None , numpy.NaN, 23, 25],  # 年齢
        'size': ['L', 'M', 'L', 'XL', None, 'S']  # 服のサイズ
        }

df = pandas.DataFrame(data,columns=['name', 'gender', 'age', 'height', 'weight', 'size'])

X = df['height'].values.reshape(-1,1).astype(float)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X)

'平均値:', X_std.mean()
'標準偏差:', X_std.std()

#正規化
#最大値: 1.0  , 最小値: 0.0
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

'最大値:', X_norm.max()
'最小値:', X_norm.min()

#特徴量の選択
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
X = breast_cancer.data[:,:10]
y = breast_cancer.target
columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=None,n_estimators=100,random_state=42)
rf.fit(X,y)

import matplotlib.pyplot as plt
feature_names = numpy.array(columns)
feature_importances = rf.feature_importances_
indices = numpy.argsort(feature_importances )

plt.figure(figsize=(11, 7))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), feature_names[indices],fontname='MS Gothic')

mu = feature_importances.mean()
print(feature_names[feature_importances > mu])

