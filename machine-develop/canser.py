from turtle import color
from sklearn.datasets import load_breast_cancer
from pandas import DataFrame 
from matplotlib import pyplot
pyplot.rcParams["font.family"] = "IPAGothic"
from pandas.plotting import scatter_matrix  #散布図
pyplot.rcParams["font.family"] = "IPAGothic"

breast_canser = load_breast_cancer() #dataの格納
X = breast_canser.data[:,:10] #特徴量
y = breast_canser.target #目的変数 
feature_names = breast_canser.feature_names #特徴量名の表示

"""
特徴量（説明変数）を用いて学習を繰り返すことによって、目的変数に近づける => 教師あり学習
"""
columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元'] #日本語の特徴量名

df = DataFrame(data=X,columns=columns) #dataとcolumns名（列）を格納する
df["目的変数"] = y

colors = ["red" if v==0 else "blue" for v in y]
axes = scatter_matrix(df[columns],figsize=(20,20),diagonal="kde",c=colors)

