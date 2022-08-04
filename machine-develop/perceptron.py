from sklearn.datasets import load_breast_cancer
from pandas import DataFrame 
from matplotlib import pyplot
pyplot.rcParams["font.family"] = "IPAGothic"

breast_canser = load_breast_cancer() 
X = breast_canser.data[:,:10] #特徴量
y = breast_canser.target #目的変数 
"""
入力に対して適切な出力をするようんな重みwを適切に求める => perceptron
"""
columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元'] 

df = DataFrame(data=X,columns=columns) #dataとcolumns名（列）を格納する

