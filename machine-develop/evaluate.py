from pandas import DataFrame
from sklearn.datasets import load_breast_cancer
from sklearn.tree import plot_tree

breast_cancer = load_breast_cancer()
X = breast_cancer.data[:,:10]
y = breast_cancer.target

columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

df = DataFrame(data=X[:,:10],columns=columns)
df["目的変数"] = y

X = df[["面積","へこみ"]].values
y = df["目的変数"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion="gini",max_depth=1,n_estimators=10,random_state=42)
rf.fit(X_train_std,y_train)

pred = rf.predict(X_test_std)

#混同行列の計算
from sklearn.metrics import confusion_matrix
confat = confusion_matrix(y_true=y_test ,y_pred=pred)
#y_true =>正解データ、y_pred =>予測

import numpy 
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

plot_confusion_matrix(conf_mat=confat,figsize=(10,10))# conf_mat に上で計算した混同行列を加える

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)

#適合率(precision) は、陽性クラスと予測されたサンプルの内、実際に陽性クラスだったサンプル PRE
from sklearn.metrics import precision_score
precision_score(y_test,pred)

#再現率 (recall)は、正解ラベルが陽性であるサンプルの内、予測サンプルが陽性 REC
from sklearn.metrics import recall_score
recall_score(y_test,pred)

#F1スコア (F1-score) は、適合率と再現率を組み合わせた尺度で、これらの調和平均を計算 F1
from sklearn.metrics import f1_score
f1_score(y_test, pred)




