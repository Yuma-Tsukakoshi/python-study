"""
決定木の弱点を補うために、複数の決定木を組み合わせる機械学習手法
ランダムフォレストは、複数の決定木の アンサンブル (ensemble) 
アンサンブルは、複数の予測器を作成して多数決を取るアルゴリズム
ランダムフォレストは複数の予測器として、決定木を採用します。

ランダムフォレストのアルゴリズムは以下のステップにまとめられます。

1.サイズ N のブートストラップ標本（学習データを復元抽出する）を K 個作成する
2.各標本に対して、特徴量を非復元抽出し、決定木を K 個学習する
3.作成した決定木の予測をまとめて「多数決」によりクラスラベルを予測する
"""
from pandas import DataFrame
from  sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X = breast_cancer.data[:,:10]
y = breast_cancer.target

columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

df = DataFrame(data=X[:,:10],columns=columns)
df["目的変数"] = y

X = df[["面積","へこみ"]].values
y = df["目的変数"].values

from sklearn.model_selection import PredefinedSplit, train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#<学習>
"""
ランダムフォレストの実装=> RandomForestClassifier 
情報利得の計算にはデフォルトのジニ指数を (criterion='gini')
(max_depth=1):決定木の深さ 
(n_estimators=10):決定木の数

ランダムフォレストの学習のポイント => なるべく簡単な（深くない）決定木を多く作ること 
単純な決定木を多数組み合わせることで、良い性能を達成しやすくなることが多い。
"""
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier(criterion='gini', max_depth=1, n_estimators=10, random_state=42)
rf.fit(X_train_std,y_train)

pred = rf.predict(X_test_std)
proba = rf.predict_proba(X_test_std)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)

#<決定木のプロット>
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

N = 100
sampled_X = np.vstack((X_train_std[:N],X_test_std[:N]))
sampled_y = np.hstack((y_train[:N],y_test[:N]))

plt.figure(figsize=(12,12))
plt.xlabel("面積",fontname='MS Gothic')
plt.ylabel("へこみ",fontname='MS Gothic')
plt.title("ランダムフォレストの決定領域",fontname='MS Gothic')

plot_decision_regions(sampled_X,sampled_y,clf=rf,legend=2,X_highlight=X_test_std[:N])

import graphviz
from sklearn.tree import export_graphviz
from IPython.display import display, SVG

# 学習した決定木は estimators_ に格納されている
for i, tree in enumerate(rf.estimators_, start=1):
    print(f"Tree {i}")
    out = export_graphviz(tree, out_file=None, feature_names=['面積', 'へこみ'], class_names=['悪性', '良性'],
                                        label='all', filled=True, leaves_parallel=False, rotate=False, rounded=True, impurity=False)
    dot = graphviz.Source(out)
    display(SVG(dot.pipe('svg')))
    print()
    
#特徴量重要度のプロット    
feature_names = np.array(['面積', 'へこみ'])
# 特徴重要度は feature_importances_ に格納されている
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances )

plt.figure(figsize=(11, 7))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), feature_names[indices])
plt.show()    