import pandas as pd
data = {'name': ['Ryo', 'Kaori', 'Hideyuki', 'Hayato', 'Miki', 'Saeko'],  # 名前
        'gender': ['M', 'F', 'M', 'M', 'F', 'F'],  # 性別
        'size': ['L', 'M', 'L', 'XL', 'M', 'S']  # 服のサイズ
        }

df = pd.DataFrame(data,columns=['name', 'gender', 'size'])

#順序尺度
size2int = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
df['size'] = df['size'].map(size2int)

int2size = {v: k for k, v in size2int.items()}
df['size'] = df['size'].map(int2size)


#名義尺度
df['gender'] = pd.Categorical(df['gender'], categories=['M', 'F'],ordered=False)
#categories に表れるカテゴリ（クラス、ラベル）を忘れずに定義 ordered :True =>順序尺度 (categories の順序が保持)
# False => 名義尺度
#データタイプを category にしたら、次に pandas に含まれる get_dummies() 関数を利用して、ダミー化 と呼ばれる操作を適用
# ダミー化は、カテゴリ毎に特徴量を新たに追加し、0/1 の二値の組み合わせで、もとの特徴量を表現する変換

print(pd.get_dummies(df, columns=['gender']))
"""
        name  size  gender_M  gender_F
0       Ryo    L         1         0
1     Kaori    M         0         1
2  Hideyuki    L         1         0
3    Hayato   XL         1         0
4      Miki    M         0         1
5     Saeko    S         0         1
"""
print(pd.get_dummies(df, columns=['gender'], drop_first=True))
"""
        name size  gender_F
0       Ryo    L         0
1     Kaori    M         1
2  Hideyuki    L         0
3    Hayato   XL         0
4      Miki    M         1
5     Saeko    S         1
"""
