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

df.isnull()
"""
欠損値が NaN や None => True それ以外のとき False を返します。
    name  gender    age  height  weight   size
0  False   False  False   False   False  False
1  False   False  False   False   False  False
2  False   False   True   False   False  False
3  False   False   True   False   False  False
4  False   False  False   False    True   True
5  False   False  False   False    True  False
"""
df.isnull().sum()#各列の欠損値の数はsum()で集計

df.dropna(axis=0)#行の消去
df.dropna(axis=1)#列の消去

#欠損値の補間
from sklearn.impute import SimpleImputer
imp_num = SimpleImputer(missing_values=numpy.nan,strategy="mean")
#missing_values=>欠損値と見なす値、strategy =>補間方法
imput_data = imp_num.fit_transform(df.values[:,[2,4]])
#欠損値を補間これには fit_transform() メソッドを利用

size2int = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}

df['size'] = df['size'].map(size2int)
imp_cat = SimpleImputer(missing_values=numpy.nan,strategy="most_frequent")
imp_cat.fit_transform(df.values[:,[5]])

#欠損値のダミー化
int2size = {v: k for k, v in size2int.items()}
df['size'] = df['size'].map(int2size)

df['size'] = pandas.Categorical(df['size'], categories=['S', 'M', 'L', 'XL'],ordered=True)
pandas.get_dummies(df,columns=["size"],dummy_na=True)
#dummy_na=True の引数を与えることで欠損値自体をダミー化の対象
