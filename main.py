#plot test
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#one input(x) data has 8 features and including 600 ~ 3000 people info.
#一つの入力データ(x)は8つの特徴量と600~3000人の情報を含んでいます。
from sklearn.datasets import fetch_california_housing

#detaset load
california = fetch_california_housing()
data = california.data #(20640, 8)
y = california.target   #(20640,)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
#['住民の収入の中央値','蚊帳の築年数の中央値','部屋数の平均値','寝室数の平均値','人口','住居の占める割合','ブロックの緯度','ブロックの経度']
feat = california.feature_names

#make graph
ax_id = 421 #4×2 number=1
plt.figure(figsize=(8, 10))
for f in feat:
    plt.subplot(ax_id)
    plt.scatter(data[:, feat.index(f)], y)
    plt.xlabel(feat[feat.index(f)])
    plt.ylabel('House price')
    ax_id += 1 #next number...

plt.show()

#Standardization
#標準化
#data = (data - np.mean(data)) / np.std(data)


