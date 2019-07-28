# plot test
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# one input(x) data has 8 features and including 600 ~ 3000 people info.
# 一つの入力データ(x)は8つの特徴量と600~3000人の情報を含んでいます。
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# data load
# データの読み込み
california = fetch_california_housing()
data = california.data  # (20640, 8)
y = california.target  # (20640,)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# ['住民の収入の中央値','蚊帳の築年数の中央値','部屋数の平均値','寝室数の平均値','人口','住居の占める割合','ブロックの緯度','ブロックの経度']
feat = california.feature_names

# model construction(LinearRegression)
# モデル構築(線形回帰)
X_train, X_test, y_train, y_test = train_test_split(data, y, train_size=0.8, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# prediction and model evaluation
# 予測とモデル評価
y_pred = model.predict(X_test)

# yyplot
plt.scatter(y_test, y_pred)
plt.title('fetch_california_housing × LinearRegression')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

# evaluation index
# 評価指数
print('Algorithm...LinearRegression')
print('MAE: ' + str(mean_absolute_error(y_test, y_pred)))
print('MSE: ' + str(mean_squared_error(y_test, y_pred)))
print('R^2: ' + str(r2_score(y_test, y_pred)))
# MAE: 0.5351261336554731
# MSE: 0.5289841670367224
# R^2: 0.594323265246619

