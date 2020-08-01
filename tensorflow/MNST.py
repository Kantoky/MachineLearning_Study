# -*- coding: utf-8 -*-

#モジュールのインポート
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#mnistのインポート
mnist = keras.datasets.mnist

#訓練用とテスト用に分ける
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#0.0 ~ 1.0 の間にスケール
train_images = train_images / 255.0
test_images = test_images /255.0

#モデルの記述
model = keras.Sequential([
    #2次元配列を1次元配列に変換
    keras.layers.Flatten(input_shape=(28,28)),
    #入力層。入力はピクセル値(28*28)。隠れ層1へ出力
    keras.layers.Dense(512,activation='relu',input_shape=(784,)) , 
    #隠れ層１から隠れ層２へ
    keras.layers.Dense(512,activation='relu'),
    #出力層。softmax関数(シグモイド関数の親戚)で総和を1に
    keras.layers.Dense(10, activation='softmax')
])

#モデルのコンパイル
#損失関数...予測と正解の誤差を表す関数。クロスエントロピーや平均二乗誤差など。
#オプティマイザ...損失関数を最小化する(最適化)の手法。ニュートン法や再急降下法など。
#メトリクス...モデルの評価指標。accuracy(正解率)やLog Loss など。
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#学習させる。（5週）
model.fit(train_images, train_labels, epochs=5)

#テストして、正解率を出す。
test_loss, test_acc = model.evaluate(test_images,  test_labels)

print('\nTest accuracy:', test_acc)
