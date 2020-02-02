# -*- coding: utf-8 -*-
import numpy as np

# 3種類（身長，体重，胸囲）のデータを格納する7行3列の行列
X = np.array([[170,60,80],[167,52,93],[174,57,85],[181,70,80],\
  [171,62,70],[171,66,95],[171,66,95],[168,54,85]])

# Xを転置（変数×データ数）して，分散共分散行列の計算
cov_bias = np.cov(X.T)
cov_nobias =np.cov(X.T,bias=1)

# 標準出力
print(f"データX：\n{X}\n")
print(f"分散共分散行列 バイアスあり：\n{cov_bias}\n")
print(f"分散共分散行列 バイアスなし：\n{cov_nobias}\n")