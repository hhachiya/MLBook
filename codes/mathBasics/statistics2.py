# -*- coding: utf-8 -*-
import numpy as np

# 3種類（身長，体重，胸囲）のデータを格納する7行3列の行列
X = np.array([[170,60,80],[167,52,93],[174,57,85],[181,70,80],\
    [171,62,70],[171,66,95],[171,66,95],[168,54,85]])

# 行方向（axis=0）に対して，分散と標準偏差を計算
vars = np.var(X,axis=0)
stds = np.std(X,axis=0)

# Xを転置（変数×データ数）して，分散共分散行列の計算
cov_nobias = np.cov(X.T)
cov_bias = np.cov(X.T,bias=1)

# 標準出力
print(f"データX:\n{X}\n")

print(f"分散）身長:{vars[0]:.2f}，体重:{vars[1]:.2f}，胸囲:{vars[2]:.2f}\n")

print(f"標準偏差）身長:{stds[0]:.2f}，体重:{stds[1]:.2f}，胸囲:{stds[2]:.2f}\n")

print(f"分散共分散行列 バイアスなし:\n{cov_nobias}\n")

print(f"分散共分散行列 バイアスあり:\n{cov_bias}\n")
