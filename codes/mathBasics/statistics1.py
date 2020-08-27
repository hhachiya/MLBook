# -*- coding: utf-8 -*-
import numpy as np

# 3種類（身長，体重，胸囲）のデータを格納する7行3列の行列
X = np.array([[170,60,80],[167,52,93],[174,57,85],[181,70,80],\
    [171,62,70],[171,66,95],[171,66,95],[168,54,85]])

# 行方向（axis=0）に対して，平均値を計算
means = np.mean(X,axis=0)

# 行方向（axis=0）に対して，中央値を計算
medians = np.median(X,axis=0)


# 標準出力
print(f"データX:\n{X}\n")

print(f"平均値）身長:{means[0]:.2f}，体重:{means[1]:.2f}，胸囲:{means[2]:.2f}\n")

print(f"中央値）身長:{medians[0]:.2f}，体重:{medians[1]:.2f}，胸囲:{medians[2]:.2f}\n")
