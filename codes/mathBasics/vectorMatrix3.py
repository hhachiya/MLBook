# -*- coding: utf-8 -*-
import numpy as np

# 行列：3x2のnumpy arrayをを定義
X = np.array([ [1.0,2.0], [2.0,4.0], [3.0,6.0] ])

# 0行目の行ベクトル(1×2)の取り出し
Xrow0 = X[[0],:]

# 1列目の列ベクトル(3×1)の取り出し
Xcol1 = X[:,[1]]

# 標準出力
print(f"行列X）形:{X.shape}, 型:{X.dtype}\n{X}\n")
print(f"0行目）形:{Xrow0.shape}, 型:{Xrow0.dtype}\n{Xrow0}\n")
print(f"1列目）形:{Xcol1.shape}, 型:{Xcol1.dtype}\n{Xcol1}\n")