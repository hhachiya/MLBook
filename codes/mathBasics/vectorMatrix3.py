# -*- coding: utf-8 -*-
import numpy as np

# 行列：3行2列のnumpy.ndarrayをを定義
X = np.array([ [1.0,2.0],[2.0,4.0],[3.0,6.0] ])

# 1行目の行ベクトル(1行2列)の取り出し
Xrow1 = X[[0],:]

# 2列目の列ベクトル(3行1列)の取り出し
Xcol2 = X[:,[1]]

# 標準出力
print(f"行列X）形:{X.shape},型:{X.dtype}\n{X}\n")
print(f"1行目）形:{Xrow1.shape},型:{Xrow1.dtype}\n{Xrow1}\n")
print(f"2列目）形:{Xcol2.shape},型:{Xcol2.dtype}\n{Xcol2}\n")