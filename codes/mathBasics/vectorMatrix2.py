# -*- coding: utf-8 -*-
import numpy as np

# 列ベクトル：2行1列のnumpy.ndarrayを定義
w = np.array([ [5.0],[3.0] ])

# 行ベクトル：1行2列のnumpy.ndarrayを定義
x = np.array([ [1.0,5.0] ])

# 内積の計算：x（行ベクトル）とw（列ベクトル）の掛け算
xw = np.matmul(x,w)

# wのノルムの計算：wの転置（行ベクトル）とw（列ベクトル）の掛け算
ww = np.matmul(w.T,w)

# 標準出力
print(f"xとwの内積）形:{xw.shape},型:{xw.dtype}\n{xw}\n")
print(f"wのノルム）形:{ww.shape},型:{ww.dtype}\n{ww}\n")