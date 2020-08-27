# -*- coding: utf-8 -*-
import numpy as np

# 行列：2行2列のnumpy.ndarrayを定義
A = np.array([ [6.0,2.0],[2.0,5.0] ])
B = np.array([ [6.0,3.0],[2.0,1.0] ])

# 行列式の計算
detA = np.linalg.det(A)
detB = np.linalg.det(B)

# 標準出力
print(f"行列Aの行列式:{detA:.1f}")
print(f"行列Bの行列式:{detB:.1f}")