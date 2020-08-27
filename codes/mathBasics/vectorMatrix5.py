# -*- coding: utf-8 -*-
import numpy as np

# 行列：2行2列のnumpy.ndarrayを定義
A = np.array([ [6.0,2.0],[2.0,5.0] ])
B = np.array([ [6.0,3.0],[2.0,1.0] ])

# 行列のランクの計算
rankA = np.linalg.matrix_rank(A)
rankB = np.linalg.matrix_rank(B)

# Aの逆行列
if rankA == len(A):
    invA = np.linalg.inv(A)
    print(f"行列A）ランク:{rankA}\n逆行列:\n{invA}\n")
else:
    print(f"行列A）ランク:{rankA},特異行列\n")

# Bの逆行列
if rankB == len(B):
    invB = np.linalg.inv(B)
    print(f"行列B）ランク:{rankB}\n逆行列:\n{invB}\n")
else:
    print(f"行列B）ランク:{rankB},特異行列\n")