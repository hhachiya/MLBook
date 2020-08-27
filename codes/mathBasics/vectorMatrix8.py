# -*- coding: utf-8 -*-
import numpy as np

# 行列Aの定義
A = np.array([ [3,1],[1,3] ])

# 固有値問題
L,V = np.linalg.eig(A)

# 標準出力
print(f"行列Aの固有値:\n{L}\n")

print(f"行列Aの固有ベクトル:\n{V}\n")

print(f"固有ベクトルの内積:{np.matmul(V[:,[0]].T,V[:,[1]])}\n")

print(f"固有値の和:{np.sum(L)}\n")

print(f"行列Aのトレース（対角成分の和）:{np.sum(np.diag(A))}\n")

print(f"固有値の積:{np.prod(L)}\n")

print(f"行列Aの行列式:{np.linalg.det(A):.1f}\n")

print(f"行列Vの逆行列:\n{np.linalg.inv(V)}\n")

print(f"行列Vの転置:\n{V.T}\n")