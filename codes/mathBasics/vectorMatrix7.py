# -*- coding: utf-8 -*-
import numpy as np

# 行列Aの定義
A = np.array([ [3,2],[4,1] ])

# 固有値問題の解
L,V = np.linalg.eig(A)

# 標準出力
print(f"固有値:\n{L}")
print(f"固有ベクトル:\n{V}")
