# -*- coding: utf-8 -*-
import numpy as np

# 行列Aの定義
A=np.array([ [2,-3], [4,1]])

# 劣ベクトルbの定義
b=np.array([ [5], [-2] ])

# 行列式が非ゼロか否かを確認
if np.linalg.det(A) != 0:
    # Aの逆行列とベクトルbの積で解を求める
    x=np.matmul(np.linalg.inv(A),b)
    print(f"xの解）\n{x}")
else:
    # 解が存在しない
    print(f"解が存在しません．")
