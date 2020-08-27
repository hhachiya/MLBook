# -*- coding: utf-8 -*-
import numpy as np

# 行列：3行2列のnumpy.ndarrayを定義
X = np.array([ [1.0,2.0],[2.0,4.0],[3.0,6.0] ])

# 列ベクトルw：2行1列のnumpy.ndarrayを定義
w = np.array([ [5.0],[3.0] ])

# 列ベクトルb：3行1列のnumpy.ndarrayを定義
b = np.array([ [1.0],[1.0],[1.0] ])

# 行列とベクトルの積と和
res = np.matmul(X,w) + b

# 標準出力
print(f"積和の結果）\n{res}")