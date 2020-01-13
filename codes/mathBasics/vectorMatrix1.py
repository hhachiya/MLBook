# -*- coding: utf-8 -*-
import numpy as np

# 列ベクトル：2×1のnumpy arrayをを定義
w = np.array([ [5.0], [3.0] ])

# 行ベクトル：1×2のnumpy arrayをを定義
x = np.array([ [1.0, 5.0] ])

# 標準出力
print(f"ベクトルw）形:{w.shape}, 型:{w.dtype}\n{w}\n")
print(f"ベクトルx）形:{x.shape}, 型:{x.dtype}\n{x}")