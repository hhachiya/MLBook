# -*- coding: utf-8 -*-
import numpy as np

# 列ベクトル：2行1列のnumpy.ndarrayを定義
w = np.array([ [5.0],[3.0] ])

# 行ベクトル：1行2列のnumpy.ndarrayを定義
x = np.array([ [1.0,5.0] ])

# 全ての要素が0または1の1行5列のnumpy.ndarrayを生成
zeros = np.zeros([1,5])
ones = np.ones([1,5])

# 一様分布または正規分布に従ってランダムに1行5列のnumpy.ndarrayを生成
uniform = np.random.rand(1,5)
normal = np.random.normal(size=[1,5])

# 標準出力
print(f"ベクトルw）形:{w.shape},型:{w.dtype}\n{w}\n")

print(f"ベクトルx）形:{x.shape},型:{x.dtype}\n{x}\n")

print(f"ベクトルzeros）形:{zeros.shape},型:{zeros.dtype}\n{zeros}\n")

print(f"ベクトルones）形:{ones.shape},型:{ones.dtype}\n{ones}\n")

print(f"ベクトルuniform）形:{uniform.shape},型:{uniform.dtype}\n{uniform}\n")

print(f"ベクトルnormal）形:{normal.shape},型:{normal.dtype}\n{normal}")