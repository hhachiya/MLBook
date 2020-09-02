# -*- coding: utf-8 -*-
import numpy as np

# p=6/13（青色のボールの確率）のベルヌーイ分布に従って10回試行を行う
X1 = np.random.binomial(1,6/13,10)
X2 = np.random.binomial(1,6/13,10)
X3 = np.random.binomial(1,6/13,10)

# 標準出力
print(f"データX1:\n{X1}")
print(f"データX2:\n{X2}")
print(f"データX3:\n{X3}")