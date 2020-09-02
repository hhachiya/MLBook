# -*- coding: utf-8 -*-
import numpy as np

# 平均0の分散1の正規分布に従って10回試行を行う
X1 = np.random.normal(0,1,10)
# 平均3の分散1の正規分布に従って10回試行を行う
X2 = np.random.normal(3,1,10)
# 平均-3の分散1の正規分布に従って10回試行を行う
X3 = np.random.normal(-3,1,10)

# 標準出力
print(f"データX1:\n{X1}")
print(f"データX2:\n{X2}")
print(f"データX3:\n{X3}")