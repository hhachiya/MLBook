# -*- coding: utf-8 -*-
import numpy as np

values = np.array([10,3,1,5,8,6])

#-------------------
# 通常のfor文
# 空のリスト
passed_values = []
for ind in np.arange(len(values)):
    # 通常のif文
    if values[ind] > 5:
        passed_values.append(values[ind])

# 結果を標準出力
print("5以上の値",passed_values)
#-------------------

#-------------------
# リスト内包表記のfor文とif文
passed_values = values[[True if values[ind]>5 else False for ind in np.arange(len(values))]]

# 結果を標準出力
print("5以上の値",passed_values)
#-------------------