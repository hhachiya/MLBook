# -*- coding: utf-8 -*-
import numpy as np

# 学習率の設定
alpha = 0.2

# パラメータの初期化
w = 1

for ite in range(50):
    print(f"反復:{ite}, w={round(w,2):.2f}")

    # 最急降下法によるパラメータの更新
    w -= alpha * 3 * w**2

