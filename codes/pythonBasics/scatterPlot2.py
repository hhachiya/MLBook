# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pylab as plt

#-------------------
# 1. データの読み込み
data = pd.read_csv('../data/house-prices-advanced-regression-techniques/train.csv')
#-------------------

#-------------------
# 2. プロット 

# figureの初期化
fig = plt.figure()

ax=fig.add_subplot(1,1,1)  #グラフの位置指定

# MSSubClass=30の時のLotArea対SalePriceの散布図
ax.plot(data[data['MSSubClass']==30]['GrLivArea'],data[data['MSSubClass']==30]['SalePrice'],'.',label="1-Story 1945 & Older")

# MSSubClass=60の時のLotArea対SalePriceの散布図
ax.plot(data[data['MSSubClass']==60]['GrLivArea'],data[data['MSSubClass']==60]['SalePrice'],'.',label="2-Story 1946 & Newer")

ax.set_xlabel('GrLivArea')  # x軸のラベル
ax.set_ylabel('SalePrice')  # y軸のラベル
ax.legend()  # 凡例

fig.tight_layout()  # グラフ間に隙間をあける
plt.show()  # グラフの表示
#-------------------