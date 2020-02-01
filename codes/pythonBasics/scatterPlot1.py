# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pylab as plt

#-------------------
# データの読み込み
data = pd.read_csv('../../data/house-prices-advanced-regression-techniques/train.csv')
#-------------------

#-------------------
# 散布図のプロット 

# figureの初期化
fig = plt.figure()

# GrLivArea対SalePriceの散布図
ax=fig.add_subplot(1,2,1)   #グラフの位置指定（1行2列の1列目）
ax.plot(data['GrLivArea'],data['SalePrice'],'.')
ax.set_xlabel('GrLivArea',fontSize=14)  #x軸のラベル
ax.set_ylabel('SalePrice',fontSize=14)  #y軸のラベル

# MSSubClass対SalePriceの散布図
ax=fig.add_subplot(1,2,2)   #グラフの位置指定（1行2列の2列目）
ax.plot(data['MSSubClass'],data['SalePrice'],'.')
ax.set_xlabel('MSSubClass',fontSize=14) #x軸のラベル
ax.set_ylabel('SalePrice',fontSize=14)  #y軸のラベル

fig.tight_layout()  # グラフ間に隙間をあける
plt.show()  # グラフの表示
#-------------------