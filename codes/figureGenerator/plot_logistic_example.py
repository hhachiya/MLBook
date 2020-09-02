# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

#------------------- 
# データの読み込み
data = pd.read_csv('../../data/house-prices-advanced-regression-techniques/train.csv')
#------------------- 

#------------------- 
# 2値分類のデータの作成
X = data[(data['MSSubClass']==30) | (data['MSSubClass']==60)][['GrLivArea']].values
Y = data[(data['MSSubClass']==30) | (data['MSSubClass']==60)][['MSSubClass']].values
#------------------- 

#------------------- 
# プロット
#fig = plt.figure(figsize=(6,4),dpi=100)
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(X[Y==30],Y[Y==30],'r.',label="1階建て、1945年以前建設")
ax.plot(X[Y==60],Y[Y==60],'b.',label="2階建て、1946年以降建設")
ax.set_xlabel("居住面積x[平方フィート]",fontSize=14)
ax.set_ylabel("建物の等級y",fontSize=14)
ax.legend(fontsize=14)

plt.tight_layout()
#plt.show()
plt.savefig("logistic_regression_example.pdf")
#------------------- 
