# -*- coding: utf-8 -*-
import housePriceData as hpd

#-------------------
# 1. housePriceDataクラスのインスタンス化
myData = hpd.housePriceData('../../data/house-prices-advanced-regression-techniques/train.csv')
#-------------------

#-------------------
# 2. MSSubClassとタイトルのリスト作成
levels = [20,30,60,70]

titles = []
titles.append('1-Story 1946 & Newer')
titles.append('1-Story 1945 & Older')
titles.append('2-Story 1946 & Newer')
titles.append('2-Story 1945 & Older')
#-------------------

#-------------------
# 3. 散布図をプロット
myData.plotScatter(titles,levels)
#-------------------
