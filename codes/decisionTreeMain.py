# -*- coding: utf-8 -*-
import numpy as np
import decisionTree as dt
import data

#-------------------
# 1. データの作成
myData = data.decisionTree()
myData.makeData(dataType=2)
#-------------------

#-------------------
# 2. 決定木の作成（version=1: ID3,version=2: CART）
myModel = dt.decisionTree(myData.Xtr,myData.Ytr,version=2)
myModel.train()
#-------------------