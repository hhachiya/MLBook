# -*- coding: utf-8 -*-
import numpy as np
import FA
import data

#-------------------
# 1. データの作成
myData = data.unsupervised()
myData.makeData(dataType=1)
#-------------------

#-------------------
# 2. 主因子法による因子の抽出
myModel = FA.FA(myData.X)
myModel.extractFactor(lowerDim=2)
#-------------------

#-------------------
# 3. 因子の表示
print(f"因子負荷量:\nW=\n{np.round(myModel.W,decimals=2)}")
print(f"独自因子の分散:\nE=\n{np.round(myModel.E,decimals=2)}")
#-------------------

#-------------------
# 4. 因子のプロット
# 主成分得点のプロット（２次元への次元削減時のみ実行可）
if myModel.lowerDim == 2:
    myModel.drawRadarChart(labels=myData.labels,fName=f"../results/FA_result_{myData.dataType}.pdf")
#-------------------

#-------------------
# 5. 共通性と独自性の計算
comVar,uniVar = myModel.compVariances()
print(f"共通性:{np.round(comVar,decimals=2)}")
print(f"独自性:{np.round(uniVar,decimals=2)}")
#-------------------
