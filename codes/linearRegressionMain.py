# -*- coding: utf-8 -*-
import numpy as np
import linearRegression as lr
import data

#-------------------
# 1. データの作成
myData = data.regression()
myData.makeData(dataType=1)

# # 標準化
# myData.X = (myData.X-np.mean(myData.X,axis=0))/np.std(myData.X,axis=0)
# myData.Y = (myData.Y-np.mean(myData.Y,axis=0))/np.std(myData.Y,axis=0)
#-------------------

#-------------------
# 2. データを学習と評価用に分割
dtrNum = int(len(myData.X)*0.9)  # 学習データ数

# 学習データ（全体の90%）
Xtr = myData.X[:dtrNum]
Ytr = myData.Y[:dtrNum]

# 評価データ（全体の10%）
Xte = myData.X[dtrNum:]
Yte = myData.Y[dtrNum:]
#-------------------

#-------------------
# 3. 線形モデルの学習
myModel = lr.linearRegression(Xtr,Ytr)
myModel.train()
#myModel.trainRegularized(lamb=1)
#-------------------

#-------------------
# 4. 線形モデルの評価
print(f"モデルパラメータ:\nw={myModel.w},\nb={myModel.b}")
print(f"平方平均二乗誤差={myModel.RMSE(Xte,Yte):.2f}ドル")
print(f"決定係数={myModel.R2(Xte,Yte):.2f}")
#-------------------

#-------------------
# 5. 線形モデルのプロット
myModel.plotResult(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/linearRegression_result_train_{myData.dataType}.pdf")
#-------------------