# -*- coding: utf-8 -*-
import numpy as np
import logisticRegression as lr
import data

#-------------------
# 1. データの作成
myData = data.classification(negLabel=0,posLabel=1)
myData.makeData(dataType=1)
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
# 3. 入力データの標準化
xMean = np.mean(Xtr,axis=0)
xStd = np.std(Xtr,axis=0)
Xtr = (Xtr-xMean)/xStd
Xte = (Xte-xMean)/xStd
#-------------------

#-------------------
# 4. ロジスティックモデルの学習と評価
myModel = lr.logisticRegression(Xtr,Ytr)

trLoss = []
teLoss = []

for ite in range(1001):
    trLoss.append(myModel.CE(Xtr,Ytr))
    teLoss.append(myModel.CE(Xte,Yte))
    
    if ite%100==0:
        print(f"反復:{ite}")
        print(f"モデルパラメータ:\nw={myModel.w},\nb={myModel.b}")
        print(f"平均交差エントロピー損失={myModel.CE(Xte,Yte):.2f}")
        print(f"正解率={myModel.accuracy(Xte,Yte):.2f}")
        print("----------------")
        
    # モデルパラメータの更新
    myModel.update(alpha=1)
#-------------------

#-------------------
# 5. 真値と予測値のプロット
if Xtr.shape[1] == 1:
    myModel.plotModel1D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/logistic_result_train_{myData.dataType}.pdf")
elif Xtr.shape[1] == 2:
    myModel.plotModel2D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/logistic_result_train_{myData.dataType}.pdf")
#-------------------

#-------------------
# 6. 学習と評価損失のプロット
myModel.plotEval(trLoss,teLoss,fName=f"../results/logistic_CE_{myData.dataType}.pdf")
#myModel.plotLoss(trLoss,teLoss)
#-------------------