# -*- coding: utf-8 -*-
import numpy as np
import neuralNetwork as nn
import data

#-------------------
# 0. ハイパーパラメータの設定
dataType = 5    # データの種類
activeType = 2  # 活性化関数の種類
hDim = 20       # 中間層のノード数
alpha = 1       # 学習率
rate = 0.5      # ノード選択確率（ドロップアウト）
#-------------------

#-------------------
# 1. データの作成
myData = data.classification(negLabel=0,posLabel=1)
myData.makeData(dataType=dataType)
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
# 4. ニューラルネットワークの学習と評価
myModel = nn.neuralNetwork(Xtr,Ytr,hDim=hDim,activeType=activeType)

trLoss = []
teLoss = []
trAcc = []
teAcc = []

for ite in range(1001):
    # 学習データの設定
    Xbatch = Xtr
    Ybatch = Ytr

    # 損失と正解率の記録
    trLoss.append(myModel.CE(Xtr,Ytr))
    teLoss.append(myModel.CE(Xte,Yte))
    trAcc.append(myModel.accuracy(Xtr,Ytr))
    teAcc.append(myModel.accuracy(Xte,Yte))
    
    # 評価の出力
    if ite%100 == 0:
        print(f"反復:{ite}")
        print(f"平均交差エントロピー損失={myModel.CE(Xte,Yte):.2f}")
        print(f"正解率={myModel.accuracy(Xte,Yte):.2f}")
        print("----------------")

    # パラメータの更新
    myModel.update(Xbatch,Ybatch,alpha=alpha)
    #myModel.updateDropout(Xbatch,Ybatch,alpha=alpha,rate=rate)
#-------------------

#-------------------
# 5. 真値と予測値のプロット
if Xtr.shape[1] == 1:
    myModel.plotModel1D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,
        fName=f"../results/neuralNet_result_train_{myData.dataType}_{activeType}_{hDim}_{str(alpha).replace('.','')}.png")
        
elif Xtr.shape[1] == 2:
    myModel.plotModel2D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,
        fName=f"../results/neuralNet_result_train_{myData.dataType}_{activeType}_{hDim}_{str(alpha).replace('.','')}.png")
#-------------------

#-------------------
# 6. 学習と評価損失のプロット
myModel.plotEval(trLoss,teLoss,"損失",fName=f"../results/neuralNet_CE_{myData.dataType}_{activeType}_{hDim}_{str(alpha).replace('.','')}.png")
myModel.plotEval(trAcc,teAcc,"正解率",fName=f"../results/neuralNet_accuracy_{myData.dataType}_{activeType}_{hDim}_{str(alpha).replace('.','')}.png")
#-------------------
