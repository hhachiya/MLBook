# -*- coding: utf-8 -*-
import numpy as np
import kernelFunc as kf
import kernelSVM as svm
import data

#-------------------
# 1. データの作成
myData = data.classification(negLabel=-1.0,posLabel=1.0)
myData.makeData(dataType=5)
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
# 3. 標準化
xMean = np.mean(Xtr,axis=0)
xStd = np.std(Xtr,axis=0)
Xtr = (Xtr-xMean)/xStd
Xte = (Xte-xMean)/xStd
#-------------------

#-------------------
# 4. カーネル関数の作成
myKernel = kf.kernelFunc(kernelType=1,kernelParam=1)
#-------------------

#-------------------
# 5. SVMのモデルの学習
myModel = svm.SVM(Xtr,Ytr,kernelFunc=myKernel)
myModel.trainSoft(0.5)
#-------------------

#-------------------
# 6. SVMモデルの評価
print(f"モデルパラメータ:\nw={myModel.w}\nb={myModel.b}")
print(f"評価データの正解率={myModel.accuracy(Xte,Yte):.2f}")
#-------------------

#-------------------
# 7. 真値と予測値のプロット
myModel.plotModel2D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,
    title=f"学習正解率:{myModel.accuracy(Xtr,Ytr):.2f},評価正解率:{myModel.accuracy(Xte,Yte):.2f}",
    fName=f"../results/kernelSVM_result_{myData.dataType}_{myKernel.kernelType}_{str(myKernel.kernelParam).replace('.','')}.pdf",
    isLinePlot=True)
#-------------------
