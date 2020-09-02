# -*- coding: utf-8 -*-
import numpy as np
import kernelFunc as kf
import kernelSVM as svm
import data
import matplotlib.pylab as plt

kernelType = 2

#-------------------
# 1. データの作成
myData = data.classification(negLabel=-1.0,posLabel=1.0)
myData.makeData(dataType=4)
#-------------------

#-------------------
# 2. データを学習と評価用に分割
dNum = int(len(myData.X)*0.9)  # 学習データ数
# 学習データ（全体の90%）
Xtr = myData.X[:dNum]
Ytr = myData.Y[:dNum]

# 評価データ（全体の10%）
Xte = myData.X[dNum:]
Yte = myData.Y[dNum:]
#-------------------

#-------------------
# 3. 標準化
xMean = np.mean(Xtr,axis=0)
xStd = np.std(Xtr,axis=0)
Xtr = (Xtr-xMean)/xStd
Xte = (Xte-xMean)/xStd
#-------------------

#-------------------
# 3.5. モデル選択
# ハイパーパラメータの候補
if kernelType == 1: # ガウスカーネルの幅
    kernelParams = [0.1,0.25,0.5,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0]
elif kernelType == 2: # 多項式カーネルのオーダー
    kernelParams = [1.0,2.0,3.0,4.0,5.0]

# fold数
foldNum = 5

# 各foldで用いる学習データ数
dNumFold = int(dNum/foldNum)

# ランダムにデータを並べ替える
randInds = np.random.permutation(len(Xtr))

# 正解率を格納する変数
accuracies = np.zeros([len(kernelParams),foldNum])

# ハイパーパラメータの候補のループ
for paramInd in range(len(kernelParams)):

    # 交差検証によ正解率の推定
    for foldInd in range(foldNum):
    
        # 学習データ数dNumFold分左にシフト
        randIndsTmp = np.roll(randInds,-dNumFold*foldInd)
        
        # 学習と評価データの分割
        XtrTmp = Xtr[randIndsTmp[dNumFold:]]
        YtrTmp = Ytr[randIndsTmp[dNumFold:]]
        XteTmp = Xtr[randIndsTmp[:dNumFold]]
        YteTmp = Ytr[randIndsTmp[:dNumFold]]

        try:
            # 手順1) SVMのモデルの学習
            myKernel = kf.kernelFunc(kernelType=kernelType,kernelParam=kernelParams[paramInd])
            myModel = svm.SVM(XtrTmp,YtrTmp,kernelFunc=myKernel)
            myModel.trainSoft(0.5)
        except:
            continue

        # 手順2) 評価データに対する正解率を格納
        accuracies[paramInd,foldInd] = myModel.accuracy(XteTmp,YteTmp)

# 手順3) 平均正解率が最大のパラメータ
selectedParam = kernelParams[np.argmax(np.mean(accuracies,axis=1))]
print(f"選択したパラメータ:{selectedParam}")
#-------------------

#-------------------
# 3.75 正解率のプロット
plt.plot(kernelParams,np.mean(accuracies,axis=1),'r-o',lineWidth=2)
plt.xlabel("カーネルパラメータ",fontSize=14)
plt.ylabel("推定した正解率",fontSize=14)
plt.savefig(f"../results/kernelSVM_CV_{myData.dataType}_{kernelType}.pdf")
#-------------------

#-------------------
# 4. カーネル関数の作成
myKernel = kf.kernelFunc(kernelType=kernelType,kernelParam=selectedParam)
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
    title=f"学習正解率：{myModel.accuracy(Xtr,Ytr):.2f},評価正解率：{myModel.accuracy(Xte,Yte):.2f}",
    fName=f"../results/kernelSVM_result_{myData.dataType}_{myKernel.kernelType}_{str(myKernel.kernelParam).replace('.','')}.png",
    isLinePlot=False)
#-------------------