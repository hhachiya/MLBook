# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import kernelFunc as kf
import kernelSVM as svm
import matplotlib.pylab as plt
import pdb

dataType = 3
plotType = 2
kernelType = 1

####################
# データ作成
np.random.seed(1)

if dataType == 1:# 線形分離可能な２つのガウス分布に従うデータ
  cov = [[1,-0.6], [-0.6, 1]]
  Xneg = np.random.multivariate_normal([1,2], cov, 100)
  Xpos = np.random.multivariate_normal([-2,-1], cov, 100)
  xlabel = "$x_1$（標準化）"
  ylabel = "$x_2$（標準化）"

elif dataType == 2: # 分類境界がアルファベッドのCの形をしている場合
  cov1 = [[1,-0.8], [-0.8, 1]]
  cov2 = [[1,0.8], [0.8, 1]]    
    
  Xneg = np.random.multivariate_normal([0.5, 1], cov1, 60)
  Xpos = np.random.multivariate_normal([-1, -1], cov1, 30)
  Xpos = np.append(Xpos,np.random.multivariate_normal([-1, 4], cov2, 30),axis=0)
  Xpos = Xpos[np.random.permutation(60)]
  xlabel = "$x_1$（標準化）"
  ylabel = "$x_2$（標準化）"

elif dataType == 3: # 複数の島がある場合
  cov = [[1,-0.8], [-0.8, 1]]    
  Xneg = np.random.multivariate_normal([0.5, 1], cov, 60)
  Xpos = np.random.multivariate_normal([-1, -1], cov, 30)
  Xpos = np.append(Xpos, np.random.multivariate_normal([2, 2], cov, 30),axis=0)
  xlabel = "$x_1$（標準化）"
  ylabel = "$x_2$（標準化）"

elif dataType == 4: # ボストン物件データ、建物等級の分類
  data = pd.read_csv('house_prices_train.csv')
  Xneg = data[data['MSSubClass']==30][['GrLivArea']].values
  Xpos = data[data['MSSubClass']==60][['GrLivArea']].values
  xlabel = "居住面積x[平方フィート]"
  ylabel = "建物等級ラベルy"

elif dataType == 5: # ボストン物件データ、建物等級の分類
  data = pd.read_csv('house_prices_train.csv')
  Xneg = data[data['MSSubClass']==30][['GrLivArea','GarageArea']].values
  Xpos = data[data['MSSubClass']==60][['GrLivArea','GarageArea']].values
  xlabel = "居住面積x[平方フィート]"
  ylabel = "車庫面積x[平方フィート]"
  
####################


####################
# 学習と評価に分割
dNum = 50
teNum = 10
Xtr = np.concatenate([Xneg[:dNum],Xpos[:dNum]],axis=0)
Ytr = np.concatenate([np.ones(shape=[dNum,1])*-1,np.ones(shape=[dNum,1])],axis=0)
Xte = np.concatenate([Xneg[dNum:dNum+teNum],Xpos[dNum:dNum+teNum]],axis=0)
Yte = np.concatenate([np.ones(shape=[teNum,1])*-1,np.ones(shape=[teNum,1])],axis=0)
####################

####################
# 標準化
xMean = np.mean(Xtr,axis=0)
xStd = np.std(Xtr,axis=0)
Xtr = (Xtr-xMean)/xStd
Xte = (Xte-xMean)/xStd
####################

####################
# モデル選択
if kernelType == 1: # ガウスカーネルの幅
  kernelParams = [0.1,0.25,0.5,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0]
elif kernelType == 2: # 多項式カーネルのオーダー
  kernelParams = [1.0,2.0,3.0,4.0,5.0]

# fold数
foldNum = 5

# 各foldで用いる学習データ数
dNumTmp = int(len(Xtr)/foldNum)

# 1) ランダムにデータを並べ替える
randInds = np.random.permutation(len(Xtr))

# 正解率を格納する変数
accuracies = np.zeros([len(kernelParams), foldNum])

for paramInd in range(len(kernelParams)):

  # 交差確認によ正解率の推定
  for foldInd in range(foldNum):
  
    # 2) 学習データ数dNumTmp分左にシフト
    randIndsTmp = np.roll(randInds,-dNumTmp*foldInd)
    
    # 学習・評価データの設定
    XtrTmp = Xtr[randIndsTmp[dNumTmp:]]
    YtrTmp = Ytr[randIndsTmp[dNumTmp:]]
    XteTmp = Xtr[randIndsTmp[:dNumTmp]]
    YteTmp = Ytr[randIndsTmp[:dNumTmp]]

    try:
      # SVMのモデルの学習
      myKernel = kf.kernelFunc(kernelType=kernelType, kernelParam=kernelParams[paramInd])
      myModel = svm.SVM(XtrTmp,YtrTmp, kernelFunc=myKernel)
      myModel.trainSoft(0.5)
    except:
      continue

    # 3) 評価データに対する正解率を格納
    accuracies[paramInd,foldInd] = myModel.accuracy(XteTmp,YteTmp)

# 4) 平均正解率が最大のパラメータ
selectedParam = kernelParams[np.argmax(np.mean(accuracies,axis=1))]
print(f"選択したパラメータ:{selectedParam}")

# 正解率のプロット
plt.plot(kernelParams,np.mean(accuracies,axis=1),'r-o',lineWidth=2)
#plt.ylim([0.5,1])
plt.xlabel("カーネルパラメータ",fontSize=14)
plt.ylabel("推定した正解率",fontSize=14)
plt.savefig(f"../figures/SVM_kernel_CV_{dataType}_{kernelType}_{str(selectedParam).replace('.','')}.png")
####################

####################
# SVMのモデルの学習と予測誤差
myKernel = kf.kernelFunc(kernelType=kernelType, kernelParam=selectedParam)
myModel = svm.SVM(Xtr,Ytr, kernelFunc=myKernel)
myModel.trainSoft(0.5)      
print(f"モデルパラメータ：\nw={myModel.w}\nb={myModel.b}")
print(f"学習データの正解率={myModel.accuracy(Xtr,Ytr):.2f}")
print(f"評価データの正解率={myModel.accuracy(Xte,Yte):.2f}")
    
myModel.plotResult(xlabel=xlabel,ylabel=ylabel,
  fName=f"../figures/SVM_result_{dataType}_{kernelType}_{str(selectedParam).replace('.','')}.png",
  title=f"学習誤差：{myModel.accuracy(Xtr,Ytr):.2f}, 評価誤差：{myModel.accuracy(Xte,Yte):.2f}")    
####################