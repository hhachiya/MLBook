# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import kernelFunc as kf
import kernelSVM as svm
import matplotlib.pylab as plt
import pdb

dataType = 3
plotType = 2

####################
# データの選択
np.random.seed(1)

# 説明用のデータ
#Xtr = np.array([[3,4],[4,5],[5,9],[9,5],[8,4],[7,3]])
#Ytr = np.array([[1],[1],[1],[-1],[-1],[-1]])

if dataType == 1:# 線形分離可能な２つのガウス分布に従うデータ
  cov = [[1,-0.6], [-0.6, 1]]
  Xneg = np.random.multivariate_normal([1,2], cov, 100)
  Xpos = np.random.multivariate_normal([-2,-1], cov, 100)

elif dataType == 2: # 分類境界がアルファベッドのCの形をしている場合
  cov1 = [[1,-0.8], [-0.8, 1]]
  cov2 = [[1,0.8], [0.8, 1]]    
    
  Xneg = np.random.multivariate_normal([0.5, 1], cov1, 60)
  Xpos = np.random.multivariate_normal([-1, -1], cov1, 30)
  Xpos = np.append(Xpos,np.random.multivariate_normal([-1, 4], cov2, 30),axis=0)
  Xpos = Xpos[np.random.permutation(60)]

elif dataType == 3: # 複数の島がある場合
  cov = [[1,-0.8], [-0.8, 1]]    
  Xneg = np.random.multivariate_normal([0.5, 1], cov, 60)
  Xpos = np.random.multivariate_normal([-1, -1], cov, 30)
  Xpos = np.append(Xpos, np.random.multivariate_normal([2, 2], cov, 30),axis=0)

elif dataType == 4: # 物件データ
  data = pd.read_csv('house_prices_train.csv')
  Xneg = data[data['MSSubClass']==30][['GrLivArea','GarageArea']].values
  Xpos = data[data['MSSubClass']==60][['GrLivArea','GarageArea']].values
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

kernelTypes = [1,2]
kernelParams = [0.1,0.5,0.8,1,2,3,4]

####################
for kernelType in kernelTypes:
  for kernelParam in kernelParams:
    try:
      # SVMのモデルの学習と予測誤差
      myKernel = kf.kernelFunc(kernelType=kernelType, kernelParam=kernelParam)
      myModel = svm.SVM(Xtr,Ytr, kernelFunc=myKernel)
      myModel.trainSoft(0.5)      
    except:
      continue
      
    print(f"モデルパラメータ：\nw={myModel.w}\nb={myModel.b}")
    print(f"学習データの正解率={myModel.accuracy(Xtr,Ytr):.2f}")
    print(f"評価データの正解率={myModel.accuracy(Xte,Yte):.2f}")
    
    myModel.plotResult(fName=f"../figures/SVM_result_{dataType}_{kernelType}_{str(kernelParam).replace('.','')}.png",title=f"学習正解率：{myModel.accuracy(Xtr,Ytr):.2f}, 評価正解率：{myModel.accuracy(Xte,Yte):.2f}")    
####################