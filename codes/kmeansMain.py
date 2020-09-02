# -*- coding: utf-8 -*-
import numpy as np
import kmeans
import PCA
import data

#-------------------
# 1. データの作成
myData = data.unsupervised()
myData.makeData(dataType=1)
#-------------------

#-------------------
# 2. 主成分分析による2次元に次元削減
myModel = PCA.PCA(myData.X)
myModel.reduceDim(lowerDim=2)
X = myModel.F
#-------------------

#-------------------
# 3. k平均法を用いたクラスタリングと結果のプロット
myModel = kmeans.kmeans(X=X,K=3)

for ite in np.arange(10):

    # クラスター中心の出力
    print(f"反復{ite+1}、クラスター中心:\n{myModel.C}")
    
    # クラスターの更新
    myModel.updateCluster()
    
    # クラスターのプロット
    if X.shape[1] == 2:
        myModel.plotCluster(fName=f"../results/kmeans_results_{myData.dataType}_{myModel.K}_{ite}.pdf")
#-------------------
