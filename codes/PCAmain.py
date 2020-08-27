# -*- coding: utf-8 -*-
import numpy as np
import PCA
import data

#-------------------
# 1. データの作成
myData = data.unsupervised()
myData.makeData(dataType=1)
#-------------------

#-------------------
# 2. 主成分分析による次元削減
myModel = PCA.PCA(myData.X)
myModel.reduceDim(lowerDim=2)
#-------------------

#-------------------
# 3. モデルパラメータの表示
print(f"固有値:\nlambda={myModel.L}")
print(f"正規直交基底ベクトル:\nw=\n{myModel.W}")
#-------------------

#-------------------
# 4. データと主成分軸のプロット
# 次元削減後のデータ（主成分得点）のプロット
if myModel.lowerDim == 2:
    myModel.plotResult(fName=f"../results/PCA_result_{myData.dataType}.pdf")
    
# 主成分軸（平面）のプロット
if (myData.X.shape[1] == 3) & (myModel.lowerDim == 2):
    myModel.plotModel3D(xLabel=myData.labels[0],yLabel=myData.labels[1],zLabel=myData.labels[2],nGrids=20,
        fName=f"../results/PCA_result_plane_{myData.dataType}.pdf")
#-------------------

#-------------------
# 5. モデルの評価
# 寄与率と累積寄与率の計算
contRatio,cumContRatio = myModel.compContRatio()
print(f"寄与率:{np.round(contRatio,decimals=1)}")
print(f"累積寄与率:{np.round(cumContRatio,decimals=1)}")

# 主成分負荷量の計算
print(f"主成分負荷量:\n{np.round(myModel.compLoading(),decimals=1)}")
#-------------------
