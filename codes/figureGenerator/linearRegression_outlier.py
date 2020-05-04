# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pdb
import copy

####################
# クラス
class linearRegression():
    #-------------------
    # 学習データの初期化
    # X: 入力データ（次元数×データ数のnumpy.array）
    # Y: 出力データ（データ数×1のnumpy.array）
    def __init__(self,X,Y):
        # 学習データの設定
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]
        self.xDim = X.shape[1]
    #-------------------

    #-------------------
    # 最小二乗法を用いてモデルパラメータを最適化
    def train(self):
        # 行列Xに「1」の要素を追加
        Z = np.append(self.X,np.ones([self.dNum,1]),axis=1)

        # 分母の計算
        ZZ = np.matmul(Z.T,Z) + 0.01*np.eye(self.xDim+1)

        # 分子の計算
        ZY = np.matmul(Z.T,self.Y)

        # パラメータvの最適化
        v = np.matmul(np.linalg.inv(ZZ),ZY)

        # パラメータw,bの決定
        self.w = v[:-1]
        self.b = v[-1]
    #-------------------

    #-------------------
    # 予測
    # X: 入力データ（次元数×データ数のnumpy.array）
    def predict(self,x):
        return np.matmul(x,self.w) + self.b
    #-------------------
    
    #-------------------
    # 平均平方二乗誤差（Root Mean Squared Error）
    # X: 入力データ（次元数×データ数のnumpy.array）
    # Y: 出力データ（データ数×1のnumpy.array）
    def RMSE(self,X,Y):
        return np.sqrt(np.mean(np.square(self.predict(X) - Y)))
    #-------------------

    #-------------------
    # 決定係数
    # X: 入力データ（次元数×データ数のnumpy.array）
    # Y: 出力データ（データ数×1のnumpy.array）
    def R2(self,X,Y):
        return 1 - np.sum(np.square(self.predict(X) - Y))/np.sum(np.square(Y-np.mean(Y,axis=0)))
    #-------------------
####################

####################
# メイン

#------------------- 
# データの読み込み
data = pd.read_csv('../../data/house-prices-advanced-regression-techniques/train.csv')
#X = data[data['MSSubClass']==60][['GrLivArea','GarageArea','PoolArea','BedroomAbvGr','TotRmsAbvGrd']].values
X = data[data['MSSubClass']==60][['GrLivArea']].values
Y = data[data['MSSubClass']==60][['SalePrice']].values
#------------------- 

#------------------- 
# 学習と評価に分割
dtrNum = int(len(X)*0.9)
Xtr = X[:dtrNum]
Ytr = Y[:dtrNum]
Xte = X[dtrNum:]
Yte = Y[dtrNum:]

# 外れ値
YtrOut = copy.deepcopy(Ytr)
outInds = np.where(YtrOut>700000)
YtrOut[outInds]-=700000
#------------------- 

#------------------- 
# 線形回帰モデルの学習と予測誤差
myModel = linearRegression(Xtr,Ytr)
myModel.train()
myModelOut = linearRegression(Xtr,YtrOut)
myModelOut.train()
print("【外れ値なし】")
print(f"モデルパラメータ：\nw={myModel.w},\nb={myModel.b}")
print(f"平方平均二乗誤差={myModel.RMSE(Xte,Yte):.2f}ドル")
print(f"決定係数={myModel.R2(Xte,Yte):.2f}")

print("【外れ値あり】")
print(f"モデルパラメータ：\nw={myModelOut.w},\nb={myModelOut.b}")
print(f"平方平均二乗誤差={myModelOut.RMSE(Xte,Yte):.2f}ドル")
print(f"決定係数={myModelOut.R2(Xte,Yte):.2f}")
#------------------- 

#------------------- 
# 学習データと線形回帰モデルのプロット
# 直線
Xlin = np.array([[0],[np.max(X)]])
Yplin = myModel.predict(Xlin)
YOutplin = myModelOut.predict(Xlin)

plt.plot(Xtr,Ytr,'k.',label="学習データ")
plt.plot(Xtr[outInds],Ytr[outInds],'b.',label="学習データ（外れ値なし）",markerSize=10)
plt.plot(Xtr[outInds],YtrOut[outInds],'r.',label="学習データ（外れ値あり）",markerSize=10)
plt.plot(Xlin,Yplin,'b',label="線形回帰モデル（外れ値なし）")
plt.plot(Xlin,YOutplin,'r',label="線形回帰モデル（外れ値あり）")
plt.ylim([0,np.max(Y)+np.max(Y)*0.05])
plt.xlim([0,np.max(X)+np.max(X)*0.05])
plt.xlabel("居住面積x[平方フィート]",fontSize=14)
plt.ylabel("物件価格y[ドル]",fontSize=14)
plt.legend(fontsize=12)
#plt.show()
plt.tight_layout()
plt.savefig("linear_regression_result_outlier.pdf")
####################

