# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt

class linearRegression():
    #-------------------
    # 1. 学習データの初期化
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×1のnumpy.ndarray）
    def __init__(self,X,Y):
        # 学習データの設定
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]  # 学習データ数
        self.xDim = X.shape[1]  # 入力の次元数
    #-------------------

    #-------------------
    # 2. 最小二乗法を用いてモデルパラメータを最適化
    def train(self):
        # 行列Xに「1」の要素を追加
        Z = np.concatenate([self.X,np.ones([self.dNum,1])],axis=1)

        # 分母の計算
        ZZ = 1/self.dNum * np.matmul(Z.T,Z)

        # 分子の計算
        ZY = 1/self.dNum * np.matmul(Z.T,self.Y)

        # パラメータvの最適化
        v = np.matmul(np.linalg.inv(ZZ),ZY)

        # パラメータw,bの決定
        self.w = v[:-1]
        self.b = v[-1]
    #-------------------
    
    #-------------------
    # L2ノルム正則化最小二乗法を用いてモデルパラメータを最適化
    # lamb: 正則化の重み係数（実数スカラー）
    def trainRegularized(self,lamb=0.1):
        # 行列Xに「1」の要素を追加
        Z = np.concatenate([self.X,np.ones([self.dNum,1])],axis=1)

        # 分母の計算
        ZZ = 1/self.dNum * np.matmul(Z.T,Z) + lamb * np.eye(self.xDim)

        # 分子の計算
        ZY = 1/self.dNum * np.matmul(Z.T,self.Y)

        # パラメータvの最適化
        v = np.matmul(np.linalg.inv(ZZ),ZY)

        # パラメータw,bの決定
        self.w = v[:-1]
        self.b = v[-1]
    #-------------------

    #-------------------
    # 3. 予測
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    def predict(self,x):
        return np.matmul(x,self.w) + self.b
    #-------------------
    
    #-------------------
    # 4. 平方平均二乗誤差（Root Mean Squared Error）
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def RMSE(self,X,Y):
        return np.sqrt(np.mean(np.square(self.predict(X)-Y)))
    #-------------------
        
    #-------------------
    # 5. 決定係数の計算
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def R2(self,X,Y):
        return 1 - np.sum(np.square(self.predict(X)-Y))/np.sum(np.square(Y-np.mean(Y,axis=0)))
    #-------------------
    
    #------------------- 
    # 6. データと線形モデルのプロット
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # fName: 画像の保存先（文字列）
    def plotResult(self,X=[],Y=[],xLabel="",yLabel="",fName=""):
        if X.shape[1] != 1: return
        
        fig = plt.figure(figsize=(8,5),dpi=100)
        
        # 線形モデルの直線の端点の座標を計算
        Xlin = np.array([[0],[np.max(X)]])
        Yplin = self.predict(Xlin)

        # データと線形モデルのプロット
        plt.plot(X,Y,'.',label="データ")
        plt.plot(Xlin,Yplin,'r',label="線形モデル")
        plt.legend()
        
        # 各軸の範囲とラベルの設定
        plt.ylim([0,np.max(Y)])
        plt.xlim([0,np.max(X)])
        plt.xlabel(xLabel,fontSize=14)
        plt.ylabel(yLabel,fontSize=14)
        
        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #------------------- 
