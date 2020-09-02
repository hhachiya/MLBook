# -*- coding: utf-8 -*-
import numpy as np

class kernelFunc():
    #-------------------
    # kernelType: 線形カーネル(0),ガウスカーネル(1)、多項式カーネル(2)
    # kernelParam: カーネルの作成に用いるパラメータ（スカラー）
    def __init__(self,kernelType=0,kernelParam=1):
        self.kernelType = kernelType
        self.kernelParam = kernelParam

        # カーネル関数の設定
        kernelFuncs = [self.linear,self.gauss,self.poly]
        self.createMatrix = kernelFuncs[kernelType]
    #-------------------

    #-------------------
    # 線形カーネル
    def linear(self,X1,X2):
        return np.matmul(X1,X2.T)
    #-------------------
        
    #-------------------
    # ガウスカーネル
    # X1: 入力データ（データ数×次元数のnumpy.ndarray）
    # X2: 入力データ（データ数×次元数のnumpy.ndarray）
    def gauss(self,X1,X2):
        X1Num = len(X1)
        X2Num = len(X2)
        
        # X1とX2の全ペア間の距離の計算
        X1 = np.tile(np.expand_dims(X1.T,axis=2),[1,1,X2Num])
        X2 = np.tile(np.expand_dims(X2.T,axis=1),[1,X1Num,1])
        dist = np.sum(np.square(X1-X2),axis=0)
        
        # グラム行列（X1のデータ数×X2のデータ数）
        K = np.exp(-dist/(2*(self.kernelParam**2)))
        
        return K
    #-------------------
    
    #-------------------
    # 多項式カーネル
    # X1: 入力データ（データ数×次元数のnumpy.ndarray）
    # X2: 入力データ（データ数×次元数のnumpy.ndarray）
    def poly(self,X1,X2):

        # グラム行列（X1のデータ数×X2のデータ数）
        K = (np.matmul(X1,X2.T)+1)**self.kernelParam

        return K
    #-------------------
