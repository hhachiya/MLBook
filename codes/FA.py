# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# クラス
class FA:
    #-------------------
    # 1. 因子分析の各種初期化
    # X: 学習データ（データ数×次元数のnumpy.ndarray）
    def __init__(self,X):

        # データの標準化
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        self.X = (X - self.mean)/self.std
    #-------------------

    #-------------------
    # 2. 主因子法を用いた因子負荷量および独自因子の分散の計算
    # lowerDim: 因子数（整数スカラー）
    def extractFactor(self,lowerDim):
        self.lowerDim = lowerDim
        
        # 分散共分散行列
        cov = np.cov(self.X.T,bias=1)
        
        # 固有値問題
        L,V = np.linalg.eig(cov)
        
        # 固有値と固有ベクトルの固有値の降順でソート
        inds = np.argsort(L)[::-1]
        self.L = L[inds]
        self.V = V[:,inds]
        
        # 因子負荷量の計算
        self.W = np.matmul(np.diag(np.sqrt(self.L)),self.V.T)
        self.W = self.W[:lowerDim,:]
        
        # 独自因子の分散の計算
        self.E = cov - np.matmul(self.W.T,self.W)
    #-------------------
    
    #-------------------
    # 3. 因子負荷量のレーダーチャートのプロット（因子数が２つの場合にのみ対応）
    # labels: 特徴量のラベル（1×次元数のリスト）
    # fName: 画像の保存先（文字列）
    def drawRadarChart(self,labels=[],fName=""):
        import matplotlib.pylab as plt
        
        # 円の作成
        thetas = np.arange(0,2*np.pi,0.1)
        C = np.array([[np.cos(theta),np.sin(theta)] for theta in thetas])
        
        # プロットの準備
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        
        # 円のプロット
        ax.plot(C[:,0],C[:,1],'b-')

        # x軸とy軸を原点中心でプロット
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        # 各特徴量の因子負荷量を点でプロット
        plt.plot(self.W[0,:],self.W[1,:],'ro',markerSize=14)
        
        # 各特徴量をテキストでプロット
        for ind in range(self.W.shape[1]):
            if not len(labels):
                label = f"x{ind}"
            else:
                label = labels[ind]
            plt.text(self.W[0,ind]+0.05,self.W[1,ind],label,size=14)
        
        # 各軸の範囲とラベルの設定
        ax.set_xlabel("第1因子負荷量",fontSize=14)
        ax.set_ylabel("第2因子負荷量",fontSize=14)
        ax.xaxis.set_label_coords(0.8,0.55)
        ax.yaxis.set_label_coords(0.55,0.8)

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------
    
    #-------------------
    # 4. 共通因子と独自因子の計算
    def compVariances(self):
        # 共通性と独自性の計算
        comVar = np.sum(np.square(self.W),axis=0)
        uniVar = 1 - comVar
        
        return comVar,uniVar
    #-------------------