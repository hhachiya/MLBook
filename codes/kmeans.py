import numpy as np
import matplotlib.pylab as plt
import os
import pdb

####################
# クラス
class kmeans:
    #-------------------
    # 1. k-meansの各種初期化
    # X: 学習データ（データ数×次元数のnumpy.ndarray）
    # K: クラスター数（整数スカラー）
    def __init__(self,X,K=5):
        # パラメータの設定
        self.dNum = len(X)
        self.K = K
        self.X = X
        
        # カラーコードの設定
        self.cmap = ['#FF0000','#00B0F0','#FF00FF','#00FF00','#0000FF']

        # ランダムにクラスター中心を設定
        self.C = X[np.random.permutation(self.dNum)[:self.K],:]
    #-------------------

    #-------------------
    # 2. クラスターの中心の更新
    def updateCluster(self):
        
        # XとCの全ペア間の距離の計算
        Ctmp = np.tile(np.expand_dims(self.C.T,axis=2),[1,1,self.dNum])
        Xtmp = np.tile(np.expand_dims(self.X.T,axis=1),[1,self.K,1])
        dist = np.sum(np.square(Ctmp - Xtmp),axis=0)
        
        # 距離が最小のクラスターのインデックスを選択
        self.cInd = np.argmin(dist,axis=0)
        
        # 各クラスタに属しているデータ点の平均値を計算し、新しいクラスター中心に設定
        self.C = np.array([np.mean(self.X[self.cInd==c],axis=0) for c in range(self.K)])
    #-------------------
    
    #-------------------
    # 3. 学習データとクラスターのプロット
    # （特徴数が2の場合）
    # fName：画像の保存先（文字列）
    def plotCluster(self,xLabel="$x_1$",yLabel="$x_2$",fName=""):
        plt.close()
        
        # クラスターごとに学習データとクラスター中心のプロット
        for c in range(self.K):
            plt.plot(self.X[self.cInd==c,0],self.X[self.cInd==c,1],'s',color=self.cmap[c],markeredgecolor='k',markersize='8')
            plt.plot(self.C[c,0],self.C[c,1],'o',color=self.cmap[c],markeredgecolor='k',markersize='16')

        # ラベル、グリッドおよび範囲の設定
        plt.xlabel(xLabel,fontsize=14)
        plt.ylabel(yLabel,fontsize=14)
        plt.tick_params(labelsize=14)
        plt.grid(True)
        plt.xlim(np.min(self.X[:,0]),np.max(self.X[:,0]))
        plt.ylim(np.min(self.X[:,1]),np.max(self.X[:,1]))


        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------
####################


