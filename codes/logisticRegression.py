# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt

# クラス
class logisticRegression():
    #-------------------
    # 1. 学習データの設定とモデルパラメータの初期化
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def __init__(self,X,Y):
        # 学習データの設定
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]  # 学習データ数
        self.xDim = X.shape[1]  # 入力の次元数
        
        # 行列Xに「1」の要素を追加
        self.Z = np.concatenate([self.X,np.ones([self.dNum,1])],axis=1)
        
        # モデルパラメータの初期値の設定
        self.w = np.random.normal(size=[self.xDim,1])
        self.b = np.random.normal(size=[1,1])
        
        # log(0)を回避するための微小値
        self.smallV = 10e-8
    #-------------------

    #-------------------
    # 2. 最急降下法を用いたモデルパラメータの更新
    # alpha: 学習率（スカラー実数）
    def update(self,alpha=0.1):

        # 予測の差の計算
        P,_ = self.predict(self.X)
        error = (P-self.Y)
        
        # パラメータの更新
        grad = 1/self.dNum*np.matmul(self.Z.T,error)
        v = np.concatenate([self.w,self.b],axis=0)
        v -= alpha * grad
        
        # パラメータw,bの決定
        self.w = v[:-1]
        self.b = v[[-1]]
    #-------------------

    #-------------------
    # 3. 予測
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    def predict(self,x):
        f_x = np.matmul(x,self.w) + self.b
        return 1/(1+np.exp(-f_x)),f_x
    #-------------------
    
    #-------------------
    # 4. 交差エントロピー損失の計算
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def CE(self,X,Y):
        P,_ = self.predict(X)
        return -np.mean(Y*np.log(P+self.smallV)+(1-Y)*np.log(1-P+self.smallV))
    #-------------------
    
    #-------------------
    # 5. 正解率の計算
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # thre: 閾値（スカラー）
    def accuracy(self,X,Y,thre=0.5):
        P,_ = self.predict(X)
        
        # 2値化
        P[P>thre] = 1
        P[P<=thre] = 0
        
        # 正解率
        accuracy = np.mean(Y==P)
        return accuracy
    #-------------------
    
    #------------------- 
    # 6. 真値と予測値のプロット（入力ベクトルが1次元の場合）
    # X: 入力データ（次元数×データ数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # fName: 画像の保存先（文字列）
    def plotModel1D(self,X=[],Y=[],xLabel="",yLabel="",fName=""):
        fig = plt.figure(figsize=(6,4),dpi=100)

        # 予測値
        P,_ = self.predict(X)

        # 真値と予測値のプロット
        plt.plot(X,Y,'b.',label="真値")
        plt.plot(X,P,'r.',label="予測")
        
        # 各軸の範囲とラベルの設定
        plt.yticks([0,0.5,1])
        plt.ylim([-0.1,1.1])
        plt.xlim([np.min(X),np.max(X)])
        plt.xlabel(xLabel,fontSize=14)
        plt.ylabel(yLabel,fontSize=14)
        plt.grid()
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------

    #-------------------
    # 7. 真値と予測値のプロット（入力ベクトルが2次元の場合）
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # title: タイトル（文字列）
    # fName: 画像の保存先（文字列）
    def plotModel2D(self,X=[],Y=[],xLabel="",yLabel="",title="",fName=""):
        #fig = plt.figure(figsize=(6,4),dpi=100)
        plt.close()
        
        # 真値のプロット（クラスごとにマーカーを変更）
        plt.plot(X[Y[:,0]==0,0],X[Y[:,0]==0,1],'cx',markerSize=14,label="ラベル0")
        plt.plot(X[Y[:,0]==1,0],X[Y[:,0]==1,1],'m.',markerSize=14,label="ラベル1")

        # 予測値のメッシュの計算
        X1,X2 = plt.meshgrid(plt.linspace(np.min(X[:,0]),np.max(X[:,0]),50),plt.linspace(np.min(X[:,1]),np.max(X[:,1]),50))
        Xmesh = np.hstack([np.reshape(X1,[-1,1]),np.reshape(X2,[-1,1])])
        Pmesh,_ = self.predict(Xmesh)
        Pmesh = np.reshape(Pmesh,X1.shape)

        # 予測値のプロット
        CS = plt.contourf(X1,X2,Pmesh,linewidths=2,cmap="bwr",alpha=0.3,vmin=0,vmax=1)

        # カラーバー
        CB = plt.colorbar(CS)
        CB.ax.tick_params(labelsize=14)

        # 各軸の範囲とラベルの設定
        plt.xlim([np.min(X[:,0]),np.max(X[:,0])])
        plt.ylim([np.min(X[:,1]),np.max(X[:,1])])
        plt.title(title,fontSize=14)
        plt.xlabel(xLabel,fontSize=14)
        plt.ylabel(yLabel,fontSize=14)
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------

    #------------------- 
    # 8. 学習と評価損失のプロット
    # trEval: 学習の損失
    # teEval: 評価の損失
    # yLabel: y軸のラベル（文字列）
    # fName: 画像の保存先（文字列）
    def plotEval(self,trEval,teEval,ylabel="損失",fName=""):
        fig = plt.figure(figsize=(6,4),dpi=100)
        
        # 損失のプロット
        plt.plot(trEval,'b',label="学習")
        plt.plot(teEval,'r',label="評価")
        
        # 各軸の範囲とラベルの設定
        plt.xlabel("反復",fontSize=14)
        plt.ylabel(ylabel,fontSize=14)
        plt.ylim([0,1.1])
        plt.legend()
        
        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------