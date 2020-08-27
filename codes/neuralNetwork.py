# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pdb

# クラス
class neuralNetwork():
    #-------------------
    # 1. 学習データの初期化
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×次元数のnumpy.ndarray）
    # hDim: 中間層のノード数（整数スカラー）
    # activeType: 活性化関数の種類（1:シグモイド関数、2:ReLU関数）
    def __init__(self,X,Y,hDim=10,activeType=1):
        # 学習データの設定
        self.xDim = X.shape[1]
        self.yDim = Y.shape[1]
        self.hDim = hDim
        
        self.activeType = activeType
        
        # パラメータの初期値の設定
        self.w1 = np.random.normal(size=[self.xDim,self.hDim])
        self.w2 = np.random.normal(size=[self.hDim,self.yDim])
        self.b1 = np.random.normal(size=[1,self.hDim])
        self.b2 = np.random.normal(size=[1,self.yDim])
        
        # log(0)を回避するための微小値
        self.smallV = 10e-8
        
        # Adamのパラメータ初期化
        # 入力と中間層の間
        self.grad1m = np.zeros([self.xDim+1,self.hDim])
        self.grad1V = np.zeros([self.xDim+1,self.hDim])
        
        # 中間と出力層の間
        self.grad2m = np.zeros([self.hDim+1,self.yDim])
        self.grad2V = np.zeros([self.hDim+1,self.yDim])
    #-------------------

    #-------------------
    # 2. 最急降下法を用いてモデルパラメータの更新
    # alpha: 学習率（実数スカラー）
    def update(self,X,Y,alpha=0.1):

        # 行列Xに「1」の要素を追加
        dNum = len(X)
        Z = np.append(X,np.ones([dNum,1]),axis=1)

        # 予測
        P,H,S = self.predict(X)

        # 予測の差の計算
        error = P - Y

        # 各階層のパラメータの準備
        V2 = np.concatenate([self.w2,self.b2],axis=0)
        V1 = np.concatenate([self.w1,self.b1],axis=0)

        # 入力層と中間層の間のパラメータの更新
        if self.activeType == 1:  # シグモイド関数
            term1 = np.matmul(error,self.w2.T)
            term2 = term1 * (1-H) * H
            grad1 = 1/dNum * np.matmul(Z.T,term2)

        elif self.activeType == 2: # ReLU関数
            Ms = np.ones_like(S)
            Ms[S<=0] = 0
            term1 = np.matmul(error,self.w2.T)
            grad1 = 1/dNum * np.matmul(Z.T,term1*Ms)

        V1 -= alpha * grad1

        # 中間層と出力層の間のパラメータの更新
        # 行列Xに「1」の要素を追加
        H = np.append(H,np.ones([dNum,1]),axis=1)
        grad2 = 1/dNum * np.matmul(H.T,error)
        V2 -= alpha * grad2
        
        # パラメータw1,b1,w2,b2の決定
        self.w1 = V1[:-1]
        self.w2 = V2[:-1]
        self.b1 = V1[[-1]]
        self.b2 = V2[[-1]]
    #-------------------
    
    #-------------------
    # 2.1 最急降下法を用いてモデルパラメータの更新
    # alpha: 学習率（実数スカラー）
    # rate：ノードの選択確率（実数スカラー）
    def updateDropout(self,X,Y,alpha=0.1,rate=1.0):

        # 行列Xに「1」の要素を追加
        dNum = len(X)
        Z = np.append(X,np.ones([dNum,1]),axis=1)
        
        # 予測
        P,H,S,Md = self.predictDropout(X,rate=rate)
        
        # 予測の差の計算
        error = P - Y

        # 各階層のパラメータの準備
        V2 = np.concatenate([self.w2,self.b2],axis=0)
        V1 = np.concatenate([self.w1,self.b1],axis=0)

        # 入力層と中間層の間のパラメータの更新
        if self.activeType == 1:  # シグモイド活性化関数
            term1 = np.matmul(error,self.w2.T)
            term2 = term1 * (1-H) * H
            grad1 = 1/dNum * np.matmul(Z.T,term2)

        elif self.activeType == 2: # ReLU活性化関数
            # マスクの作成
            Ms = np.ones_like(S)
            Ms[S<=0] = 0

            term1 = np.matmul(error,self.w2.T) * Md
            grad1 = 1/dNum * np.matmul(Z.T,term1*Ms)
            
        V1 -= alpha * grad1

        # 中間層と出力層の間のパラメータの更新
        # 行列Xに「1」の要素を追加
        H = np.append(H,np.ones([dNum,1]),axis=1)
        grad2 = 1/dNum * np.matmul(H.T,error)
        V2 -= alpha * grad2
    
        # パラメータw1,b1,w2,b2の決定
        self.w1 = V1[:-1]
        self.w2 = V2[:-1]
        self.b1 = V1[[-1]]
        self.b2 = V2[[-1]]
    #-------------------
    
    #-------------------
    # 2.2. 最急降下法を用いてモデルパラメータの更新
    # alpha: 学習率（スカラー）
    # rate：ドロップアウトの割合（実数スカラー）
    # beta：Adamの重み係数（実数スカラー）
    def updateAdam(self,X,Y,alpha=0.1,rate=1.0,beta=0.5):

        # 行列Xに「1」の要素を追加
        dNum = len(X)
        Z = np.append(X,np.ones([dNum,1]),axis=1)
        
        # 予測
        P,H,S,Md = self.predictDropout(X,rate=rate)
        
        # 予測の差の計算
        error = P - Y

        # 各階層のパラメータの準備
        V2 = np.concatenate([self.w2,self.b2],axis=0)
        V1 = np.concatenate([self.w1,self.b1],axis=0)

        # 入力層と中間層の間のパラメータの更新
        if self.activeType == 1:     # シグモイド活性化関数
            term1 = np.matmul(error,self.w2.T)
            term2 = term1 * (1-H) * H
            grad1 = 1/dNum * np.matmul(Z.T,term2)

        elif self.activeType == 2: # ReLU活性化関数
            # マスクの作成
            Ms = np.ones_like(S)
            Ms[S<=0] = 0
                    
            term1 = np.matmul(error,self.w2.T) * Md
            grad1 = 1/dNum * np.matmul(Z.T,term1*Ms)

        #-------------------
        # Adamによるパラメータの更新
        self.grad1m = beta * self.grad1m + (1-beta) * grad1
        self.grad1V = beta * self.grad1V + (1-beta) * grad1**2
        mhat = self.grad1m/(1-beta)
        Vhat = self.grad1V/(1-beta)
        
        V1 -= alpha * mhat/(np.sqrt(Vhat)+self.smallV)
        #-------------------
        
        # 中間層と出力層の間のパラメータの更新
        # 行列Xに「1」の要素を追加
        H = np.append(H,np.ones([dNum,1]),axis=1)
        grad2 = 1/dNum * np.matmul(H.T,error)
        
        #-------------------
        # Adamによるパラメータの更新
        self.grad2m = beta * self.grad2m + (1-beta) * grad2
        self.grad2V = beta * self.grad2V + (1-beta) * grad2**2
        mhat = self.grad2m/(1-beta)
        Vhat = self.grad2V/(1-beta)
        
        V2 -= alpha * mhat/(np.sqrt(Vhat)+self.smallV)
        #-------------------

        # パラメータw1,b1,w2,b2の決定
        self.w1 = V1[:-1]
        self.w2 = V2[:-1]
        self.b1 = V1[[-1]]
        self.b2 = V2[[-1]]
    #-------------------

    #-------------------
    # 3. 予測
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    def predict(self,x):
        s = np.matmul(x,self.w1) + self.b1
        H = self.activation(s)
        f_x = np.matmul(H,self.w2) + self.b2

        return 1/(1+np.exp(-f_x)),H,s
    #-------------------
    
    #-------------------
    # 3.1. 予測
    # X：入力データ（データ数×次元数のnumpy.ndarray）
    # rate：ノードの選択確率（実数スカラー）
    def predictDropout(self,x,rate):
        # ドロップアウト用のマスクの作成
        M = np.random.binomial(1,rate,size=[len(x),self.hDim])
    
        s = np.matmul(x,self.w1) + self.b1
        H = self.activation(s) * M
        f_x = np.matmul(H,self.w2) + self.b2

        return 1/(1+np.exp(-f_x)),H,s,M
    #-------------------

    #-------------------
    # 4. 活性化関数
    # s: 中間データ（データ数×次元数のnumpy.ndarray）
    def activation(self,s):
    
        if self.activeType == 1:  # シグモイド関数
            h = 1/(1+np.exp(-s))
        
        elif self.activeType == 2:  # ReLU関数
            h = s
            h[h<=0] = 0

        return h
    #-------------------

    #-------------------
    # 5. 交差エントロピー損失
    # X: 入力データ（次元数×データ数のnumpy.ndarray）
    # Y: 出力データ（データ数×次元数のnumpy.ndarray）
    def CE(self,X,Y):
        P,_,_ = self.predict(X)

        if self.yDim == 1:
            loss = -np.mean(Y*np.log(P+self.smallV)+(1-Y)*np.log(1-P+self.smallV))
        else:
            loss = -np.mean(Y*np.log(P+self.smallV))
            
        return loss
    #-------------------

    #-------------------
    # 6. 正解率の計算
    # X:入力データ（データ数×次元数のnumpy.ndarray）
    # Y:出力データ（データ数×次元数のnumpy.ndarray）
    # thre: 閾値（スカラー）
    def accuracy(self,X,Y,thre=0.5):
        P,_,_= self.predict(X)
        
        # 予測値Pをラベルに変換
        if self.yDim == 1:
            P[P>thre] = 1
            P[P<=thre] = 0
        else:
            P = np.argmax(P,axis=1)
            Y = np.argmax(Y,axis=1)
        
        # 正解率
        accuracy = np.mean(Y==P)
        return accuracy
    #-------------------

    #-------------------
    # 6.1 適合率、再現率、F1スコアの計算
    # X:入力データ（データ数×次元数のnumpy.ndarray）
    # Y:出力データ（データ数×次元数のnumpy.ndarray）
    # thre: 閾値（スカラー）
    def eval(self,X,Y,thre=0.5):
        P,_,_= self.predict(X)
        
        # 予測値Pをラベルに変換
        if self.yDim == 1:
            P[P>thre] = 1
            P[P<=thre] = 0
        else:
            P = np.argmax(P,axis=1)
            Y = np.argmax(Y,axis=1)

        # 適合率
        precision = np.array([np.sum(Y[P==c]==c)/np.sum(P==c) for c in np.unique(Y)])
        
        # 再現率
        recall = np.array([np.sum(P[Y==c]==c)/np.sum(Y==c) for c in np.unique(Y)])
        
        # F1スコア
        f1 = (2*precision*recall)/(precision+recall)
        
        return precision,recall,f1
    #-------------------
    
    #-------------------
    # 6.2. 混同行列の計算
    # X:入力データ（データ数×次元数のnumpy.ndarray）
    # Y:出力データ（データ数×次元数のnumpy.ndarray）
    # thre: 閾値（スカラー）
    def confusionMatrix(self,X,Y,thre=0.5):
        P,_,_= self.predict(X)

        # 予測値Pをラベルに変換
        if self.yDim == 1:
            P[P>thre] = 1
            P[P<=thre] = 0
        else:
            P = np.argmax(P,axis=1)
            Y = np.argmax(Y,axis=1)
            
        # 混同行列
        cm = np.array([[np.sum(P[Y==c]==p) for p in np.unique(Y)] for c in np.unique(Y)])
        
        return cm
    #-------------------

    #------------------- 
    # 7. 真値と予測値のプロット（入力ベクトルが1次元の場合）
    # X:入力データ（次元数×データ数のnumpy.ndarray）
    # Y:出力データ（データ数×次元数のnumpy.ndarray）
    # xLabel:x軸のラベル（文字列）
    # yLabel:y軸のラベル（文字列）
    # fName：画像の保存先（文字列）
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
    # 8. 真値と予測値のプロット（入力ベクトルが2次元の場合）
    # X:入力データ（データ数×次元数のnumpy.ndarray）
    # Y:出力データ（データ数×次元数のnumpy.ndarray）
    # xLabel:x軸のラベル（文字列）
    # yLabel:y軸のラベル（文字列）
    # title:タイトル（文字列）
    # fName：画像の保存先（文字列）
    def plotModel2D(self,X=[],Y=[],xLabel="",yLabel="",title="",fName=""):
        #fig = plt.figure(figsize=(6,4),dpi=100)
        plt.close()
        
        # 真値のプロット（クラスごとにマーカーを変更）
        plt.plot(X[Y[:,0]==0,0],X[Y[:,0]==0,1],'cx',markerSize=14,label="ラベル0")
        plt.plot(X[Y[:,0]==1,0],X[Y[:,0]==1,1],'m.',markerSize=14,label="ラベル1")

        # 予測値のメッシュの計算
        X1,X2 = plt.meshgrid(plt.linspace(np.min(X[:,0]),np.max(X[:,0]),50),plt.linspace(np.min(X[:,1]),np.max(X[:,1]),50))
        Xmesh = np.hstack([np.reshape(X1,[-1,1]),np.reshape(X2,[-1,1])])
        Pmesh,_,_ = self.predict(Xmesh)
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
    # trEval:学習の損失
    # teEval:評価の損失
    # yLabel:y軸のラベル（文字列）
    # fName:画像の保存先（文字列）
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
