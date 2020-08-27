# -*- coding: utf-8 -*-
import numpy as np
import cvxopt
import matplotlib.pylab as plt

# クラス
class SVM():
    #-------------------
    # 1. 学習データの初期化
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def __init__(self,X,Y):

        # 学習データの設定
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]  # 学習データ数
        self.xDim = X.shape[1]  # 入力の次元数
        
        # サポートベクトルの閾値設定
        self.spptThre = 0.1
    #-------------------

    #-------------------
    # 2. SVMのモデルパラメータを最適化
    def train(self):

        # 行列Pの作成
        P = np.matmul(self.Y,self.Y.T) * np.matmul(self.X,self.X.T)
        P = cvxopt.matrix(P)

        # q,G,h,A,bを作成
        q = cvxopt.matrix(-np.ones(self.dNum))
        G = cvxopt.matrix(np.diag(-np.ones(self.dNum)))
        h = cvxopt.matrix(np.zeros(self.dNum))
        A = cvxopt.matrix(self.Y.astype(float).T)
        b = cvxopt.matrix(0.0)

        # 凸二次計画法
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        self.lamb = np.array(sol['x'])
        # 'x'がlambdaに対応する
        
        # サポートベクトルのインデックス
        self.spptInds = np.where(self.lamb>self.spptThre)[0]

        # wとbの計算
        self.w = np.matmul((self.lamb*self.Y).T,self.X).T
        self.b = np.mean(self.Y[self.spptInds]-np.matmul(self.X[self.spptInds,:],self.w))
    #-------------------

    #-------------------
    # 2.5 ソフトマージンSVMのモデルパラメータを最適化
    # C: 誤差の重要度ハイパーパラメータ（スカラー、デフォルトでは0.1）
    def trainSoft(self,C=0.1):
        X = self.X
        
        # 行列Pの作成
        P = np.matmul(self.Y,self.Y.T) * np.matmul(X,X.T)
        P = cvxopt.matrix(P)
        
        # q,G,h,A,bを作成
        q = cvxopt.matrix(-np.ones(self.dNum))
        G1 = np.diag(-np.ones(self.dNum))
        G2 = np.diag(np.ones(self.dNum))
        G = cvxopt.matrix(np.concatenate([G1,G2],axis=0))
        h1 = np.zeros([self.dNum,1])
        h2 = C * np.ones([self.dNum,1])
        h = cvxopt.matrix(np.concatenate([h1,h2],axis=0))
        A = cvxopt.matrix(self.Y.astype(float).T)
        b = cvxopt.matrix(0.0)

        # 凸二次計画法
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        self.lamb = np.array(sol['x'])
        # 'x'がlambdaに対応する
        
        # サポートベクトルのインデックス
        self.spptInds = np.where(self.lamb>self.spptThre)[0]
        
        # wとbの計算
        self.w = np.matmul((self.lamb*self.Y).T,X).T
        self.b = np.mean(self.Y[self.spptInds]-np.matmul(X[self.spptInds,:],self.w))
    #-------------------
    
    #-------------------
    # 3. 予測
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    def predict(self,x):
        y = np.matmul(x,self.w) + self.b
        return np.sign(y),y
    #-------------------
    
    #-------------------
    # 4. 正解率の計算
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def accuracy(self,X,Y):
        predict,_ = self.predict(X)
        return np.sum(predict==Y)/len(X)
    #-------------------

    #-------------------
    # 5. 真値と予測値のプロット（入力ベクトルが2次元の場合）
    # X:入力データ（データ数×次元数のnumpy.ndarray）
    # Y:出力データ（データ数×１のnumpy.ndarray）
    # spptInds:サポートベクトルのインデックス（インデックス数のnumpy.ndarray)
    # xLabel:x軸のラベル（文字列）
    # yLabel:y軸のラベル（文字列）
    # title:タイトル（文字列）
    # fName：画像の保存先（文字列）
    # isLinePlot：分類境界の直線をプロットするかしないか（boolean)
    def plotModel2D(self,X=[],Y=[],spptInds=[],xLabel="",yLabel="",title="",fName="",isLinePlot=False):
        plt.close()
        
        # 真値のプロット（クラスごとにマーカーを変更）
        plt.plot(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],'cx',markerSize=14,label="カテゴリ-1")
        plt.plot(X[Y[:,0]== 1,0],X[Y[:,0]== 1,1],'m.',markerSize=14,label="カテゴリ+1")

        # 予測値のメッシュの計算
        X1,X2 = plt.meshgrid(plt.linspace(np.min(X[:,0]),np.max(X[:,0]),50),plt.linspace(np.min(X[:,1]),np.max(X[:,1]),50))
        Xmesh = np.hstack([np.reshape(X1,[-1,1]),np.reshape(X2,[-1,1])])
        _,Ymesh = self.predict(Xmesh)
        Ymesh = np.reshape(Ymesh,X1.shape)

        # contourプロット
        CS = plt.contourf(X1,X2,Ymesh,linewidths=2,cmap="bwr",alpha=0.3,vmin=-5,vmax=5)

        # カラーバー
        CB = plt.colorbar(CS)
        CB.ax.tick_params(labelsize=14)
        
        # サポートベクトルのプロット
        if len(spptInds):
            plt.plot(X[spptInds,0],X[spptInds,1],'o',color='none',markeredgecolor='r',markersize=18,markeredgewidth=3,label="サポートベクトル")

        # 直線のプロット
        if isLinePlot:
            x1 = np.arange(np.min(X[:,0]),np.max(X[:,0]),(np.max(X[:,0]) - np.min(X[:,0]))/100)
            x2 = -(x1*self.w[0]+self.b)/self.w[1]
            plt.plot(x1,x2,'r-',label="f(x)")

        # 各軸の範囲、タイトルおよびラベルの設定
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
