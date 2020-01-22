# -*- coding: utf-8 -*-
import numpy as np
import cvxopt
import cvxopt.solvers
import pdb
import matplotlib.pylab as plt


####################
# クラス
class SVM():
  #-------------------
  # 学習データの初期化
  # X: 入力データ（データ数×次元数のnumpy.array）
  # Y: 出力データ（データ数×1のnumpy.array）
  # spptThre: サポートベクトルの閾値（スカラー、デフォルトでは0.1）
  # kernelFunc: kernelFuncクラスのインスタンス    
  def __init__(self, X, Y, spptThre=0.1, kernelFunc=None):

    # カーネルの設定
    self.kernelFunc = kernelFunc
    
    # 学習データの設定
    self.X = X
    self.Y = Y
    self.dNum = self.X.shape[0]
    self.xDim = self.X.shape[1]
    self.spptThre = spptThre
  #-------------------

  #-------------------
  # SVMのモデルパラメータを最適化
  def train(self):
    X = self.kernelFunc.createMatrix(self.X,self.X)
    
    # 行列Pの作成
    P = np.matmul(self.Y,self.Y.T) * X
    P = cvxopt.matrix(P)
    
    # q,G,h,A,bを作成
    q = cvxopt.matrix(-np.ones(self.dNum))           # 全ての要素が-1の列ベクトル（Nx1）
    G = cvxopt.matrix(np.diag(-np.ones(self.dNum)))  # 対角成分が-1行列（NxN）
    h = cvxopt.matrix(np.zeros(self.dNum))           # 全ての要素が0の列ベクトル（Nx1）
    A = cvxopt.matrix(self.Y.T)                      # 各要素がカテゴリラベルの行ベクトル（1xN）
    b = cvxopt.matrix(0.0)                           # スカラー0.0

    # 凸二次計画法
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    self.lamb = np.array(sol['x'])                   # 'x'がlambdaに対応する
    
    # サポートベクトルのインデックス
    self.spptInds = np.where(self.lamb > self.spptThre)[0]

    # wとbの計算
    self.w = np.matmul((self.lamb*self.Y).T,X).T
    self.b = np.mean(self.Y[self.spptInds] - np.matmul(X[self.spptInds,:],self.w))
  #-------------------

  #-------------------
  # ソフトマージンSVMのモデルパラメータを最適化
  # C: 誤差の重要度ハイパーパラメータ（スカラー、デフォルトでは0.1）
  def trainSoft(self,C=0.1):
    X = self.kernelFunc.createMatrix(self.X,self.X)
    
    # 行列Pの作成
    P = np.matmul(self.Y,self.Y.T) * X
    P = cvxopt.matrix(P)
    
    # q,G,h,A,bを作成
    q = cvxopt.matrix(-np.ones(self.dNum)) # 全ての要素が-1の列ベクトル（Nx1）
    G1 = np.diag(-np.ones(self.dNum))      # 対角成分が-1行列（NxN）
    G2 = np.diag(np.ones(self.dNum))       # 対角成分が1行列（NxN）
    G = cvxopt.matrix(np.vstack([G1,G2]))  # 2NxNの行列
    
    h1 = np.zeros([self.dNum,1])           # 全ての要素が0の列ベクトル（Nx1）
    h2 = C*np.ones([self.dNum,1])          # 全ての要素がCの列ベクトル（Nx1）
    h = cvxopt.matrix(np.vstack([h1,h2]))  # 2Nx1のベクトル
    
    A = cvxopt.matrix(self.Y.T)            # 各要素がカテゴリラベルの行ベクトル（1xN）
    b = cvxopt.matrix(0.0)                 # スカラー0.0

    # 凸二次計画法
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    self.lamb = np.array(sol['x'])         # 'x'がlambdaに対応する
    
    # サポートベクトルのインデックス
    self.spptInds = np.where(self.lamb > self.spptThre)[0]
    
    # wとbの計算
    self.w = np.matmul((self.lamb*self.Y).T,X).T
    self.b = np.mean(self.Y[self.spptInds] - np.matmul(X[self.spptInds,:],self.w))
  #-------------------

  #-------------------
  # 予測
  # X: 入力データ（データ数×次元数のnumpy.array）
  def predict(self,x):
    x = self.kernelFunc.createMatrix(x,self.X)
    y = np.matmul(x,self.w)+self.b
    return np.sign(y),y
  #-------------------
  
  #-------------------
  # 正解率の計算
  # X: 入力データ（データ数×次元数のnumpy.array）
  # Y: 出力データ（データ数×1のnumpy.array）  
  def accuracy(self,x,y):
    pred, _ = self.predict(x)
    return np.sum(pred==y)/len(x)
  #-------------------

  #-------------------
  # 学習データと、獲得した分類境界f(x)のプロット
  # xlabel：x軸のラベル（文字列）
  # ylabel：y軸のラベル（文字列）   
  # isLinePlot：f(x)の直線をプロットするかしないか（boolean)  
  # fName：画像の保存先（文字列）
  # title：タイトル（文字列）
  # isSppt：サポートベクトルのプロットするか否か（boolean）
  def plotModel2D(self,xlabel="",ylabel="",X=[],Y=[],isLinePlot=False,fName="../figures/result.png",title="",isSppt=False):
    #fig = plt.figure(figsize=(6,4),dpi=100)
    plt.close()
    
    #-------------------
    # 学習データのプロット
    plt.plot(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],'cx',markerSize=14,label="カテゴリ-1")
    plt.plot(X[Y[:,0]==1,0],X[Y[:,0]==1,1],'m.',markerSize=14,label="カテゴリ+1")

    if isSppt:
      # サポートベクトルのプロット
      plt.plot(X[self.spptInds,0], X[self.spptInds,1], 'o', color='none', markeredgecolor='r', markersize=18, markeredgewidth=3, label="サポートベクトル")
    #-------------------

    #-------------------
    # 関数f(x)のプロット
    X1, X2 = plt.meshgrid(plt.linspace(np.min(X[:,0]),np.max(X[:,0]),50), plt.linspace(np.min(X[:,1]),np.max(X[:,1]),50))
    X = np.hstack([np.reshape(X1,[-1,1]),np.reshape(X2,[-1,1])])
    _, Y = self.predict(X)
    Y = np.reshape(Y,X1.shape)

    # contourプロット
    CS = plt.contourf(X1,X2,Y,linewidths=2,cmap="bwr",alpha=0.3,vmin=-5,vmax=5)

    # contourのカラーバー
    CB = plt.colorbar(CS)
    CB.ax.tick_params(labelsize=14)
    
    if isLinePlot:   # 直線
      x1 = np.arange(np.min(X[:,0]),np.max(X[:,0]),(np.max(X[:,0]) - np.min(X[:,0]))/100)
      x2 = -(x1*self.w[0]+self.b)/self.w[1]
      plt.plot(x1,x2,'r-',label="f(x)")
    #-------------------

    plt.xlim([np.min(X[:,0]),np.max(X[:,0])])
    plt.ylim([np.min(X[:,1]),np.max(X[:,1])])

    plt.title(title,fontSize=14)
    plt.xlabel(xlabel,fontSize=14)
    plt.ylabel(ylabel,fontSize=14)
    plt.legend()
    plt.savefig(fName)
    #-------------------    
####################

