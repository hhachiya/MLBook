# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# クラス
class PCA:
    #-------------------
    # 1. 主成分分析の各種初期化
    # X: 学習データ（データ数×次元数のnumpy.ndarray）
    def __init__(self,X):

        # データの中心化
        self.mean = np.mean(X,axis=0)
        self.X = X - self.mean
    #-------------------

    #-------------------
    # 2. 主成分分析を用いた次元削減
    # lowerDim: 低次元空間の次元数（整数スカラー）
    def reduceDim(self,lowerDim):
        self.lowerDim = lowerDim

        # 分散共分散行列
        cov = np.cov(self.X.T,bias=1)
        
        # 固有値問題
        L,V = np.linalg.eig(cov)
        
        # 固有値と固有ベクトルの固有値の降順でソート
        inds = np.argsort(L)[::-1]
        self.L = L[inds]
        self.W = V[:,inds]
        
        # 主成分得点の計算
        self.F = np.matmul(self.X,self.W[:,:lowerDim])
    #-------------------
    
    #-------------------
    # 3. 次元削減後のデータ（主成分得点）のプロット
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # fName: 画像の保存先（文字列）
    def plotResult(self,xLabel="$z_1$",yLabel="$z_2$",fName=""):
        import matplotlib.pylab as plt
        
        # 主成分得点のプロット
        plt.plot(self.F[:,0],self.F[:,1],'k.',markerSize=14)
        
        # 各軸の範囲とラベルの設定
        plt.xlabel(xLabel,fontSize=14)
        plt.ylabel(yLabel,fontSize=14)

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------

    #-------------------
    # 4. 真値と主成分軸（平面）のプロット
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # zLabel: z軸のラベル（文字列）
    # nGrids: 格子の数（整数のスカラー）
    # fName: 画像の保存先（文字列）
    def plotModel3D(self,xLabel="",yLabel="",zLabel="",nGrids=10,fName=""):
        
        # 平面の法線ベクトルの計算
        normal = np.cross(self.W[:,0],self.W[:,1])
        
        # XとY軸のメッシュ計算
        Xmin = np.min(self.X,axis=0)
        Xmax = np.max(self.X,axis=0)
        Xrange = np.arange(Xmin[0],Xmax[0],int((Xmax[0]-Xmin[0])/nGrids))
        Yrange = np.arange(Xmin[1],Xmax[1],int((Xmax[1]-Xmin[1])/nGrids))
        XX,YY = np.meshgrid(Xrange,Yrange)
        
        # 各グリッドのz成分の計算
        ZZ = -(normal[0]*XX + normal[1]*YY)/normal[2]
        
        # 3Dプロット
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = Axes3D(fig)
        
        # 学習データのプロット
        ax.plot(self.X[:,0],self.X[:,1],self.X[:,2],color="k",marker=".",linestyle='None',markerSize=14)
        
        # 平面のプロット
        ax.plot_wireframe(XX,YY,ZZ,color="c")

        # 各軸の範囲とラベルの設定
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel(xLabel,fontSize=14)
        ax.set_ylabel(yLabel,fontSize=14)
        ax.set_zlabel(zLabel,fontSize=14)

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------
    
    #-------------------
    # 5. 寄与率と累積寄与率の計算
    def compContRatio(self):

        # 寄与率の計算
        contRatio = self.L/np.sum(self.L) * 100

        # 累積寄与率の計算
        cumContRatio = [np.sum(contRatio[:i+1]) for i in range(len(self.L))]
        
        return contRatio,cumContRatio
    #-------------------

    #-------------------
    # 6. 主成分負荷量の計算
    def compLoading(self):
        # 特徴量Xと主成分得点Fの各ペア間の相関係数
        Z = np.concatenate([self.X,self.F],axis=1)
        PCL = np.corrcoef(Z.T,bias=1)[:self.X.shape[1],-self.F.shape[1]:]
        return PCL
    #-------------------
    