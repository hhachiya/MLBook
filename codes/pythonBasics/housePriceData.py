# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

class housePriceData:
    #-------------------
    # 1. コンストラクタ
    # path: ファイルのパス（文字列）
    def __init__(self,path):
        self.data = pd.read_csv(path)  # dataframeの読み込み
    #-------------------
    
    #-------------------
    # 2. 建物の等級（MSSubClass）を限定した散布図（横軸GrLivArea. 縦軸SalePrice）をプロットするメソッド
    # titles: タイトル（グラフ数のリスト）
    # levels: 建物の等級（グラフ数のリスト）
    def plotScatter(self,titles,levels):
    
        # figureの初期化
        fig = plt.figure()

        # 横軸と縦軸の範囲計算
        xrange = [np.min(self.data['GrLivArea'].values),np.max(self.data['GrLivArea'].values)]
        yrange = [np.min(self.data['SalePrice'].values),np.max(self.data['SalePrice'].values)]
        
        # 列数の計算
        ncol = int(len(titles)/2)
        
        # 各グラフのプロット
        for ind in np.arange(len(titles)):
         
            # グラフの位置を設定
            ax = fig.add_subplot(2,ncol,ind+1)
            
            # タイトルの設定
            ax.set_title(titles[ind])
            
            # 散布図のプロット
            ax.plot(self.data[self.data['MSSubClass']==levels[ind]]['GrLivArea'].values,
                    self.data[self.data['MSSubClass']==levels[ind]]['SalePrice'].values,'.')
            
            # 各軸の範囲とラベルの設定
            ax.set_xlim([xrange[0],xrange[1]])
            ax.set_ylim([yrange[0],yrange[1]])
            ax.set_xlabel('GrLivArea',fontSize=14)
            ax.set_ylabel('SalePrice',fontSize=14)

        plt.tight_layout() # グラフ間に隙間をあける
        plt.show() # グラフの表示
    #-------------------