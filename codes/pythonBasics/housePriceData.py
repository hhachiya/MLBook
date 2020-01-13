# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

class housePriceData:
  #-------------------
  # 1. コンストラクタ
  # path: ファイルのパス（文字列）
  def __init__(self, path):
    self.data = pd.read_csv(datapath)  # dataframeの読み込み
  #-------------------
  
  #-------------------
  # 2. 建物等級（MSSubClass）を限定した散布図をプロットするメソッド
  # nrow: 行数（整数スカラー）
  # xattri: x軸のデータの種類（文字列）
  # yattri: y軸のデータの種類（文字列）
  # titles: タイトル（グラフ数のリスト）
  # classes: 建物等級（グラフ数のリスト）
  def plotScatterMSSubClass(self, nrow, xattri, yattri, titles, classes):
  
    # figureの初期化
    fig = plt.figure()

    # x軸とy軸の範囲計算
    xrange = [np.min(self.data[xattri].values),np.max(self.data[xattri].values)]
    yrange = [np.min(self.data[yattri].values),np.max(self.data[yattri].values)]
    
    # 列数の計算
    ncol = int(len(titles)/nrow)
    
    # 各グラフのプロット
    for ind in range(len(titles)):
     
      # グラフの位置を設定
      ax=fig.add_subplot(nrow,ncol,ind+1)
      
      # タイトルの設定
      ax.set_title(titles[ind])
      
      # 散布図のプロット
      ax.plot(self.data[self.data['MSSubClass']==classes[ind]][xattri].values,
          self.data[self.data['MSSubClass']==classes[ind]][yattri].values,'.')
      
      # 各軸の範囲とラベルの設定
      ax.set_xlim([xrange[0], xrange[1]])
      ax.set_ylim([yrange[0], yrange[1]])
      ax.set_xlabel(xattri)
      ax.set_ylabel(yattri)

    plt.tight_layout()  # グラフ間に隙間をあける
    plt.show() # グラフの表示
  #-------------------