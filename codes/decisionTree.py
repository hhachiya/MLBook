import pandas as pd
import numpy as np
import data
import pdb

class decisionTree:

    # スペースの記号
    space="　　　"
    lineFlag = False

    #-------------------
    # 1. 学習データの初期化
    # X: 入力データ（データ数×カラム数のdataframe）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # version: 決定木のバージョン番号（1: ID3,2: CART）
    def __init__(self,X,Y,version=2):
        self.X = X
        self.Y = Y
        self.version = version
        
        # 情報量を計算する関数infoFuncをversionに基づき設定
        if self.version == 1: # ID3（情報エントロピー）
            self.infoFunc = self.compEntropy
        elif self.version == 2: # CART（ジニ不純度）
            self.infoFunc = self.compGini
    #-------------------
    
    #-------------------
    # 2.1. 情報エントロピーの計算
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def compEntropy(self,Y):
        probs = [np.sum(Y==y)/len(Y) for y in np.unique(Y)]

        return -np.sum(probs*np.log2(probs))
    #-------------------

    #-------------------
    # 2.2. ジニ不純度の計算
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def compGini(self,Y):
        probs = [np.sum(Y==y)/len(Y) for y in np.unique(Y)]

        return 1 - np.sum(np.square(probs))
    #-------------------
    
    #-------------------
    # 3. 説明変数の選択
    # X: 入力データ（データ数×説明変数の数のdataframe）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def selectX(self,X,Y):

        # 出力Yの情報エントロピーまたはジニ不純度の計算
        allInfo = self.infoFunc(Y)

        # 各説明変数の平均情報量および利得の記録
        colInfos = []
        gains = []

        # 説明変数のループ
        for col in X.columns:
            
            # 説明変数を限定した平均情報エントロピーまたはジニ不純度の計算
            colInfo = np.sum([np.sum(X[col]==value)/len(X)*
                self.infoFunc(Y[X[col]==value]) for value in np.unique(X[col])])
            colInfos.append(colInfo)

            # 利得の計算およgainsに記録
            gains.append(allInfo-colInfo)

        # 最大利得を返す
        return np.argmax(gains),allInfo
    #-------------------
    
    #-------------------
    # 4. 説明変数の削除
    # X: 入力データ（データ数×カラム数のdataframe）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # col: 削除する説明変数の名前
    # value: 説明変数の値
    def delCol(self,X,Y,col,value):
        # 説明変数colの削除
        subX = X[X[col]==value]
        subX = subX.drop(col,axis=1)

        # 目的変数から値を削除
        subY = Y[X[col]==value]

        return subX,subY
    #-------------------
    
    #-------------------
    # 5. 決定木の作成
    # X: 入力データ（データ数×カラム数のdataframe）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # layer: 階層番号（整数スカラー、デフォルトでは0）
    def train(self,X=[],Y=[],layer=0):
        if not len(X): X = self.X
        if not len(Y): Y = self.Y

        # 葉ノードの標準出力
        if self.infoFunc(Y) == 0:
            print(f" --> {Y[0][0]}")
            return Y[0][0]
        else:
            print("\n",end="")

        # 説明変数の選択
        colInd,allInfo = self.selectX(X,Y)

        # 説明変数名の取得
        col = X.columns[colInd]

        # 説明変数colの値ごとに枝を分岐
        for value in np.unique(X[col]):
        
            # 説明変数colの削除
            subX,subY = self.delCol(X,Y,col,value)

            #-----------
            # 分岐ノードの標準出力
            if self.lineFlag:
                print(f"{self.space*(layer-1)}｜")
            self.lineFlag = True

            if layer > 0:
                print(f"{self.space*(layer-1)}＋― ",end="")

            print(f"{col} ({round(allInfo,2)}) = '{value}' ({round(self.infoFunc(subY),2)})",end="")
            #-----------

            # 分岐先の枝で決定木を作成
            self.train(subX,subY,layer+1)
    #-------------------
