# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import re
import gzip
import pdb
from sklearn import datasets

####################
# 教師なしデータ用のクラス
class unsupervised:
    #-------------------
    # パスの設定
    def __init__(self):
        self.path = "../data"
    #-------------------
    #-------------------
    # データの作成
    # dataType: データの種類（整数スカラー）
    def makeData(self,dataType=1):
        self.dataType = dataType
        
        # iris
        if dataType == 1:
            self.labels = ["がく長","がく幅","花びら長","花びら幅"]
        
            iris = datasets.load_iris()
            self.X = iris.data
        
        # 物件価格）説明変数:居住面積,車庫面積,全部屋数
        elif dataType == 2:
            self.labels = ["GrLivArea","GarageArea","TotRmsAbvGrd"]
            data = pd.read_csv(os.path.join(self.path,"house-prices-advanced-regression-techniques/train.csv"))
            self.X = data[(data['MSSubClass']==30)|(data['MSSubClass']==60)][[self.labels[0],self.labels[1],self.labels[2]]].values
####################

####################
# 回帰デモデータ用のクラス
class regression:
    #-------------------
    # パスの設定
    def __init__(self):
        self.path = "../data"
    #-------------------
    
    #-------------------
    # データの作成
    # dataType: データの種類（整数スカラー）
    def makeData(self,dataType=1):
        self.dataType = dataType
    
        # 物件価格）説明変数:居住面積
        if dataType == 1:
            data = pd.read_csv(os.path.join(self.path,"house-prices-advanced-regression-techniques/train.csv"))
            self.X = data[data['MSSubClass']==60][['GrLivArea']].values
            self.Y = data[data['MSSubClass']==60][['SalePrice']].values
            self.xLabel = "居住面積x[平方フィート]"
            self.yLabel = "物件価格y[ドル]" 
            
        # 物件価格）説明変数:居住面積,車庫面積,プール面積,ベッド部屋数,全部屋数
        elif dataType == 2:
            data = pd.read_csv(os.path.join(self.path,"house-prices-advanced-regression-techniques/train.csv"))
            self.X = data[data['MSSubClass']==60][['GrLivArea','GarageArea','PoolArea','BedroomAbvGr','TotRmsAbvGrd']].values
            self.Y = data[data['MSSubClass']==60][['SalePrice']].values
            self.xLabel = ""
            self.yLabel = "物件価格y[ドル]"
            
        # 物件価格）説明変数:居住面積、物件価格に外れ値を追加
        elif dataType == 3:
            data = pd.read_csv(os.path.join(self.path,"house-prices-advanced-regression-techniques/train.csv"))
            self.X = data[data['MSSubClass']==60][['GrLivArea']].values
            self.Y = data[data['MSSubClass']==60][['SalePrice']].values
            self.Y[self.Y>700000] -= 700000 # 外れ値
            self.xLabel = "居住面積x[平方フィート]"
            self.yLabel = "物件価格y[ドル]"
    #-------------------
####################

####################
# 分類デモデータ用のクラス
class classification:
    #-------------------
    # パス、ラベルの設定
    def __init__(self,negLabel=-1,posLabel=1):
        self.path = "../data"
        self.negLabel = negLabel
        self.posLabel = posLabel
    #-------------------
    
    #-------------------
    # データの作成
    # dataType: データの種類（整数スカラー）
    def makeData(self,dataType=1):
        self.dataType = dataType
    
        # 建物等級）説明変数:GrLivArea
        if dataType == 1:
            data = pd.read_csv(os.path.join(self.path,"house-prices-advanced-regression-techniques/train.csv"))
            self.X = data[(data['MSSubClass']==30) |(data['MSSubClass']==60)][['GrLivArea']].values
            self.Y = data[(data['MSSubClass']==30) |(data['MSSubClass']==60)][['MSSubClass']].values
            self.Y[self.Y==30] = self.negLabel
            self.Y[self.Y==60] = self.posLabel
            self.xLabel = "居住面積x[平方フィート]"
            self.yLabel = "建物等級ラベルy"

        # 建物等級）説明変数:GrLivArea,GarageArea
        elif dataType == 2:
            data = pd.read_csv(os.path.join(self.path,"house-prices-advanced-regression-techniques/train.csv"))
            self.X = data[(data['MSSubClass']==30) |(data['MSSubClass']==60)][['GrLivArea','GarageArea']].values
            self.Y = data[(data['MSSubClass']==30) |(data['MSSubClass']==60)][['MSSubClass']].values
            self.Y[self.Y==30] = self.negLabel
            self.Y[self.Y==60] = self.posLabel
            self.xLabel = "居住面積x[平方フィート]"
            self.yLabel = "車庫面積x[平方フィート]"

        # トイデータ） 線形分離可能な2つのガウス分布に従場合
        elif dataType == 3:
            dNum = 120
            np.random.seed(1)
            
            cov = [[1,-0.6],[-0.6,1]]
            X = np.random.multivariate_normal([1,2],cov,int(dNum/2))
            X = np.concatenate([X,np.random.multivariate_normal([-2,-1],cov,int(dNum/2))],axis=0)
            Y = np.concatenate([self.negLabel*np.ones([int(dNum/2),1]),self.posLabel*np.ones([int(dNum/2),1])],axis=0)
            randInds = np.random.permutation(dNum)
            self.X = X[randInds]
            self.Y = Y[randInds]
            self.xLabel = "$x_1$"
            self.yLabel = "$x_2$"

        # トイデータ）分類境界がアルファベッドのCの形をしている場合
        elif dataType == 4:
            dNum = 120
            np.random.seed(1)
            
            cov1 = [[1,-0.8],[-0.8,1]]
            cov2 = [[1,0.8],[0.8,1]]
                
            X = np.random.multivariate_normal([0.5,1],cov1,int(dNum/2))
            X = np.concatenate([X,np.random.multivariate_normal([-1,-1],cov1,int(dNum/4))],axis=0)
            X = np.concatenate([X,np.random.multivariate_normal([-1,4],cov2,int(dNum/4))],axis=0)
            Y = np.concatenate([self.negLabel*np.ones([int(dNum/2),1]),self.posLabel*np.ones([int(dNum/2),1])],axis=0)
            randInds = np.random.permutation(dNum)
            self.X = X[randInds]
            self.Y = Y[randInds]
                
            self.xLabel = "$x_1$"
            self.yLabel = "$x_2$"

        # トイデータ）複数の島がある場合
        elif dataType == 5:
            dNum = 120
            np.random.seed(1)
            
            cov = [[1,-0.8],[-0.8,1]]
            X = np.random.multivariate_normal([0.5,1],cov,int(dNum/2))
            X = np.concatenate([X,np.random.multivariate_normal([-1,-1],cov,int(dNum/4))],axis=0)
            X = np.concatenate([X,np.random.multivariate_normal([2,2],cov,int(dNum/4))],axis=0)
            Y = np.concatenate([self.negLabel*np.ones([int(dNum/2),1]),self.posLabel*np.ones([int(dNum/2),1])],axis=0)
            
            # データのインデックスをシャッフル
            randInds = np.random.permutation(dNum)
            self.X = X[randInds]
            self.Y = Y[randInds]
            self.xLabel = "$x_1$"
            self.yLabel = "$x_2$"
            
        # トイデータ）分類境界がアルファベッドのCの形をしている場合（ノイズあり）
        elif dataType == 6:
            dNum = 120
            np.random.seed(1)
                    
            cov1 = [[1,-0.8],[-0.8,1]]
            cov2 = [[1,0.8],[0.8,1]]

            X = np.random.multivariate_normal([0.5,1],cov1,int(dNum/2))
            X = np.concatenate([X,np.random.multivariate_normal([-1,-1],cov1,int(dNum/4))],axis=0)
            X = np.concatenate([X,np.random.multivariate_normal([-1,4],cov2,int(dNum/4))],axis=0)
            Y = np.concatenate([self.negLabel*np.ones([int(dNum/2),1]),self.posLabel*np.ones([int(dNum/2),1])],axis=0)
            
            # ノイズ
            X = np.concatenate([X,np.array([[-1.5,-1.5],[-1,-1]])],axis=0)
            Y = np.concatenate([Y,self.negLabel*np.ones([2,1])],axis=0)
            dNum += 2
            
            randInds = np.random.permutation(dNum)
            self.X = X[randInds]
            self.Y = Y[randInds]
                
            self.xLabel = "$x_1$"
            self.yLabel = "$x_2$"
            
        # MNIST
        elif dataType == 7:
            #-------------------
            # 学習用
            # 入力画像
            fp = gzip.open(os.path.join(self.path,'MNIST/train-images-idx3-ubyte.gz'),'rb')
            data = np.frombuffer(fp.read(),np.uint8,offset=16)
            self.Xtr = np.reshape(data,[-1,28*28])/255
                
            # ラベル
            fp = gzip.open(os.path.join(self.path,'MNIST/train-labels-idx1-ubyte.gz'),'rb')
            self.Ytr = np.frombuffer(fp.read(),np.uint8,offset=8)
            #-------------------
            
            #-------------------
            # 評価用
            # 入力画像
            fp = gzip.open(os.path.join(self.path,'MNIST/t10k-images-idx3-ubyte.gz'),'rb')
            data = np.frombuffer(fp.read(),np.uint8,offset=16)
            self.Xte = np.reshape(data,[-1,28*28])/255
            
            # ラベル
            fp = gzip.open(os.path.join(self.path,'MNIST/t10k-labels-idx1-ubyte.gz'),'rb')
            self.Yte = np.frombuffer(fp.read(),np.uint8,offset=8)
            #-------------------

            #-------------------
            # one-hot表現
            self.Ttr = np.eye(10)[self.Ytr]
            self.Tte = np.eye(10)[self.Yte]
            #-------------------
    #-------------------
####################

####################
# 決定木デモデータ用のクラス
class decisionTree:
    #-------------------
    # パス、ラベルの設定
    def __init__(self):
        self.path = "../data/decisionTree"
    #-------------------
    
    #-------------------
    # データの作成
    # dataType: データの種類（整数スカラー）
    def makeData(self,dataType=1):
        self.dataType = dataType

        # テニスをするか否か
        if dataType == 1:
            df = pd.read_csv(f'{self.path}/playTennis.csv')

        # 動物の種類
        elif dataType == 2:
            df = pd.read_csv(f'{self.path}/animals.csv')

        self.Xtr = df.drop('Y',axis=1)
        self.Ytr = df[['Y']].values
    #-------------------
####################

####################
# 感情分類データ用のクラス
class sentimentLabelling:
    
    #-------------------
    # パスの設定
    def __init__(self):
        self.path = '../data/sentiment_labelled_sentences'
    #-------------------
    
    #-------------------
    # データの作成
    # dataType: データの種類（整数スカラー）
    def makeData(self,dataType=1):
        self.dataType = dataType
        np.random.seed(1)
        
        # データの選択
        if dataType==1:   # amazonのレビュー
            fName = 'amazon_cells_labelled.txt'
        elif dataType==2: # yelpのレビュー
            fName = 'yelp_labelled.txt'
        elif dataType==3: # imdbのレビュー
            fName = 'imdb_labelled.txt'

        # csvファイルの読み込み
        data = pd.read_csv(os.path.join(self.path,fName),'\t',header=None)

        # データのインデックスをランダムにシャッフル
        dNum = len(data)
        randInds = np.random.permutation(dNum)

        # 文章データ
        sentences = data[0][randInds]
        self.Y = data[1][randInds]
        self.X = self.sentence2words(sentences)
    #-------------------

    #-------------------
    # 文章を単語に分割
    # sentence: 文章（文字列）
    def sentence2words(self,sentences):
        # 句読点
        punc = re.compile(r'[\[,\],-.?!,:;()"|0-9]')

        # 文章を小文字（lower）に変換し、スペースで分割（split）し、
        # 句読点を取り除き（punc.sub）、語wordsを取り出す
        words = [[punc.sub("",word) for word in sentence.lower().split()] for sentence in sentences] 

        # 空の要素を削除
        if words.count(""):
            words.pop(words.index(""))

        return words
    #-------------------
####################
