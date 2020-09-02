# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# クラス
class naiveBayes():
    #-------------------
    # 1. 学習データの初期化
    # X: 各文章の単語リスト（データ数のリスト）
    # Y: 出力データ（データ数×1のnumpy.ndarray）
    # priors: 事前確率（1×カテゴリ数のnumpy.ndarray）
    def __init__(self,X,Y,priors):
        
        # 学習データの設定
        self.X = X
        self.Y = Y

        # 重複なしの単語一覧の作成
        self.wordDict = list(np.unique(np.concatenate(self.X)))

        # 各文章の単語の出現回数
        self.wordNums = self.countWords(self.X)
        
        # 事前確率の設定
        self.priors = priors
    #-------------------
    
    #-------------------
    # 2. 各文章中の単語の出現回数をカウント
    # X: 各文章の単語リスト（データ数のリスト）
    def countWords(self,X):
        
        # 各文章中の単語の出現回数をカウント
        cntWordsAll = []
        for words in X:
            cntWords = np.zeros(len(self.wordDict))
    
            for word in words:
                cnt = self.wordDict.count(word)
                
                if cnt:
                    cntWords[self.wordDict.index(word)] += cnt
    
            cntWordsAll.append(cntWords)
    
        return np.array(cntWordsAll)
    #-------------------

    #-------------------
    # 3. 単語の尤度の計算
    def train(self):

        # カテゴリごとの単語の出現回数
        wordNumsCat = [np.sum(self.wordNums[self.Y==y],axis=0) for y in np.unique(self.Y)]
        wordNumsCat = np.array(wordNumsCat)

        # 単語の尤度p(x|y)（カテゴリごと単語の出現割合）の計算
        self.wordL = wordNumsCat/np.sum(wordNumsCat,axis=1,keepdims=True)
    #-------------------
    
    #-------------------
    # 4. 文章の事後確率の計算
    # X: 各文章の単語リスト（データ数のリスト）
    def predict(self,X):
        
        # 文章を単語に分割
        wordNums = self.countWords(X)

        # 文章の尤度計算
        sentenceL = [np.product(self.wordL[ind]**wordNums,axis=1) for ind in range(len(np.unique(self.Y)))]
        sentenceL = np.array(sentenceL)
        
        # 事後確率の計算
        sentenceP = sentenceL.T * self.priors

        # 予測
        predict = np.argmax(sentenceP,axis=1)

        return predict
    #-------------------
    
    #-------------------
    # 5. 正解率の計算
    # X:入力データ（データ数×次元数のnumpy.ndarray）
    # Y:出力データ（データ数×1のnumpy.ndarray）
    def accuracy(self,X,Y):
        return np.sum(self.predict(X) - Y.T==0)/len(Y)
    #-------------------
        
    #-------------------
    # 5. 予測結果のcsvファイルへの書き込み
    # X: 文章データ（データ数のリスト）
    # Y: 真値（データ数×1のnumpy.ndarray）
    # fName：csvファイルの保存先（文字列）
    def writeResult2CSV(self,X,Y,fName="../results/sentimental_results.csv"):
        P = self.predict(X)
        
        # データフレームの作成
        df = pd.DataFrame(np.array([Y,P,X]).T,columns=['gt','predict','sentence'],index=np.arange(len(X)))
        
        # csvファイルに書き出し
        df.to_csv(fName,index=False)
    #-------------------
####################
