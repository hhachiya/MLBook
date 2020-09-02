# -*- coding: utf-8 -*-
import numpy as np
import naiveBayes
import data

#-------------------
# 1. データの選択と読み込み
myData = data.sentimentLabelling()
myData.makeData(dataType=1)
#-------------------

#-------------------
# 2. データを学習と評価用に分割
dtrNum = int(len(myData.X)*0.9)  # 学習データ数
# 学習データ（全体の90%）
Xtr = myData.X[:dtrNum]
Ytr = myData.Y[:dtrNum]

# 評価データ（全体の10%）
Xte = myData.X[dtrNum:]
Yte = myData.Y[dtrNum:]
#-------------------

#-------------------
# 3. ナイーブベイズの学習

# 事前確率の設定
priors = np.array([[0.5,0.5]])

myModel = naiveBayes.naiveBayes(Xtr,Ytr,priors)
myModel.train()
#-------------------

#-------------------
# 4. ナイーブベイズの評価
print(f"学習データの正解率:{np.round(myModel.accuracy(Xtr,Ytr),decimals=2)}")
print(f"評価データの正解率:{np.round(myModel.accuracy(Xte,Yte),decimals=2)}")
#-------------------

#-------------------
# 5. 予測結果のCSVファイルへの出力
myModel.writeResult2CSV(Xtr,Ytr,fName=f"../results/naiveBayes_result_train_{myData.dataType}.csv")
myModel.writeResult2CSV(Xte,Yte,fName=f"../results/naiveBayes_result_test_{myData.dataType}.csv")
#-------------------

