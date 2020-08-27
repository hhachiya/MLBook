# -*- coding: utf-8 -*-
import numpy as np
import LDA as lda
import data

#-------------------
# 1. データの作成
myData = data.classification(negLabel=-1,posLabel=1)
myData.makeData(dataType=2)
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
'''
#-------------------
# アンダーサンプリング

# カテゴリの最小のデータ数
minNum = np.min([np.sum(Ytr==-1),np.sum(Ytr==1)])

# 各カテゴリのデータ
Xneg = Xtr[Ytr[:,0]==-1]
Xpos = Xtr[Ytr[:,0]==1]

# 最小データ数分だけ各カテゴリから抽出し結合
Xtr = np.concatenate([Xneg[:minNum],Xpos[:minNum]],axis=0)
Ytr = np.concatenate([-1*np.ones(shape=[minNum,1]),1*np.ones(shape=[minNum,1])])
#-------------------
'''

#-------------------
# 3. 線形判別モデルの学習
myModel = lda.LDA(Xtr,Ytr)
myModel.train()
#-------------------

#-------------------
# 4. 線形判別モデルの評価
print(f"モデルパラメータ:\nw={myModel.w},\n平均m={myModel.m}")
print(f"正解率={myModel.accuracy(Xte,Yte):.2f}")
#-------------------

#-------------------
# 5. 真値と予測値のプロット
if Xtr.shape[1] == 2:
    myModel.plotModel2D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/LDA_result_train_{myData.dataType}.pdf")    
#-------------------