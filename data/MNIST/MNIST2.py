# -*- coding: utf-8 -*-
import gzip
import numpy as np

#-------------------
# 学習用
# 入力画像
fp = gzip.open('train-images-idx3-ubyte.gz','rb')
data = np.frombuffer(fp.read(),np.uint8,offset=16)
Xtr = np.reshape(data,[-1,28*28])/255
    
# ラベル
fp = gzip.open('train-labels-idx1-ubyte.gz','rb')
Ytr = np.frombuffer(fp.read(),np.uint8,offset=8)
#-------------------
#-------------------
# 評価用
# 入力画像
fp = gzip.open('t10k-images-idx3-ubyte.gz','rb')
data = np.frombuffer(fp.read(),np.uint8,offset=16)
Xte = np.reshape(data,[-1,28*28])/255
  
# ラベル
fp = gzip.open('t10k-labels-idx1-ubyte.gz','rb')
Yte = np.frombuffer(fp.read(),np.uint8,offset=8)  
#-------------------

#-------------------
# one-hot表現
Ttr = np.eye(10)[Ytr]
Tte = np.eye(10)[Yte]

print(f"Ttr shape={Ttr.shape}")
print(f"Tte shape={Tte.shape}")
print(f"Ttr=\n{Ttr[:3]}")
#-------------------