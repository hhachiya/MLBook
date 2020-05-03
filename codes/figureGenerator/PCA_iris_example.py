import pandas as pd
import numpy as np
import pdb
import matplotlib.pylab as plt
from sklearn import datasets

####################
# irisデータ
iris = datasets.load_iris()
X = iris.data
Y = iris.target
####################

####################
# 主成分分析
X = X-np.mean(X,axis=0)
cov=np.cov(X.T,bias=1)
lam,W=np.linalg.eig(cov)
inds = np.argsort(lam)[::-1]
lam = lam[inds]
W=W[:,inds]
####################

####################
# 次元削減
Xlow = np.matmul(X,W[:,[0,1]])
####################

####################
# 低次元空間の写像のプロット

plt.plot(Xlow[Y==0,0],Xlow[Y==0,1],'m.',markerSize=14,label="Setosa")
plt.plot(Xlow[Y==1,0],Xlow[Y==1,1],'bx',markerSize=14,label="Versicolour")
plt.plot(Xlow[Y==2,0],Xlow[Y==2,1],'m^',markerSize=14,label="Virginica")
plt.legend(fontsize=14)

plt.grid()
plt.savefig("PCA_iris_example.pdf")
####################