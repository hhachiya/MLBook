import numpy as np
import matplotlib.pylab as  plt

#fig = plt.figure(figsize=(6,4),dpi=100)
x=np.array([6/13,7/13])
plt.bar([0,1],x,color=['b','r'])
plt.xticks([0,1],('青b','赤r'),fontSize=14)
plt.ylabel('確率',fontSize=14)
plt.xlabel('事象',fontSize=14)
plt.ylim([0,1])
plt.savefig("bernoulli_distribution_example.pdf")