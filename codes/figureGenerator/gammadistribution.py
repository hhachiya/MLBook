import matplotlib.pylab as plt
import numpy as np

mode = 1

if mode == 1:     # single mu and sigma
  sigmas = [1.0]
  mus = [0.0]
elif mode == 2:   # multipe mus
  sigmas = [0.5,1.0,1.5,2.0]
  mus = [0.0]
elif mode == 3:   # multpile sigmas
  sigmas = [1.0]
  mus = [-3.0,-1.5,0.0,1.5,3.0]

fig = plt.figure(figsize=(8,4),dpi=100)
ax = fig.add_subplot(1,1,1)
x=np.arange(-8,8,0.1)

maxp = 0
for mu in mus:
  for sigma in sigmas:
    p=1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))
    ax.plot(x,p,lineWidth=2.0,label=f"$\mu=${mu},$\sigma^2=${sigma}")
    maxp = np.max([maxp,np.max(p)])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

if mode == 1:
  ax.set_xlim([-3,3])
  ax.set_ylim([0,maxp+maxp*0.01])
  
elif mode == 2:
  ax.set_xlim([-6,10])
  ax.set_ylim([-0.01,maxp+maxp*0.01])
  
elif mode == 3:
  ax.set_xlim([-6,6])
  ax.set_ylim([-0.01,maxp+maxp*0.01])
  

ax.set_xlabel("$x$",fontSize=14)
ax.legend()
ax.grid()
plt.show()