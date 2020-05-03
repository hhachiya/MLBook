import numpy as np
import matplotlib.pylab as  plt

#------
# 2次関数
x=np.arange(-5,5,0.1)
y=np.power(x,2)

fig = plt.figure(figsize=(6,4),dpi=100)

plt.plot(x,y,'r')

plt.yticks([-1,-0.5,0,0.5,1])
plt.xlabel('$f(x_i)$',fontSize=14)
plt.ylabel('$p_i$',fontSize=14)
plt.ylim([-1,1])
plt.xlim([-1.5,1.0])
plt.grid()

plt.show()
#------

#------
# 3次関数
x=np.arange(-5,5,0.1)
y=np.power(x,3)

fig = plt.figure(figsize=(6,4),dpi=100)

plt.plot(x,y,'b')

plt.yticks([0])
plt.xlabel('$w$')
plt.ylabel('$f(w)$')
plt.ylim([-1,1])
plt.xlim([-1,1.0])
plt.grid()

plt.show()
#------

#------
# 3次関数
x=np.arange(-5,5,0.1)
y=np.power(x,3)

fig = plt.figure(figsize=(6,4),dpi=100)

plt.plot(x,y,'b',lineWidth=8)

plt.yticks([0])
plt.xlabel('$w$')
plt.ylabel('$\mathcal{E}(w)$')
plt.ylim([-1,1])
plt.xlim([-1,1.0])
plt.grid()

plt.show()
#------

#------
# シグモイド関数
x=np.arange(-10,10,0.1)
y=1/(1+np.exp(-x))

fig = plt.figure(figsize=(6,4),dpi=100)

plt.plot(x,y,'r',lineWidth=6)

plt.yticks([0,0.25,0.5,0.75,1])
plt.xlabel('$f(x_i)$',fontSize=14)
plt.ylabel('$p_i$',fontSize=14)
plt.ylim([-0.02,1.02])
plt.xlim([-10,10])
plt.grid()

#plt.show()
plt.savefig("sigmoid_function.pdf")
#------

#------
# シグモイド関数2
x=np.arange(-10,10,0.1)
y=1/(1+np.exp(-x))

#fig = plt.figure(figsize=(6,4),dpi=100)

plt.yticks([0,0.25,0.5,0.75,1])
plt.plot(x,y,'r',lineWidth=8)
plt.xlabel('$s$',fontSize=14)
plt.ylabel('$\sigma(s)$',fontSize=14)
plt.ylim([-0.02,1.02])
plt.xlim([-10,10])
plt.grid()
#plt.show()
plt.savefig("sigmoid_function_activation.pdf")
#------

#------
# ReLU
plt.close()

x=np.arange(-10,10,0.1)
y=np.array([0 if xtmp < 0 else xtmp for xtmp in x])
#fig = plt.figure(figsize=(6,4),dpi=100)

plt.plot(x,y,'r',lineWidth=8)

#plt.yticks([0,0.25,0.5,0.75,1])
plt.xlabel('$s$',fontSize=14)
plt.ylabel('$\mathrm{ReLU}(s)$',fontSize=14)
plt.ylim([-0.02,10.02])
plt.xlim([-10,10])
plt.grid()

#plt.show()
plt.savefig("ReLU_activation.pdf")
#------

#------
# ReLU2
x=np.arange(-10,10,0.1)
y=np.array([0 if xtmp < 0 else xtmp for xtmp in x])

fig = plt.figure(figsize=(6,4),dpi=100)

plt.plot(x,y,'r',lineWidth=8)

#plt.yticks([0,0.25,0.5,0.75,1])
plt.xlabel('$s$')
plt.ylabel('$\mathrm{ReLU}(s)$')
plt.ylim([-0.5,10.5])
plt.xlim([-10,10])
#plt.grid()

plt.show()
#------

#------
# log関数
x=np.arange(0.00001,2,0.0001)
y=np.log(x)

#fig = plt.figure(figsize=(6,4),dpi=100)

plt.plot(x,y,'r',lineWidth=6)

plt.xticks([0,0.25,0.5,0.75,1,2])
plt.yticks([-10,-7.5,-5,-2.5,-1,0,1,2])

plt.xlabel('$x$')
plt.ylabel('$log(x)$')
plt.ylim([-10,1.02])
plt.xlim([-0.02,2])
plt.grid()

#plt.show()
plt.savefig('log_function.pdf')
#------