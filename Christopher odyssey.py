import numpy as np
import matplotlib.pyplot as plt

#================================================
#PARAMETERS
#================================================
m=20 #how many dynamical systems\
T=100
N=100
sigma=1.0
beta=1
sigma2=[0.5,1.0,1.5,2.0,2.5]

#================================================
#GENERATE c_l = F(l/m) 
#================================================
l=np.arange(1,m+1) #array
def F(x):
  return x**beta
c=F(l/m)

#================================================
#GENERATE \bar{f^*_l} 
#================================================
def g(x):
  if 0<=x<0.5:
      return x
  elif 0.5<=x<=1:
      return -x
def bar_f(x,c): #dependent on l
  return c*g(x)

#================================================
#GENERATE f^*_l
#================================================
def f(x,c):
  return bar_f(x % 1,c)

#================================================
#GENERATE TRAJECTORIES
#================================================
np.random.seed(1)
eta = np.random.normal(loc=0.0, scale=sigma, size=(m, T)) #generate i.i.d. N(0,\sigma^2)
x = np.zeros((m, T+1))  #initial state x_0=0
for i in range(m):
  c_l=c[i]
  for t in range(T):
    x[i,t+1]=f(x[i,t],c_l)+eta[i,t]

#================================================
#SANITY CHECK
#================================================

beta2=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] #values of beta to try
m2=[100,200,300,400,500,600,700,800,900,1000]

for m in m2:
    y = []
    for beta in beta2:
        c = (np.arange(1, m + 1) / m)**(beta)
        d = np.sum((c[:-1] - c[1:])**2)
        y.append(d)
    plt.plot(beta2, y, label=f"m={m}")


plt.xlabel(r"Value of $\beta$")
plt.ylabel(r"$\sum_{l=1}^{m-1} |c_l - c_{l+1}|^2$")
plt.legend()
plt.show()
