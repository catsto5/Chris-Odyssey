import numpy as np

#================================================
#PARAMETERS
#================================================
m=100 #how many dynamical systems --- l=1,...,m
T=100
N=100
sigma=1.0

#================================================
#GENERATE c_l = F(l/m) 
#================================================
l=np.arange(1,m+1) #array
def F(x):
  return np.sin(2*np.pi*x)
c=F(l/m)

#================================================
#GENERATE \bar{f^*_l} 
#================================================
def g(x):
  return np.sqrt(2)*np.sin(2*np.pi*x)
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
#PLOTS
#================================================


#================================================
