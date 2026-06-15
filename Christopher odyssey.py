import numpy as np
#PARAMETERS --------------------------------------
m=100 #let l range from 1 to 100

#GENERATE c_l = F(l/m) ---------------------------
l=np.arange(1,m+1)
def F(x):
  return np.sin(2*np.pi*x)
c=F(l/m)

#GENERATE \bar{f^*_l} ---------------------------
def g(x):
  return np.sqrt(2)*np.sin(2*np.pi*x)
def bar_f(x,c): #dependent on l
  return c*g(x)

#GENERATE
