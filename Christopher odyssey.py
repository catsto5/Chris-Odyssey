import numpy as np
import matplotlib.pyplot as plt

#================================================
#PARAMETERS
#================================================
m=5 #how many dynamical systems\
T=5
N=5
sigma=1.0
beta=1
lam=1 #lambda
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
      return 1-x
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
x=generate_traj(m, T, sigma, c)

#================================================
#FOURIER COEFFICIENTS θ_j^*(l)
#================================================
def theta_j_star_l(c_l, j): #in R
    if j == 1:
        return c_l / 4
    elif j % 4 == 2:
        return -np.sqrt(2) * c_l / (np.pi**2 * (j/2)**2)
    else:
        return 0
#for example, θ*_2^(3)=theta_j_star_l(c[3-1],2)
    
def theta_star_l(c_l, N): #in R^N
    theta = np.zeros(N)
    for j in range(1, N+1):
        theta[j-1] = theta_j_star_l(c_l, j)
    return theta
#for example, θ*^(3)=theta_star_l(c[3-1],N)

def theta_star(c,N): #in R^mn
    m=len(c)
    theta=np.zeros(m*N)
    for l in range(m):
        start=l*N
        theta[start:start+N]=theta_star_l(c[l],N)
    return theta

theta=theta_star(c,N)


#================================================
#PENALISED LOSS FUNCTION f(θ)
#================================================
#DATA FIT TERM
def phi(j,x): #in R
    if j==1:
        return 1
    elif j%2==0:
        return np.sqrt(2)*np.cos(np.pi*x*j)
    else:
        return np.sqrt(2)*np.sin(np.pi*x*(j-1))
    
def phi_t_l(t,l,x,N):  #in R^N    
    phi_vec=np.zeros(N)
    for j in range(1,N+1):
        phi_vec[j-1]=phi(j,x[l,t])
    return phi_vec

def Phi_l(l,x,N): #in R^{TxN}
    T=x.shape[1]-1
    Phi_l=np.zeros((T,N))
    for t in range(T):
        for j in range(1,N+1):
            Phi_l[t,j-1]=phi(j,x[l,t])
    return Phi_l

def Phi(x,N):
    m = x.shape[0]
    T = x.shape[1] - 1
    Phi1=np.zeros((m*T,m*N))
    for l in range(m):
        Phi_l=np.zeros((T,N))
        for t in range(T):
            for j in range(1,N+1):
                Phi_l[t,j-1]=phi(j,x[l,t])
        Phi1[l*T:l*T+T,l*N:l*N+N]=Phi_l
    return Phi1
Phi=Phi(x,N)

def x_l(l,x): #in R^T
    return x[l,1:]

def x_vec(x): #in R^{mT}
    m=x.shape[0]
    T=x.shape[1]-1
    x1=np.zeros(m*T)
    for l in range(m):
        x1[l*T:(l+1)*T]=x_l(l,x)
    return x1

X=x_vec(x)

v=X-Phi@theta
dft=0
for i in range(m*T):
    dft+=(v[i])**2


#SMOOTHNESS PENALTY
I=np.eye(N)
A=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        if i>j and i-j==1:
            A[(i,j)]=1
        elif i<j and j-i==1:   
            A[(i,j)]=1
D=np.zeros((m,m))
for i in range(m):
    for j in range(m):
        if i==j:
            D[(i,j)]=2
        D[(0,0)]=1
        D[(m-1,m-1)]=1         
L=D-A   
sp=lam*(theta_star(c,N).T@np.kron(L,I)@theta_star(c,N))        


#PENALISED LOSS FUNCTION
plf=dft+sp

#EIGENVALUES OF f(θ)
f=Phi.T@Phi + lam*(np.kron(L,I))
eigenvalues=np.linalg.eigvals(f)
min_eigenvalues=min(eigenvalues)
#================================================
#SANITY CHECK
#================================================

beta2=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] #values of beta to try
m2=[100,200,300,400,500,600,700,800,900,1000]

for i in m2:
    y = []
    for beta in beta2:
        c_test = (np.arange(1, i + 1) / i)**(beta)
        d_test = np.sum((c_test[:-1] - c_test[1:])**2)
        y.append(d_test)
    #plt.plot(beta2, y, label=f"m={m}")


#plt.xlabel(r"Value of $\beta$")
#plt.ylabel(r"$\sum_{l=1}^{m-1} |c_l - c_{l+1}|^2$")
#plt.legend()
