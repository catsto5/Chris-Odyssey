import numpy as np
import matplotlib.pyplot as plt

#=================================================
#Generate trajectories x
#=================================================
def generate_traj(m, T, sigma, c):
    np.random.seed(1)
    eta = np.random.normal(loc=0, scale=sigma, size=(m, T)) #generate i.i.d. N(0,\sigma^2)
    x = np.zeros((m, T+1))  #initial state x_0=0
    for l in range(m):
        c_l=c[l]
        for t in range(T):
            x[l,t+1]=f(x[l,t], c_l)+eta[l,t]
    return x

#=================================================
#Generate c_l
#=================================================
def generate_c(m, beta):
    l = np.arange(1, m+1)
    return (l/m)**beta

#=================================================
#Generate \bar{f^*_l} 
#=================================================
def g(x):
    if 0 <= x < 0.5:
        return x
    elif 0.5 <= x <= 1:
        return 1-x
    else:
        return 0

def bar_f(x, c): #dependent on l
    return c * g(x) 

#================================================
#Generate f^*_l
#================================================
def f(x, c):
    return bar_f(x % 1, c)

#=================================================
#Fourier coeffs 
#=================================================
def theta_j_star_l(c_l, j): #in R
    if j == 1:
        return c_l / 4
    elif j % 4 == 2:
        return -np.sqrt(2) * c_l / (np.pi**2 * (j/2)**2)
    else:
        return 0

def theta_star_l(c_l, N): #in R^N
    theta = np.zeros(N)
    for j in range(1, N+1):
        theta[j-1] = theta_j_star_l(c_l, j)
    return theta

def theta_star(c, N): #in R^mn
    m = len(c)
    theta = np.zeros(m*N)
    for l in range(m):
        start = l*N
        theta[start:start+N] = theta_star_l(c[l], N)
    return theta

#================================================
#PENALISED LOSS FUNCTION f(θ)
#================================================
#DATA FIT TERM
def phi(j, x): #in R
    if j == 1:
        return 1
    elif j % 2 == 0:
        return np.sqrt(2) * np.cos(np.pi * x * j)
    else:
        return np.sqrt(2) * np.sin(np.pi * x * (j-1))
    
def phi_t_l(t, l, x, N):  #in R^N    
    phi_vec = np.zeros(N)
    for j in range(1, N+1):
        phi_vec[j-1] = phi(j, x[l, t])
    return phi_vec

def Phi_l(l, x, N): #in R^{TxN}
    T = x.shape[1]-1
    Phi_l = np.zeros((T, N))
    for t in range(T):
        for j in range(1, N+1):
            Phi_l[t, j-1] = phi(j, x[l, t])
    return Phi_l

def Phi_vec(x, N):
    m = x.shape[0]
    T = x.shape[1] - 1
    Phi1 = np.zeros((m*T, m*N))
    for l in range(m):
        Phi_l_temp = np.zeros((T, N))
        for t in range(T):
            for j in range(1, N+1):
                Phi_l_temp[t, j-1] = phi(j, x[l, t])
        Phi1[l*T:l*T+T, l*N:l*N+N] = Phi_l_temp
    return Phi1

def x_l(l, x): #in R^T
    return x[l, 1:]

def x_vec(x): #in R^{mT}
    m = x.shape[0]
    T = x.shape[1]-1
    x1 = np.zeros(m*T)
    for l in range(m):
        x1[l*T:(l+1)*T] = x_l(l, x)
    return x1

def data_fit_term(X, Phi, theta):
    v = X - Phi @ theta
    return np.sum(v**2)

def build_laplacian(m):
    A = np.zeros((m, m))
    for i in range(m-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    D = np.diag([1] + [2]*(m-2) + [1])
    return D - A

def smoothness_penalty(c, N, lam):
    m = len(c)
    I = np.eye(N)
    L = build_laplacian(m)
    theta = theta_star(c, N)
    return lam * (theta.T @ np.kron(L, I) @ theta)
