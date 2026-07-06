import numpy as np
import matplotlib.pyplot as plt
from utils import *

#================================================
#PARAMETERS
#================================================
m = 5 
T = 5
N = 5
sigma = 1.0
beta = 1
lam = 1 #lambda
sigma2 = [0.5, 1.0, 1.5, 2.0, 2.5]

#================================================
#GENERATE DATA
#================================================
c = generate_c(m, beta)
x = generate_traj(m, T, sigma, c)

#================================================
#COMPUTE QUANTITIES
#================================================
# Compute theta, Phi, and X
theta = theta_star(c, N)
Phi = Phi_vec(x, N)
X = x_vec(x)

# Compute penalized loss function
plf = data_fit_term(X, Phi, theta) + smoothness_penalty(c, N, lam)

#================================================
#EIGENVALUES OF M
#================================================
L = build_laplacian(m)
I = np.eye(N)
M = Phi.T @ Phi + lam * (np.kron(L, I))
eigenvalues = np.linalg.eigvals(H)
