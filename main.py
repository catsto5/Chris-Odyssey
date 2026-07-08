import numpy as np
import matplotlib.pyplot as plt
import os; os.chdir(r"C:\Users\chris\Downloads\Odyssey")
from utils import *

# ---------- PARAMETERS --------------
R=50
beta = 1
T = 5
N = 10
C=1
sigma = 1.0
results = {}
np.random.seed(1)
# ---------- SIMULATIONS --------------
for m in [5,25,50,100]:
    lam = C*(m)**((4/5)*beta) #lambda
    c = generate_c(m, beta)
    L = build_laplacian(m)  #builds the path graph G([m],E)
    theta_true_N = theta_star(c, N)
    theta_true_10N = theta_star(c, 10*N)
    trunc_error= np.linalg.norm(theta_true_10N[m*N:])**2 / m #this is the 2nd part of the MSE
    est_errors = [] #store error
    for r in range(1,R+1):
        #Generate trajectories
        x=generate_traj(m, T, sigma, c)    #random
        X = x_vec(x)  #random
        Phi = Phi_vec(x, N) #not random
        
        #Obtain $\hat\theta$ using $G(L)$ and trajectories.
        M = Phi.T @ Phi + lam * (np.kron(L, I)) #random
        theta_hat = np.linalg.lstsq(M, Phi.T @ X, rcond=None)[0] #this is random
        
        #Find MSE
        est_error = np.linalg.norm(theta_hat - theta_true_N)**2 / m
        est_errors.append(est_error) #store the error into the vector
    results[m] = {
        'raw_est_errors': est_errors,
        'trunc_error': trunc_error,
        'mean_est_error': np.mean(est_errors)
    }
    print(f"  Mean Est Error: {np.mean(est_errors):.6f}")
    
    
# ------------------- BOX PLOT OF TOTAL APPROXIMATED MSE -------------------
ms = sorted(results.keys())

data_to_plot = []
for m in ms:
    total_per_trial = [err + results[m]['trunc_error'] for err in results[m]['raw_est_errors']]
    data_to_plot.append(total_per_trial)

plt.figure(figsize=(10, 6))
box = plt.boxplot(data_to_plot, labels=ms, patch_artist=True, 
                  showmeans=True, meanline=True)

for patch in box['boxes']:
    patch.set_facecolor('lightblue')

plt.xlabel('Number of systems ($m$)', fontsize=12)
plt.ylabel('Approximated MSE', fontsize=12)
plt.title(f'Distribution of Approximated MSE over {R} Trials', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.yscale('log')
plt.show()

#plf = data_fit_term(X, Phi, theta) + smoothness_penalty(c, N, lam)
