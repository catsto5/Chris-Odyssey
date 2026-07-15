import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\chris\Downloads\Odyssey")
from utils import *

# ---------- PARAMETERS ----------
R = 50
N = 10
beta = 0.5
T = 20

np.random.seed(1)
I = np.eye(N)

ms = [2, 5, 25, 50, 100]
Cs = [0, 1]

all_results = {}

# ---------- SIMULATIONS ----------
for C in Cs:
    print(f"\nRunning simulations for C = {C}")
    results = {}

    for m in ms:
        lam = C * m**((4/5) * beta)

        c = generate_c(m, beta)
        L = build_laplacian(m)

        theta_true_N = theta_star(c, N)
        theta_true_10N = theta_star(c, 10 * N)

        trunc_error = np.linalg.norm(theta_true_10N[m*N:])**2 / m

        est_errors = []

        for r in range(R):
            # Generate trajectories
            x = generate_traj(m, T, c)
            X = x_vec(x)
            Phi = Phi_vec(x, N)

            # Estimate theta
            M = Phi.T @ Phi + lam * np.kron(L, I)
            theta_hat = np.linalg.lstsq(M, Phi.T @ X, rcond=None)[0]

            # Estimation error
            est_error = np.linalg.norm(theta_hat - theta_true_N)**2 / m
            est_errors.append(est_error)

        results[m] = {
            'raw_est_errors': est_errors,
            'trunc_error': trunc_error,
            'mean_est_error': np.mean(est_errors)
        }

        print(f"m = {m:3d}, Mean Est Error = {np.mean(est_errors):.6f}")

    all_results[C] = results

# ---------- PREPARE DATA FOR PLOTTING ----------
data_C0 = []
data_C1 = []

for m in ms:
    total_errors_C0 = [
        err + all_results[0][m]['trunc_error']
        for err in all_results[0][m]['raw_est_errors']
    ]

    total_errors_C1 = [
        err + all_results[1][m]['trunc_error']
        for err in all_results[1][m]['raw_est_errors']
    ]

    data_C0.append(total_errors_C0)
    data_C1.append(total_errors_C1)

# ---------- PLOT ----------
plt.figure(figsize=(10, 6))

pos0 = np.arange(len(ms)) * 2 - 0.3
pos1 = np.arange(len(ms)) * 2 + 0.3

box0 = plt.boxplot(
    data_C0,
    positions=pos0,
    widths=0.5,
    patch_artist=True,
    showmeans=True,
    meanline=True
)

box1 = plt.boxplot(
    data_C1,
    positions=pos1,
    widths=0.5,
    patch_artist=True,
    showmeans=True,
    meanline=True
)

for patch in box0['boxes']:
    patch.set_facecolor('lightcoral')

for patch in box1['boxes']:
    patch.set_facecolor('lightblue')

plt.xticks(np.arange(len(ms)) * 2, ms)
plt.xlabel(r'Number of systems ($m$)', fontsize=30)
plt.ylabel(r'Approx. MSE', fontsize=30)
plt.title(rf'Trial for $\beta={beta}$ and $T={T}$', fontsize=30)

plt.tick_params(axis='both', labelsize=25)


plt.yscale('log')
plt.ylim(1e-1, 1e5)


plt.plot([], [], color='lightcoral', linewidth=8, label=r'$\lambda=0$')
plt.plot([], [], color='lightblue', linewidth=8, label=r'$\lambda=m^{4\beta/5}$')
plt.legend()

plt.tight_layout()
plt.show()
