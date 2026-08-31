import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

"""
FREE PARAMETERS SUMMARY TABLE
Parameter   | Type  | Description / Usage
-------------------------------------------------------------------------------
n / n_nodes | int   | Total number of vehicles (nodes) in the platoon (e.g., 50)
epsilon     | float | Directional bias/coupling offset for asymmetric flow (e.g., 0.02)
pinning     | float | Leader anchoring gain applied to the first vehicle (0.0 or 0.5)
dynamics    | str   | Control architecture: 'RPAV' or 'RPRV'
leader      | bool  | Flag indicating if a pinned leader exists (True/False)
-------------------------------------------------------------------------------
"""

# 1. GRAPH THEORETIC FOUNDATIONS
def create_platoon_laplacian(n: int, epsilon: float, pinning: float = 0.0) -> np.ndarray:
    L = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            L[i, 0] = 1.0 - epsilon
            L[i, 1] = -1.0 + epsilon
        elif i == n - 1:
            L[i, n - 2] = -1.0 - epsilon
            L[i, n - 1] = 1.0 + epsilon
        else:
            L[i, i - 1] = -1.0 - epsilon
            L[i, i] = 2.0
            L[i, i + 1] = -1.0 + epsilon
    L[0, 0] += pinning
    return L

# 2. SYSTEM DYNAMICS & H2 NORM EVALUATION
def evaluate_h2_profile(n: int, epsilon: float, dynamics: str, leader: bool) -> np.ndarray:
    alpha_pin = 0.3174 if leader else 0.0
    beta_pin = 0.126 if leader else 0.0
    
    L_alpha = create_platoon_laplacian(n, epsilon, alpha_pin)
    if not leader:
        L_alpha += 1e-7 * np.eye(n) 
        
    if dynamics == 'RPAV':
        A = np.block([[np.zeros((n, n)), np.eye(n)], [-L_alpha, -np.eye(n)]])
    else: 
        L_beta = create_platoon_laplacian(n, epsilon, beta_pin)
        if not leader:
            L_beta += 1e-7 * np.eye(n)
        A = np.block([[np.zeros((n, n)), np.eye(n)], [-L_alpha, -L_beta]])
        
    B = np.block([[np.zeros((n, n))], [np.eye(n)]])
    
    W_c = la.solve_continuous_lyapunov(A, -B @ B.T)
    
    h2_data = np.zeros(n - 1)
    for i in range(n - 1):
        C = np.zeros((1, 2 * n))
        C[0, i] = 1.0
        C[0, i + 1] = -1.0
        h2_data[i] = (C @ W_c @ C.T)[0, 0]
        
    return h2_data

n_nodes = 50
bias = 0.07531
pairs_range = np.arange(1, n_nodes) 

rpav_sym_no = evaluate_h2_profile(n_nodes, 0.0, 'RPAV', False)
rpav_asy_no = evaluate_h2_profile(n_nodes, bias, 'RPAV', False)
rpav_sym_wL = evaluate_h2_profile(n_nodes, 0.0, 'RPAV', True)
rpav_asy_wL = evaluate_h2_profile(n_nodes, bias, 'RPAV', True)

rprv_sym_no = evaluate_h2_profile(n_nodes, 0.0, 'RPRV', False)
rprv_asy_no = evaluate_h2_profile(n_nodes, bias, 'RPRV', False)
rprv_sym_wL = evaluate_h2_profile(n_nodes, 0.0, 'RPRV', True)
rprv_asy_wL = evaluate_h2_profile(n_nodes, bias, 'RPRV', True)

# 4. PLOTTING
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- (a) RPAV Control ---
ax1.plot(pairs_range, rpav_sym_no, '-', color='tab:blue', label='Sym RPAV no/L (Sim)')
ax1.plot(pairs_range, rpav_asy_no, '-', color='tab:red', label='Asy RPAV no/L (Sim)')
ax1.plot(pairs_range, rpav_sym_wL, '--', color='tab:green', label='Sym RPAV w/L (Sim)')
ax1.plot(pairs_range, rpav_asy_wL, ':', color='k', label='Asy RPAV w/L (Sim)')

ax1.set_xlabel('Vehicle Pair Along Line (Spatial Index $i$)')
ax1.set_ylabel(r'Gap Variance / Collision Potential $||G_\alpha||_{H_2}^2$')
ax1.set_title('(a) RPAV Control', fontsize=10)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax1.margins(x=0)
ax1.legend(fontsize='small', ncol=2)
ax1.grid(True, linestyle=':')

# --- (b) RPRV Control ---
ax2.plot(pairs_range, rprv_sym_no, '-', color='tab:blue', label='Sym RPRV no/L (Sim)')
ax2.plot(pairs_range, rprv_asy_no, '-', color='tab:red', label='Asy RPRV no/L (Sim)')
ax2.plot(pairs_range, rprv_sym_wL, '--', color='tab:orange', label='Sym RPRV w/L (Sim)')
ax2.plot(pairs_range, rprv_asy_wL, ':', color='k', label='Asy RPRV w/L (Sim)')

ax2.set_xlabel('Vehicle Pair Along Line (Spatial Index $i$)')
ax2.set_ylabel(r'Gap Variance / Collision Potential $||G_{\alpha\beta}||_{H_2}^2$')
ax2.set_title('(b) RPRV Control', fontsize=10)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax2.margins(x=0)
ax2.legend(fontsize='small', ncol=2)
ax2.grid(True, linestyle=':')

plt.tight_layout()
plt.show()
