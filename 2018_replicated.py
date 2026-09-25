import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

REL_OUTPUT_DIR = "2018_replicated"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), REL_OUTPUT_DIR)


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


def evaluate_h2_profile(n: int, epsilon: float, dynamics: str, leader: bool) -> np.ndarray:
    alpha_pin = 0.5 if leader else 0.0
    beta_pin = 0.5 if leader else 0.0

    L_alpha = create_platoon_laplacian(n, epsilon, alpha_pin)
    if not leader:
        # regularize the singular consensus mode so the Lyapunov solve doesn't blow up
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
bias = 0.02
pairs_range = np.arange(1, n_nodes)

rpav_sym_no = evaluate_h2_profile(n_nodes, 0.0, 'RPAV', False)
rpav_asy_no = evaluate_h2_profile(n_nodes, bias, 'RPAV', False)
rpav_sym_wL = evaluate_h2_profile(n_nodes, 0.0, 'RPAV', True)
rpav_asy_wL = evaluate_h2_profile(n_nodes, bias, 'RPAV', True)

rprv_sym_no = evaluate_h2_profile(n_nodes, 0.0, 'RPRV', False)
rprv_asy_no = evaluate_h2_profile(n_nodes, bias, 'RPRV', False)
rprv_sym_wL = evaluate_h2_profile(n_nodes, 0.0, 'RPRV', True)
rprv_asy_wL = evaluate_h2_profile(n_nodes, bias, 'RPRV', True)

# (simulation array, digitized i values, digitized variance values)
comparison_data = {
    "Sym RPAV w/L": (rpav_sym_wL, 
                     [2.34642, 6.92426, 11.97335, 16.61851, 20.85975, 25.23562, 29.67882, 33.44881, 37.55540, 48.93268],
                     [0.50004, 0.50004, 0.50004, 0.50016, 0.50010, 0.50004, 0.50004, 0.50010, 0.50010, 0.50004]),
    "Sym RPAV no/L": (rpav_sym_no,
                      [1.47125, 9.75175, 13.99299, 18.57083, 22.94670, 26.04348, 29.00561, 33.71809, 38.90182, 46.03787],
                      [0.50004, 0.50004, 0.50010, 0.50004, 0.50010, 0.50004, 0.50004, 0.50004, 0.50010, 0.50004]),
    "Asy RPAV w/L": (rpav_asy_wL,
                     [1.06732, 1.53857, 2.41374, 3.22160, 4.02945, 5.37588, 6.92426, 9.07854, 12.04067, 15.54137, 19.24404, 23.21599, 27.12062, 31.42917, 36.94951, 43.41234, 46.10519, 47.45161, 48.39411, 48.93268],
                     [0.47762, 0.47992, 0.48303, 0.48478, 0.48640, 0.48827, 0.48957, 0.49107, 0.49244, 0.49381, 0.49487, 0.49580, 0.49680, 0.49767, 0.49879, 0.50010, 0.50066, 0.50159, 0.50346, 0.50502]),
    "Asy RPAV no/L": (rpav_asy_no,
                      [1.13464, 1.67321, 2.41374, 3.49088, 5.57784, 8.53997, 11.77139, 16.48387, 20.79243, 24.56241, 27.65919, 29.81346, 32.16971, 37.69004, 41.59467, 43.21038, 44.75877, 46.71108, 48.05750, 49.00000],
                      [0.49536, 0.49692, 0.49860, 0.49948, 0.50010, 0.50022, 0.50035, 0.50041, 0.50041, 0.50041, 0.50041, 0.50041, 0.50047, 0.50047, 0.50047, 0.50047, 0.50066, 0.50116, 0.50222, 0.50489]),
    "Sym RPRV w/L": (rprv_sym_wL,
                     [1.06362, 4.30701, 7.03135, 10.01549, 12.41551, 13.97237, 15.65902, 18.44831, 21.23770, 23.24856, 24.54572, 26.29733, 29.02188, 32.52487, 34.60057, 38.10355, 42.06043, 45.49835, 47.76880, 49.00145],
                     [24.48542, 22.85592, 21.44082, 20.02573, 18.78216, 18.01029, 17.19554, 15.78045, 14.40823, 13.37907, 12.65009, 11.83533, 10.50600, 8.79074, 7.71870, 6.00343, 3.98799, 2.22985, 1.11492, 0.55746]),
    "Sym RPRV no/L": (rprv_sym_no,
                      [1.06629, 5.09792, 7.17841, 9.64873, 11.53380, 13.15873, 15.10843, 17.05779, 21.41106, 24.65892, 27.77653, 29.85457, 31.99745, 34.26968, 35.89271, 38.61894, 41.27999, 43.42165, 45.75738, 48.93661],
                      [0.51458, 2.27273, 3.04460, 3.85935, 4.41681, 4.84563, 5.27444, 5.57461, 6.13208, 6.21784, 6.17496, 6.00343, 5.78902, 5.36021, 5.06003, 4.37393, 3.60206, 2.91595, 1.92967, 0.60034]),
    "Asy RPRV w/L": (rprv_asy_wL,
                     [1.07554, 3.73937, 6.79282, 10.95112, 14.26482, 16.86382, 20.82727, 24.53090, 27.19508, 30.11861, 33.23690, 35.31527, 37.45815, 40.24933, 42.58495, 43.75221, 45.69689, 46.99271, 48.09423, 49.00145],
                     [4.07376, 4.37393, 4.63122, 5.14580, 5.57461, 5.91767, 6.43225, 6.94683, 7.37564, 7.63293, 7.84734, 7.80446, 7.59005, 6.90395, 5.87479, 5.14580, 3.64494, 2.40137, 1.37221, 0.55746]),
    "Asy RPRV no/L": (rprv_asy_no,
                      [1.06607, 4.05612, 9.06025, 11.07501, 14.12968, 16.98960, 20.04405, 22.05892, 24.07356, 26.08821, 27.97295, 31.09201, 34.08061, 36.22372, 39.01579, 41.80641, 43.49251, 45.04847, 46.79829, 49.00167],
                      [0.42882, 1.28645, 2.35849, 2.83019, 3.55918, 4.33105, 4.97427, 5.48885, 5.91767, 6.34648, 6.77530, 7.28988, 7.59005, 7.46141, 7.11835, 6.21784, 5.18868, 4.07376, 2.57290, 0.64322])
}


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

x_ticks = np.arange(1, 51, 6)

# --- (a) RPAV Control ---
ax1.plot(pairs_range, rpav_sym_no, '-', color='tab:blue', label='Sym RPAV no/L (Sim)')
ax1.plot(pairs_range, rpav_asy_no, '-', color='tab:red', label='Asy RPAV no/L (Sim)')
ax1.plot(pairs_range, rpav_sym_wL, '--', color='tab:green', label='Sym RPAV w/L (Sim)')
ax1.plot(pairs_range, rpav_asy_wL, ':', color='k', label='Asy RPAV w/L (Sim)')

ax1.plot(comparison_data["Sym RPAV no/L"][1], comparison_data["Sym RPAV no/L"][2], 's', color='tab:blue', markersize=5)
ax1.plot(comparison_data["Asy RPAV no/L"][1], comparison_data["Asy RPAV no/L"][2], 'o', color='tab:red', markersize=5)
ax1.plot(comparison_data["Sym RPAV w/L"][1], comparison_data["Sym RPAV w/L"][2], 'X', color='tab:green', markersize=5)
ax1.plot(comparison_data["Asy RPAV w/L"][1], comparison_data["Asy RPAV w/L"][2], 'd', color='k', markersize=5)

ax1.set_xlabel('Vehicle Pair Along Line (Spatial Index $i$)')
ax1.set_ylabel(r'Gap Variance / Collision Potential $||G_\alpha||_{H_2}^2$')
ax1.set_title('(a) RPAV Control\n(Markers denote digitized data from Ji & Gayme results)', fontsize=10)
ax1.set_xlim(1, 49)
ax1.set_ylim(0.47, 0.51)
ax1.set_xticks(x_ticks)
ax1.legend(fontsize='small', ncol=2)
ax1.grid(True, linestyle=':')

# --- (b) RPRV Control ---
ax2.plot(pairs_range, rprv_sym_no, '-', color='tab:blue', label='Sym RPRV no/L (Sim)')
ax2.plot(pairs_range, rprv_asy_no, '-', color='tab:red', label='Asy RPRV no/L (Sim)')
ax2.plot(pairs_range, rprv_sym_wL, '--', color='tab:orange', label='Sym RPRV w/L (Sim)')
ax2.plot(pairs_range, rprv_asy_wL, ':', color='k', label='Asy RPRV w/L (Sim)')

ax2.plot(comparison_data["Sym RPRV no/L"][1], comparison_data["Sym RPRV no/L"][2], 's', color='tab:blue', markersize=5)
ax2.plot(comparison_data["Asy RPRV no/L"][1], comparison_data["Asy RPRV no/L"][2], 'o', color='tab:red', markersize=5)
ax2.plot(comparison_data["Sym RPRV w/L"][1], comparison_data["Sym RPRV w/L"][2], 'X', color='tab:orange', markersize=5)
ax2.plot(comparison_data["Asy RPRV w/L"][1], comparison_data["Asy RPRV w/L"][2], 'd', color='k', markersize=5)

ax2.set_xlabel('Vehicle Pair Along Line (Spatial Index $i$)')
ax2.set_ylabel(r'Gap Variance / Collision Potential $||G_{\alpha\beta}||_{H_2}^2$')
ax2.set_title('(b) RPRV Control\n(Markers denote digitized data from Ji & Gayme results)', fontsize=10)
ax2.set_xlim(1, 49)
ax2.set_ylim(0, 25)
ax2.set_xticks(x_ticks)
ax2.legend(fontsize='small', ncol=2)
ax2.grid(True, linestyle=':')

plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUTPUT_DIR, "h2_profile_comparison.png"), dpi=200, bbox_inches="tight")
print(f"\nSaved figure to {REL_OUTPUT_DIR}")

plt.show()

# digitized points sit at fractional indices (pixel-mapped from the paper's plots),
# so compare against a cubic-spline interpolant of the simulated curve
print(f"\n{'Control Architecture':<22} | {'Spatial Index i':<15} | {'Digitized Variance':<18} | {'Python Variance':<18}")
print("-" * 81)

for arch_name, (sim_array, i_vals, dig_vals) in comparison_data.items():
    interp_func = interp1d(pairs_range, sim_array, kind='cubic', fill_value="extrapolate")

    for i, dig_val in zip(i_vals, dig_vals):
        py_val = interp_func(i)
        print(f"{arch_name:<22} | {i:<15.5f} | {dig_val:<18.5f} | {py_val:<18.5f}")
        arch_name = ""

    print("-" * 81)