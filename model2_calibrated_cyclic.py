import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def create_roundabout_laplacian(n: int, epsilon: float, pinning: float = 0.0) -> np.ndarray:
    """
    Constructs an n x n Laplacian matrix for a cyclic graph (roundabout) topology.
    Forward edges have weight 1 - epsilon, backward edges have weight 1 + epsilon.
    A pinning gain anchors the first node if specified, breaking perfect symmetry.

    Parameters:
    n (int): Total number of vehicles in the roundabout.
    epsilon (float): Directional asymmetry bias.
    pinning (float): Absolute reference anchoring gain applied to node 0.

    Returns:
    np.ndarray: The finalized Laplacian/Metzler matrix.
    """
    L = np.zeros((n, n))
    for i in range(n):
        # Modulo arithmetic strictly enforces the periodic boundary conditions
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        L[i, prev_idx] = -1.0 - epsilon
        L[i, i] = 2.0
        L[i, next_idx] = -1.0 + epsilon

    # Apply absolute reference anchoring to the first vehicle
    L[0,0] += pinning
    return L

def evaluate_cyclic_h2_profile(n: int, epsilon: float, dynamics: str, leader: bool,
                               alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """
    Computes the spatial H2 norm (gap variance) profile for a cyclic platoon.

    Parameters:
    n (int): Total number of vehicles.
    epsilon (float): Directional asymmetry bias.
    dynamics (str): Control architecture ('RPAV' or 'RPRV').
    leader (bool): Boolean flag indicating presence of an anchored leader.
    alpha (float): Position feedback gain scaling the position Laplacian.
    beta (float):  Velocity damping gain. alpha = beta = 1.0 is the original model.

    Returns:
    np.ndarray: Array of length n containing the steady-state gap variance for each pair.
    """
    # Define anchoring gains based on the leader parameter
    alpha_pin = 0.5 if leader else 0.0
    beta_pin = 0.5 if leader else 0.0

    # Construct the position Laplacian
    L_alpha = create_roundabout_laplacian(n, epsilon, alpha_pin)

    # Apply regularization if the system lacks an absolute spatial anchor
    # This prevents the CALE solver from failing due to the zero eigenvalue
    if not leader:
        L_alpha += 1e-7 * np.eye(n)

    # Construct the closed-loop system matrix A based on the chosen dynamics
    if dynamics == 'RPAV':
        # Absolute velocity damping (Identity matrix block)
        A = np.block([[np.zeros((n, n)),     np.eye(n)],
                    [-alpha * L_alpha,     -beta * np.eye(n)]])
    elif dynamics == 'RPRV':
        # Relative velocity damping (Velocity Laplacian block)
        L_beta = create_roundabout_laplacian(n, epsilon, beta_pin)
        if not leader:
            L_beta += 1e-7 * np.eye(n)

        A = np.block([[np.zeros((n, n)),   np.eye(n)],
                       [-alpha * L_alpha,  -beta * L_beta]])
    else:
        raise ValueError("Invalid dynamics specified. Choose 'RPAV' or 'RPRV'.")

    # Disturbance input matrix B (acting exclusively on accelerations)
    B = np.block([[np.zeros((n, n))],
                  [np.eye(n)]])

    # Solve the Continuous Algebraic Lyapunov Equation (CALE) for the Gramian
    W_c = la.solve_continuous_lyapunov(A, -B @ B.T)

    # Extract the variance for all n gaps in the roundabout
    h2_data = np.zeros(n)
    for i in range(n):
        # Output selector matrix C for gap i
        C = np.zeros((1, 2 * n))
        C[0, i] = 1.0
        C[0, (i + 1) % n] = -1.0  # Periodic wrap-around for the final gap

        # Compute H2 norm squared (variance)
        variance = C @ W_c @ C.T
        h2_data[i] = variance.item()

    return h2_data

n_nodes = 50
bias = 0.02      

# Gains substituted into A
ALPHA = 6.201
BETA  = 32.5130   

pairs_range = np.arange(n_nodes)

# RPAV Configurations
rpav_sym_no = evaluate_cyclic_h2_profile(n_nodes, 0.0, 'RPAV', False, ALPHA, BETA)
rpav_asy_no = evaluate_cyclic_h2_profile(n_nodes, bias, 'RPAV', False, ALPHA, BETA)
rpav_sym_wL = evaluate_cyclic_h2_profile(n_nodes, 0.0, 'RPAV', True, ALPHA, BETA)
rpav_asy_wL = evaluate_cyclic_h2_profile(n_nodes, bias, 'RPAV', True, ALPHA, BETA)

# RPRV Configurations
rprv_sym_no = evaluate_cyclic_h2_profile(n_nodes, 0.0, 'RPRV', False, ALPHA, BETA)
rprv_asy_no = evaluate_cyclic_h2_profile(n_nodes, bias, 'RPRV', False, ALPHA, BETA)
rprv_sym_wL = evaluate_cyclic_h2_profile(n_nodes, 0.0, 'RPRV', True, ALPHA, BETA)
rprv_asy_wL = evaluate_cyclic_h2_profile(n_nodes, bias, 'RPRV', True, ALPHA, BETA)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# RPAV
ax1.plot(pairs_range, rpav_sym_no, '-', color='tab:blue', label='Sym RPAV no/L')
ax1.plot(pairs_range, rpav_asy_no, '-', color='tab:red', label='Asy RPAV no/L')
ax1.plot(pairs_range, rpav_sym_wL, '--', color='tab:green', label='Sym RPAV w/L')
ax1.plot(pairs_range, rpav_asy_wL, ':', color='k', label='Asy RPAV w/L')
ax1.set_xlabel('Vehicle Pair Along Cycle (Spatial Index $i$)')
ax1.set_ylabel(r'Gap Variance / Collision Potential $||G_k||_{\mathcal{H}_2}^2$')
ax1.set_title(rf'(a) RPAV, $\alpha$={ALPHA:.2f}, $\beta$={BETA:.2f}', fontsize=11)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax1.margins(x=0)
ax1.legend(fontsize='small', ncol=2)
ax1.grid(True, linestyle=':')

# RPRV
ax2.plot(pairs_range, rprv_sym_no, '-', color='tab:blue', label='Sym RPRV no/L')
ax2.plot(pairs_range, rprv_asy_no, '-', color='tab:red', label='Asy RPRV no/L')
ax2.plot(pairs_range, rprv_sym_wL, '--', color='tab:orange', label='Sym RPRV w/L')
ax2.plot(pairs_range, rprv_asy_wL, ':', color='k', label='Asy RPRV w/L')
ax2.set_xlabel('Vehicle Pair Along Cycle (Spatial Index $i$)')
ax2.set_ylabel(r'Gap Variance / Collision Potential $||G_k||_{\mathcal{H}_2}^2$')
ax2.set_title(rf'(b) RPRV, $\alpha$={ALPHA:.2f}, $\beta$={BETA:.2f}', fontsize=11)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax2.margins(x=0)
ax2.legend(fontsize='small', ncol=2)
ax2.grid(True, linestyle=':')

plt.tight_layout()
plt.show()