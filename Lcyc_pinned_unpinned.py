"""
unpinned :  L_cyc                      (circulant, singular)
pinned   :  L_cyc + kappa * e0 e0^T    (rank-one anchor on node 0)
The analytical formula lambda_m = (2 - 2 cos th_m) - i(2 eps sin th_m),
th_m = 2 pi m / n, is used only as a one-line accuracy check on the unpinned case.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

N = 50
EPS = 0.02
KAPPA = 0.5

def cyclic_laplacian(n, epsilon, pinning=0.0):
    """Roundabout Laplacian: predecessor weight 1+eps, successor weight
    1-eps, self-degree 2. Optional rank-one pinning on node 0."""
    L = np.zeros((n, n))
    for i in range(n):
        L[i, (i - 1) % n] = -1.0 - epsilon
        L[i, i] = 2.0
        L[i, (i + 1) % n] = -1.0 + epsilon
    L[0, 0] += pinning
    return L

def analytical_spectrum(n, epsilon):
    th = 2.0 * np.pi * np.arange(n) / n
    return (2.0 - 2.0 * np.cos(th)) - 1j * (2.0 * epsilon * np.sin(th))

def matched_deviation(lam_a, lam_b):
    """Max and mean distance between two spectra under the optimal
    pairing. Sorting is unsafe here: modes m and n-m share a real part,
    so round-off decides their order and can swap the conjugates."""
    D = np.abs(lam_a[:, None] - lam_b[None, :])
    r, c = linear_sum_assignment(D)
    return D[r, c].max(), D[r, c].mean()

# Numerical spectra
lam_unpinned = np.linalg.eigvals(cyclic_laplacian(N, EPS))
lam_pinned = np.linalg.eigvals(cyclic_laplacian(N, EPS, KAPPA))

print("=" * 64)
print(f"L_cyc eigenvalues, n = {N}, epsilon = {EPS}")
print("=" * 64)

# short analytical cross-check (validation only)
err, _ = matched_deviation(lam_unpinned, analytical_spectrum(N, EPS))
print(f"\nAnalytical cross-check (unpinned): max |dlambda| = {err:.2e}")
print("  -- at the eigensolver's accuracy floor, so the closed form and")
print("     the computed spectrum agree exactly.")

print(f"\n{'':<12}{'drift eig':>14}{'min Re':>12}{'max Re':>10}{'max |Im|':>11}")
for name, lam in (('unpinned', lam_unpinned), ('pinned', lam_pinned)):
    d = lam[np.argmin(np.abs(lam))]
    print(f"{name:<12}{abs(d):>14.4e}{lam.real.min():>12.4e}"
          f"{lam.real.max():>10.4f}{np.abs(lam.imag).max():>11.5f}")

mx, mn = matched_deviation(lam_pinned, lam_unpinned)
print(f"\nDisplacement of the pinned spectrum from the unpinned one:")
print(f"  max  |dlambda| = {mx:.4e}")
print(f"  mean |dlambda| = {mn:.4e}")
a1 = 2 - 2 * np.cos(2 * np.pi / N)
print(f"  for reference, Fiedler value a_1 = {a1:.4e}")

# Sensitivity to the pinning gain
print(f"\n{'kappa':>8}{'drift eig':>14}{'max |dlambda|':>16}")
for k in (0.05, 0.1, 0.5, 1.0, 2.0):
    lam = np.linalg.eigvals(cyclic_laplacian(N, EPS, k))
    mxk, _ = matched_deviation(lam, lam_unpinned)
    print(f"{k:>8.2f}{np.min(np.abs(lam)):>14.4e}{mxk:>16.4e}")