import os
import csv
import numpy as np
N, EPS, KAPPA, BETA = 50, 0.02, 0.5, 0.126
TOL = 1e-12

REL_OUTPUT_DIR = "Lcyc_numerical_results"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), REL_OUTPUT_DIR)

def cyclic_laplacian(n, epsilon, pinning=0.0):
    L = np.zeros((n, n))
    for i in range(n):
        L[i, (i - 1) % n] = -1.0 - epsilon
        L[i, i] = 2.0
        L[i, (i + 1) % n] = -1.0 + epsilon
    L[0, 0] += pinning
    return L

def A_rpav(L, beta):
    n = L.shape[0]
    return np.block([[np.zeros((n, n)), np.eye(n)], [-L, -beta * np.eye(n)]])

def A_rprv(La, Lb):
    n = La.shape[0]
    return np.block([[np.zeros((n, n)), np.eye(n)], [-La, -Lb]])

def report(label, A):
    s = np.linalg.eigvals(A)
    abscissa = s.real.max()
    n_imag = int(np.sum(s.real > -TOL))
    verdict = "Hurwitz" if abscissa < -TOL else (
        "NOT Hurwitz (marginal)" if abscissa < TOL else "NOT Hurwitz (unstable)")
    print(f"{label:<28}{abscissa:>14.3e}{n_imag:>12d}   {verdict}")
    return s

def modal_roots(lam, delta):
    """Roots of s^2 + delta s + lam for each modal pair."""
    out = []
    for l, d in zip(lam, np.broadcast_to(delta, lam.shape)):
        out.extend(np.roots([1.0, d, l]))
    return np.array(out)

def matched_max_dev(a, b):
    from scipy.optimize import linear_sum_assignment
    D = np.abs(a[:, None] - b[None, :])
    r, c = linear_sum_assignment(D)
    return D[r, c].max()

L = cyclic_laplacian(N, EPS)
Lp = cyclic_laplacian(N, EPS, KAPPA)
lam, lamp = np.linalg.eigvals(L), np.linalg.eigvals(Lp)

print("=" * 78)
print(f"Closed-loop stability, n = {N}, eps = {EPS}, kappa = {KAPPA}, beta = {BETA}")
print("=" * 78)
print(f"\n{'configuration':<28}{'max Re(s)':>14}{'# Re>=0':>12}   verdict")
print("-" * 78)

cases = {
    "RPAV, unpinned": (A_rpav(L, BETA), lam, BETA),
    "RPAV, pinned":   (A_rpav(Lp, BETA), lamp, BETA),
    "RPRV, unpinned": (A_rprv(L, L), lam, lam),
    "RPRV, pinned":   (A_rprv(Lp, Lp), lamp, lamp),
}
spectra = {}
for name, (A, lm, dl) in cases.items():
    spectra[name] = (report(name, A), lm, dl)

print("\nModal-factorization check  max |s_direct - s_modal|:")
for name, (s, lm, dl) in spectra.items():
    print(f"  {name:<26}{matched_max_dev(s, modal_roots(lm, dl)):.2e}")

print("\nUnpinned drift mode: lambda_0 = 0 gives s^2 + delta_0 s = 0,")
print("  so s = 0 always survives -> both unpinned A are singular, not Hurwitz.")
print("Pinned: smallest |lambda| =",
      f"{np.min(np.abs(lamp)):.6f}, so the drift root moves to")
for arch, d in (("RPAV", BETA), ("RPRV", np.min(np.abs(lamp)))):
    l0 = lamp[np.argmin(np.abs(lamp))]
    print(f"  {arch}: s = {np.roots([1.0, d, l0])}")

print(f"\n{'kappa':>8}{'RPAV max Re(s)':>18}{'RPRV max Re(s)':>18}")
for k in (0.0, 0.05, 0.1, 0.5, 1.0, 2.0):
    Lk = cyclic_laplacian(N, EPS, k)
    a1 = np.linalg.eigvals(A_rpav(Lk, BETA)).real.max()
    a2 = np.linalg.eigvals(A_rprv(Lk, Lk)).real.max()
    print(f"{k:>8.2f}{a1:>18.3e}{a2:>18.3e}")

print("\nFirst-order model A = -L:")
for name, l in (("unpinned", lam), ("pinned", lamp)):
    print(f"  {name:<10}max Re(-lambda) = {(-l.real).max():>12.3e}")

print("\n" + "=" * 78)
print("Table data: L_cyc eigenvalues sorted by real part (index = rank, not name)")
print("=" * 78)
lam_s = lam[np.argsort(lam.real, kind="stable")]
lamp_s = lamp[np.argsort(lamp.real, kind="stable")]
print(f"{'rank j':>8}{'unpinned':>28}{'pinned':>28}")
for j in (0, 1, 2, N - 1):
    print(f"{j:>8}{lam_s[j].real:>16.6f}{lam_s[j].imag:>+11.6f}i"
          f"{lamp_s[j].real:>16.6f}{lamp_s[j].imag:>+11.6f}i")

def fmt_complex(z, tol=1e-9):
    if abs(z.imag) < tol:
        return f"{z.real:.6f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.6f}{sign}{abs(z.imag):.6f}i"

os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, f"cyclic_laplacian_eigenvalues_n={N}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index_i", "unpinned", "pinned"])
    for j in range(N):
        writer.writerow([j + 1, fmt_complex(lam_s[j]), fmt_complex(lamp_s[j])])
print(f"\nSaved all {N} unpinned/pinned eigenvalues to "
      f"{os.path.join(REL_OUTPUT_DIR, os.path.basename(csv_path))}")