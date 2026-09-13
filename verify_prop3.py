import numpy as np
import scipy.linalg as la

RHO = 1e-7
ALPHA = 1.0         
BETA = 1.0          

# topology
def cyclic_laplacian(n, eps):
    L = np.zeros((n, n))
    for i in range(n):
        L[i, (i - 1) % n] = -1.0 - eps
        L[i, i] = 2.0
        L[i, (i + 1) % n] = -1.0 + eps
    return L

def spectrum(n, eps):
    """Proposition 1: a_m and b_m for m = 1..n."""
    th = 2.0 * np.pi * np.arange(n) / n
    a = 2.0 - 2.0 * np.cos(th)
    b = -2.0 * eps * np.sin(th)
    return a, b


# analytical
def closed_form_rpav(n, eps, alpha=ALPHA, beta=BETA):
    """Eq. (closedform-rpav). The gain enters as L_alpha = alpha*L, so
    a_m -> alpha*a_m and b_m -> alpha*b_m; at alpha = 1 this is the
    proposition verbatim."""
    a, b = spectrum(n, eps)
    a, b = alpha * a, alpha * b
    m = slice(1, n)                       # drop the rigid-body mode m = 1
    return np.sum(a[m] * beta / (2.0 * (a[m] * beta ** 2 - b[m] ** 2))) / n


def closed_form_rprv(n, eps, alpha=ALPHA):
    """Eq. (closedform-rprv), with L_alpha = L_beta = alpha*L."""
    a, b = spectrum(n, eps)
    a, b = alpha * a, alpha * b
    m = slice(1, n)
    return np.sum(a[m] ** 2 /
                  (2.0 * (a[m] ** 3 - b[m] ** 2 * (1.0 - a[m])))) / n


# numerical
def h2_profile(n, eps, dynamics, alpha=ALPHA, beta=BETA):
    """Per-gap H2 norm on the unpinned ring, from the CALE."""
    L = cyclic_laplacian(n, eps)
    L_alpha = alpha * L + RHO * np.eye(n)     # L_alpha = alpha * L_cyc
    L_beta = alpha * L + RHO * np.eye(n)      # RPRV: L_beta = L_alpha
    Z, I = np.zeros((n, n)), np.eye(n)
    if dynamics == "RPAV":
        A = np.block([[Z, I], [-L_alpha, -beta * I]])
    else:
        A = np.block([[Z, I], [-L_alpha, -L_beta]])
    if np.max(np.linalg.eigvals(A).real) >= 0:
        return None                        # closed loop not Hurwitz
    B = np.block([[Z], [I]])
    W = la.solve_continuous_lyapunov(A, -B @ B.T)
    out = np.empty(n)
    for k in range(n):
        C = np.zeros((1, 2 * n))
        C[0, k], C[0, (k + 1) % n] = 1.0, -1.0
        out[k] = (C @ W @ C.T).item()
    return out


# e_crit
def eps_crit(n):
    """Proposition 2a. The radicand 1 - 4 sin^2(pi/n) is negative for n < 6
    and exactly zero at n = 6, so a finite critical offset exists only for
    n >= 7. Below that the ring is unconditionally stable in epsilon."""
    s = np.sin(np.pi / n)
    disc = 1.0 - 4.0 * s ** 2
    if disc <= 1e-12:
        return np.inf
    return 2.0 * s ** 2 / (np.cos(np.pi / n) * np.sqrt(disc))


# results
def table(dynamics, closed_form, cases):
    print(f"\n{dynamics}  (unpinned ring, rho = {RHO:g}, alpha = {ALPHA:g}"
          + (f", beta = {BETA:g})" if dynamics == "RPAV" else ")"))
    print(f"{'n':>5}{'eps':>8}{'analytical':>15}{'numerical (mean)':>19}"
          f"{'rel. err.':>13}{'spread over k':>15}")
    print("-" * 75)
    for n, eps in cases:
        num = h2_profile(n, eps, dynamics)
        ana = closed_form(n, eps)
        if num is None:
            print(f"{n:>5}{eps:>8.3f}{ana:>15.8f}"
                  f"{'not Hurwitz':>19}{'--':>13}{'--':>15}")
            continue
        mean = num.mean()
        spread = num.max() - num.min()
        rel = abs(mean - ana) / abs(ana)
        print(f"{n:>5}{eps:>8.3f}{ana:>15.8f}{mean:>19.8f}"
              f"{rel:>13.2e}{spread:>15.2e}")


if __name__ == "__main__":
    cases = [(3, 0.0), (3, 0.02), (10, 0.0), (10, 0.02), (10, 0.10),
             (20, 0.02), (50, 0.0), (50, 0.02), (50, 0.005), (100, 0.002)]

    print("=" * 75)
    print("Proposition 3: closed form vs. Lyapunov solution")
    print("=" * 75)
    table("RPAV", closed_form_rpav, cases)
    table("RPRV", closed_form_rprv, cases)

    print("\nRPRV stability limit, Proposition 2a, for reference")
    print(f"{'n':>5}{'eps_crit':>12}{'2 pi^2 / n^2':>15}")
    for n in (3, 6, 7, 10, 20, 50, 100):
        ec = eps_crit(n)
        shown = "none" if not np.isfinite(ec) else f"{ec:.5f}"
        print(f"{n:>5}{shown:>14}{2 * np.pi ** 2 / n ** 2:>15.5f}")

    print("\nSensitivity of the RPRV agreement to the regularization rho")
    print(f"{'rho':>10}{'n=10, eps=0.02':>18}{'n=50, eps=0.005':>19}")
    for rho in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        RHO_SAVE, globals()['RHO'] = RHO, rho
        row = []
        for n, eps in ((10, 0.02), (50, 0.005)):
            num = h2_profile(n, eps, "RPRV")
            row.append(abs(num.mean() - closed_form_rprv(n, eps))
                       / closed_form_rprv(n, eps))
        globals()['RHO'] = RHO_SAVE
        print(f"{rho:>10.0e}{row[0]:>18.2e}{row[1]:>19.2e}")

    print("\nSize dependence of the unpinned RPAV level (eps = 0.02)")
    print(f"{'n':>5}{'numerical':>14}{'(n-1)/(2 n beta)':>20}")
    for n in (3, 5, 10, 20, 50, 100, 200):
        num = h2_profile(n, 0.02, "RPAV")
        print(f"{n:>5}{num.mean():>14.8f}{(n - 1) / (2 * n * BETA):>20.8f}")