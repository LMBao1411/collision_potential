import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

N = 50
EPSILON = 0.02
ALPHA = 6.20
KAPPA = 0.50
RHO = 1e-7     

PANEL_W, PANEL_H = 6.20, 2.70
LEGEND_H, XLABEL_H = 0.62, 0.06
BASE_FONT, LEGEND_FONT, TITLE_PAD = 10, 8, 8.0

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT,
    "ytick.labelsize": BASE_FONT,
    "legend.fontsize": BASE_FONT,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.05,
    "figure.constrained_layout.wspace": 0.06,
})

STYLES = {
    "sym_unpinned": dict(color="#0072BD", linestyle="-", marker="o",
                         markersize=3.1, markerfacecolor="#0072BD",
                         markeredgecolor="#0072BD", markeredgewidth=0.0,
                         label="Symmetric, unpinned"),
    "asy_unpinned": dict(color="#D95319", linestyle="-",
                         label="Asymmetric, unpinned"),
    "sym_pinned":   dict(color="#EDB120", linestyle=(0, (5, 2.5)),
                         label="Symmetric, pinned"),
    "asy_pinned":   dict(color="#7E2F8E", linestyle=(0, (1, 1.6)),
                         label="Asymmetric, pinned"),
}
WIDTHS = {"sym_unpinned": 0.8, "asy_unpinned": 1.0,
          "sym_pinned": 1.2, "asy_pinned": 1.4}


# topologies
def line_laplacian(n, eps, anchor=0.0):
    L = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            L[i, 0], L[i, 1] = 1.0 - eps, -1.0 + eps
        elif i == n - 1:
            L[i, n - 2], L[i, n - 1] = -1.0 - eps, 1.0 + eps
        else:
            L[i, i - 1], L[i, i], L[i, i + 1] = -1.0 - eps, 2.0, -1.0 + eps
    L[0, 0] += anchor
    return L


def cyclic_laplacian(n, eps, anchor=0.0):
    L = np.zeros((n, n))
    for i in range(n):
        L[i, (i - 1) % n] = -1.0 - eps
        L[i, i] = 2.0
        L[i, (i + 1) % n] = -1.0 + eps
    L[0, 0] += anchor
    return L


# dynamics
def h2_profile(n, eps, topology, pinned):
    """Per-gap H2 norm of the first order closed loop xdot = -L_alpha x + w."""
    base = line_laplacian if topology == "line" else cyclic_laplacian
    L_alpha = ALPHA * base(n, eps, 0.0)
    if pinned:
        L_alpha[0, 0] += KAPPA
    else:
        L_alpha = L_alpha + RHO * np.eye(n)   # unpinned: break lambda_1 = 0

    A = -L_alpha
    B = np.eye(n)
    W = la.solve_continuous_lyapunov(A, -B @ B.T)

    ngaps = n if topology == "cyc" else n - 1
    out = np.empty(ngaps)
    for k in range(ngaps):
        C = np.zeros((1, n))
        C[0, k] = 1.0
        C[0, (k + 1) % n] = -1.0
        out[k] = (C @ W @ C.T).item()
    return out


def profiles(n, eps, topology):
    return {
        "sym_unpinned": h2_profile(n, 0.0, topology, False),
        "asy_unpinned": h2_profile(n, eps, topology, False),
        "sym_pinned":   h2_profile(n, 0.0, topology, True),
        "asy_pinned":   h2_profile(n, eps, topology, True),
    }

# analytical
def ring_closed_form(n, eps, alpha=ALPHA):
    """First order counterpart of Proposition 3 on the unpinned ring.

    Mode m obeys xdot_m = -alpha*lambda_m x_m + w_m, whose stationary
    variance is 1/(2 alpha a_m); the gap output weights mode m by
    |1 - exp(i th_m)|^2 = a_m, so the a_m cancels and every mode
    contributes the same 1/(2 alpha).  Hence
        ||G_k||^2 = (1/n) sum_{m=2}^{n} 1/(2 alpha) = (n-1)/(2 n alpha),
    identical at every gap k and independent of eps."""
    th = 2.0 * np.pi * np.arange(n) / n
    a = 2.0 - 2.0 * np.cos(th)
    m = slice(1, n)
    return np.sum(a[m] / (2.0 * alpha * a[m])) / n

def draw_panel(ax, x, curves, title):
    step = max(1, len(x) // 12)
    for key, y in curves.items():
        style = dict(STYLES[key])
        if "marker" in style:
            style["markevery"] = step
        ax.plot(x, y, linewidth=WIDTHS[key], **style)
    ax.set_title(title, pad=TITLE_PAD)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fmt = ScalarFormatter(useOffset=True)
    fmt.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_size(BASE_FONT)
    ax.margins(x=0.02)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    ax.tick_params(width=0.6, length=2.5, pad=2.5)


def make_figure(n, eps, show_legend=True):
    strip = (LEGEND_H + XLABEL_H) if show_legend else 0.0
    height = PANEL_H + strip
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PANEL_W, height),
                                   layout="constrained",
                                   num=f"first order, n = {n}")
    if show_legend:
        frac = strip / height
        fig.get_layout_engine().set(rect=(0, frac, 1, 1 - frac))

    draw_panel(ax1, np.arange(1, n), profiles(n, eps, "line"), "Line")
    draw_panel(ax2, np.arange(n), profiles(n, eps, "cyc"), "Cyclic")

    supx = fig.supxlabel(r"Vehicle pair index $i$")
    fig.supylabel(r"$\|G_i\|_{\mathcal{H}_2}^2$")
    if show_legend:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", ncol=1, frameon=True,
                   fancybox=False, edgecolor="0.4", framealpha=1.0,
                   fontsize=LEGEND_FONT, borderpad=0.35, labelspacing=0.25,
                   handlelength=2.0, handletextpad=0.4, borderaxespad=0.0,
                   bbox_to_anchor=(0.995, 0.005))

    fig.canvas.draw()
    fig.set_layout_engine("none")
    b1, b2 = ax1.get_position(), ax2.get_position()
    supx.set_x(0.5 * (b1.x0 + b2.x1))
    if show_legend:
        supx.set_y(LEGEND_H / height)
    return fig


# run
if __name__ == "__main__":
    print("=" * 72)
    print(f"First order model, alpha = {ALPHA}, n = {N}, eps = {EPSILON}")
    print("=" * 72)
    for topo, name in (("line", "Line"), ("cyc", "Cyclic")):
        print(f"\n{name}")
        print(f"{'case':>16}{'min':>13}{'max':>13}{'mean':>13}"
              f"{'argmax':>9}{'max/min':>10}")
        for key, y in profiles(N, EPSILON, topo).items():
            print(f"{key:>16}{y.min():>13.5e}{y.max():>13.5e}"
                  f"{y.mean():>13.5e}{int(np.argmax(y)):>9}"
                  f"{y.max() / y.min():>10.2f}")

    print("\nUnpinned ring, closed form vs. numerical (first order)")
    for eps in (0.0, 0.02):
        num = h2_profile(N, eps, "cyc", False)
        ana = ring_closed_form(N, eps)
        print(f"  eps = {eps:<5}  analytical {ana:.8e}   numerical "
              f"{num.mean():.8e}   rel. err. "
              f"{abs(num.mean() - ana) / ana:.2e}")

    print("\nScale of the first order result against the second order one")
    print(f"  first order, worst gap on the ring : "
          f"{profiles(N, EPSILON, 'cyc')['asy_pinned'].max():.3e}")
    print(f"  1 / (2 alpha) reference level      : {1 / (2 * ALPHA):.3e}")
    print(f"  alpha = 1 would give               : "
          f"{ALPHA * profiles(N, EPSILON, 'cyc')['asy_pinned'].max():.3e}")

    print("\nSize dependence, unpinned ring (eps = 0.02)")
    print(f"{'n':>5}{'gap variance':>16}")
    for n in (3, 5, 10, 20, 50, 100, 200):
        print(f"{n:>5}{h2_profile(n, 0.02, 'cyc', False).mean():>16.6e}")

    make_figure(N, EPSILON, show_legend=False)
    plt.show()