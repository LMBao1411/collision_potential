import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter

N = 50
EPSILON = 0.02
ALPHA = 6.20      # position coupling, calibrated from the EDR record
ALPHA_PIN = 0.50  # position anchoring gain at node 0
BETA_PIN = 0.50   # velocity anchoring gain at node 0 (RPRV only)
PANEL_W = 5.50
PANEL_H = 2.90
LEGEND_H = 0.62
XLABEL_H = 0.06
BASE_FONT = 15
LEGEND_FONT = 11
TITLE_PAD = 8.0

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
    "asy_unpinned": dict(color="#D95319", linestyle="-", label="Asymmetric, unpinned"),
    "sym_pinned":   dict(color="#EDB120", linestyle=(0, (5, 2.5)), label="Symmetric, pinned"),
    "asy_pinned":   dict(color="#7E2F8E", linestyle=(0, (1, 1.6)), label="Asymmetric, pinned"),
}
WIDTHS = {"sym_unpinned": 0.8, "asy_unpinned": 1.0,
          "sym_pinned": 1.2, "asy_pinned": 1.4}

# dynamics
def cyclic_laplacian(n: int, epsilon: float, anchor: float = 0.0) -> np.ndarray:
    """Circulant Laplacian for the roundabout. Forward edges carry 1 - epsilon,
    backward edges 1 + epsilon. `anchor` adds a self loop at node 0."""
    L = np.zeros((n, n))
    for i in range(n):
        L[i, (i - 1) % n] = -1.0 - epsilon
        L[i, i] = 2.0
        L[i, (i + 1) % n] = -1.0 + epsilon
    L[0, 0] += anchor
    return L


def h2_profile(n: int, epsilon: float, dynamics: str, pinned: bool) -> np.ndarray:
    """Per-gap H2 norm along the ring."""
    # calibrated closed loop of eq. (Acal): L_alpha <- alpha * L
    L_alpha = ALPHA * cyclic_laplacian(n, epsilon, ALPHA_PIN if pinned else 0.0)
    if not pinned:
        L_alpha += 1e-7 * np.eye(n)   # break the zero eigenvalue for the CALE solve

    if dynamics == "RPAV":
        A = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-L_alpha, -np.eye(n)]])
    elif dynamics == "RPRV":
        L_beta = cyclic_laplacian(n, epsilon, BETA_PIN if pinned else 0.0)
        if not pinned:
            L_beta += 1e-7 * np.eye(n)
        A = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-L_alpha, -L_beta]])
    else:
        raise ValueError("dynamics must be 'RPAV' or 'RPRV'")

    B = np.block([[np.zeros((n, n))], [np.eye(n)]])
    W = la.solve_continuous_lyapunov(A, -B @ B.T)

    out = np.zeros(n)
    for i in range(n):
        C = np.zeros((1, 2 * n))
        C[0, i] = 1.0
        C[0, (i + 1) % n] = -1.0      # periodic wrap-around on the last gap
        out[i] = (C @ W @ C.T).item()
    return out


def profiles(n: int, epsilon: float, dynamics: str) -> dict:
    return {
        "sym_unpinned": h2_profile(n, 0.0,     dynamics, False),
        "asy_unpinned": h2_profile(n, epsilon, dynamics, False),
        "sym_pinned":   h2_profile(n, 0.0,     dynamics, True),
        "asy_pinned":   h2_profile(n, epsilon, dynamics, True),
    }


# plotting
def draw_panel(ax, x, curves, title, yfmt="%.1f"):
    step = max(1, len(x) // 12)
    for key, y in curves.items():
        style = dict(STYLES[key])
        if "marker" in style:
            style["markevery"] = step
        ax.plot(x, y, linewidth=WIDTHS[key], **style)
    ax.set_title(title, pad=TITLE_PAD)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))
    ax.yaxis.get_offset_text().set_size(BASE_FONT)
    ax.margins(x=0.02)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    ax.tick_params(width=0.6, length=2.5, pad=2.5)


def make_figure(n: int, epsilon: float, show_legend: bool = False):
    x = np.arange(n)
    strip = (LEGEND_H + XLABEL_H) if show_legend else 0.0
    height = PANEL_H + strip
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PANEL_W, height),
                                   layout="constrained", num=f"n = {n}")
    if show_legend:
        frac = strip / height
        fig.get_layout_engine().set(rect=(0, frac, 1, 1 - frac))

    draw_panel(ax1, x, profiles(n, epsilon, "RPAV"), "Cyclic, RPAV", "%.4f")
    draw_panel(ax2, x, profiles(n, epsilon, "RPRV"), "Cyclic, RPRV")

    supx = fig.supxlabel(r"Vehicle pair index $i$")
    fig.supylabel(r"$\|G_k\|_{\mathcal{H}_2}^2$")
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


if __name__ == "__main__":
    make_figure(N, EPSILON, show_legend=False)
    plt.show()