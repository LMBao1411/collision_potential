import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

N_VALUES = (3, 10, 20, 50) 
EPSILON = 0.02 
PIN_GAIN = 0.5
PANEL_W = 4.20 
PANEL_H = 2.25 
LEGEND_H = 0.42
XLABEL_H = 0.30 
BASE_FONT = 10   

YLABEL_PAD = 8.0  
XLABEL_PAD = 6.0  
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
    "sym_unpinned": dict(color="#0072BD", linestyle="-",            marker="o",
                         markersize=3.1, markerfacecolor="#0072BD",
                         markeredgecolor="#0072BD", markeredgewidth=0.0,
                         label="Symmetric, unpinned"),
    "asy_unpinned": dict(color="#D95319", linestyle="-",            label="Asymmetric, unpinned"),
    "sym_pinned":   dict(color="#EDB120", linestyle=(0, (5, 2.5)),  label="Symmetric, pinned"),
    "asy_pinned":   dict(color="#7E2F8E", linestyle=(0, (1, 1.6)),  label="Asymmetric, pinned"),
}
WIDTHS = {"sym_unpinned": 0.8, "asy_unpinned": 1.0,
          "sym_pinned": 1.2, "asy_pinned": 1.4}


def create_roundabout_laplacian(n: int, epsilon: float, pinning: float = 0.0) -> np.ndarray:
    """
    Constructs an n x n Laplacian matrix for a cyclic graph (roundabout) topology.
    Forward edges have weight 1 - epsilon, backward edges have weight 1 + epsilon.
    A pinning gain anchors the first node if specified, breaking perfect symmetry.
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
    L[0, 0] += pinning
    return L


def evaluate_cyclic_h2_profile(n: int, epsilon: float, dynamics: str, leader: bool) -> np.ndarray:
    """ Computes the spatial H2 norm (gap variance) profile for a cyclic platoon."""
    alpha_pin = PIN_GAIN if leader else 0.0
    beta_pin = PIN_GAIN if leader else 0.0

    L_alpha = create_roundabout_laplacian(n, epsilon, alpha_pin)

    # Regularization when the system lacks an absolute spatial anchor,
    # preventing the CALE solver from failing due to the zero eigenvalue
    if not leader:
        L_alpha += 1e-7 * np.eye(n)

    if dynamics == 'RPAV':
        # Absolute velocity damping (identity block)
        A = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-L_alpha,         -np.eye(n)]])
    elif dynamics == 'RPRV':
        # Relative velocity damping (velocity Laplacian block)
        L_beta = create_roundabout_laplacian(n, epsilon, beta_pin)
        if not leader:
            L_beta += 1e-7 * np.eye(n)
        A = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-L_alpha,         -L_beta]])
    else:
        raise ValueError("Invalid dynamics specified. Choose 'RPAV' or 'RPRV'.")

    # Disturbance input matrix B (acting exclusively on accelerations)
    B = np.block([[np.zeros((n, n))],
                  [np.eye(n)]])

    W_c = la.solve_continuous_lyapunov(A, -B @ B.T)
    h2_data = np.zeros(n)
    for i in range(n):
        C = np.zeros((1, 2 * n))
        C[0, i] = 1.0
        C[0, (i + 1) % n] = -1.0  # periodic wrap-around for the final gap
        h2_data[i] = (C @ W_c @ C.T).item()

    return h2_data

def profiles(n: int, epsilon: float, dynamics: str) -> dict:
    return {
        "sym_unpinned": evaluate_cyclic_h2_profile(n, 0.0,     dynamics, False),
        "asy_unpinned": evaluate_cyclic_h2_profile(n, epsilon, dynamics, False),
        "sym_pinned":   evaluate_cyclic_h2_profile(n, 0.0,     dynamics, True),
        "asy_pinned":   evaluate_cyclic_h2_profile(n, epsilon, dynamics, True),
    }

def draw_panel(ax, x, curves, title):
    step = max(1, len(x) // 25)
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
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.7)
    ax.tick_params(width=0.6, length=2.5, pad=2.5)


def make_figure(n: int, epsilon: float, show_legend: bool = False):
    x = np.arange(n)
    strip = (LEGEND_H + XLABEL_H) if show_legend else 0.0
    height = PANEL_H + strip
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PANEL_W, height), layout="constrained", num=f"n = {n}")
    if show_legend:
        frac = strip / height
        fig.get_layout_engine().set(rect=(0, frac, 1, 1 - frac))
    draw_panel(ax1, x, profiles(n, epsilon, 'RPAV'), 'RPAV')
    draw_panel(ax2, x, profiles(n, epsilon, 'RPRV'), 'RPRV')
    supx = fig.supxlabel(r'Vehicle pair index $i$')
    fig.supylabel(r'$\|G_k\|_{\mathcal{H}_2}^2$')
    if show_legend:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, handlelength=2.6, columnspacing=1.2, handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
    fig.canvas.draw()
    fig.set_layout_engine('none')
    b1, b2 = ax1.get_position(), ax2.get_position()
    supx.set_x(0.5 * (b1.x0 + b2.x1))
    if show_legend:
        supx.set_y(LEGEND_H / height)
    return fig


if __name__ == "__main__":
    for n in N_VALUES:
        make_figure(n, EPSILON, show_legend=False)
    plt.show()