import os
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

REL_OUTPUT_DIR = "model1_OLS"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), REL_OUTPUT_DIR)

BASE_FONT, LEGEND_FONT = 14, 12

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT,
    "ytick.labelsize": BASE_FONT,
    "legend.fontsize": LEGEND_FONT,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
})

V0 = 45.00                 # (m/s)
dt = 0.01                  # (s), 100 Hz
time = np.arange(20) * dt  # 0 to 190 ms

dv = np.array([  0.00,  -1.67,  -3.33,  -8.61, -14.72,
               -17.50, -19.72, -20.83, -22.50, -23.61,
               -24.44, -25.28, -25.83, -26.11, -26.67,
               -26.94, -26.94, -27.22, -27.22, -27.22])   # (m/s)
v = V0 + dv               # absolute speed (m/s)

# Crush displacement by the trapezoidal rule, origin at t0
x = cumulative_trapezoid(v, time, initial=0)    # (m), spans 4.72 m

# dx/dt = -alpha * x + w
results = sm.OLS(v, sm.add_constant(x)).fit()

w_hat, slope = results.params[0], results.params[1]
w_se, slope_se = results.bse[0], results.bse[1]
alpha_hat, alpha_se = -slope, slope_se

print("First order fit: dx/dt = -\u03B1*x + w  (crash-pulse segment)")
print(results.summary())
print("\nExtracted Parameters:")
print(f"Alpha (\u03B1): {alpha_hat:.4f} \u00B1 {alpha_se:.4f}")
print(f"Disturbance (w): {w_hat:.4f} \u00B1 {w_se:.4f}")
print(f"R-squared: {results.rsquared:.4f}")
ALPHA_GAIN = abs(alpha_hat)
print(f"\nALPHA_GAIN for L_alpha <- alpha*L : {ALPHA_GAIN:.4f}")
print(f"Crush displacement span: {x[-1]:.3f} m")

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.scatter(x, v, c='tab:red', s=18, marker='s', zorder=3,
           label='crash pulses')
xr = np.linspace(x.min(), x.max(), 200)
ax.plot(xr, slope * xr + w_hat, 'k--', lw=1.0, label='OLS fitting')
ax.set_xlabel('Reconstructed displacement $x$ (m), from $t_0$')
ax.set_ylabel('$dx/dt = v$ (m/s)')
ax.set_title('Crash-pulse OLS fitting')
ax.grid(True, linestyle=':', linewidth=0.5)
ax.legend(fontsize=LEGEND_FONT, borderpad=0.35, labelspacing=0.25,
          handletextpad=0.4)
ax.margins(0)
plt.tight_layout(pad=0.4)

os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUTPUT_DIR, "crash_pulse_ols_fit.png"), dpi=200, bbox_inches="tight")
print(f"\nSaved figure to {REL_OUTPUT_DIR}")

plt.show()