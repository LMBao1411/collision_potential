"""
Model 2: Second-Order RPAV Regression (two regressors)
    dv/dt = -alpha * x  -  beta * v  +  w
The estimated beta is the gain in the -beta*I block of
    A_RPAV = [[0, I], [-L_alpha, -beta*I]]

Model 2 is dv/dt = -alpha*x - beta*v + w, so the fitted
slopes are the NEGATIVES of alpha and beta:
    alpha = -params[1],   beta = -params[2]
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid

# EDR data 
time = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
v_speeding = np.array([30.83, 32.78, 34.72, 36.39, 38.06, 39.17, 40.56, 41.94, 43.06, 44.17, 45.00])  # (m/s)

# reference car maintains the speeding car's initial cruising speed
v_ref = np.full_like(v_speeding, 30.83)
v_rel = v_ref - v_speeding
# relative gap set up
gap = cumulative_trapezoid(v_rel, time, initial=0)

# acceleration via finite difference
accel = np.gradient(v_speeding, time)

# OLS Regression: dv/dt = -alpha*gap - beta*v + w
# Two regressors, column-stacked. sm.add_constant fits the disturbance w.
X = sm.add_constant(np.column_stack([gap, v_speeding]))
model = sm.OLS(accel, X)
results = model.fit()
w_hat, w_se = results.params[0], results.bse[0]
alpha_hat, alpha_se = -results.params[1], results.bse[1]
beta_hat, beta_se = -results.params[2], results.bse[2]

print("--- Second-Order RPAV Regression (Model 2) ---")
print("Regressor key:  x1 = gap (m),   x2 = v (m/s)")
print(results.summary())
print("\nExtracted Parameters:")
print(f"Alpha (\u03B1):       {alpha_hat:+.5f} \u00B1 {alpha_se:.5f}"
      f"   (p = {results.pvalues[1]:.4f})")
print(f"Beta  (\u03B2):       {beta_hat:+.5f} \u00B1 {beta_se:.5f}"
      f"   (p = {results.pvalues[2]:.4f})")
print(f"Disturbance (w): {w_hat:+.5f} \u00B1 {w_se:.5f}")
print(f"R-squared:       {results.rsquared:.4f}"
      f"   (adj {results.rsquared_adj:.4f})")

# plotting
fig = plt.figure(figsize=(12, 5.2))

# ---------- 3D PLOT --------
ax = fig.add_subplot(1, 2, 1, projection='3d')
gg = np.linspace(gap.min() - 2, gap.max() + 2, 14)
vv = np.linspace(v_speeding.min() - 0.5, v_speeding.max() + 0.5, 14)
G, V = np.meshgrid(gg, vv)
plane = results.params[0] + results.params[1] * G + results.params[2] * V
ax.plot_wireframe(G, V, plane, color='tab:red', lw=0.5, alpha=0.55)
zfloor = min(accel.min(), results.fittedvalues.min()) - 0.3
ax.plot(gap, v_speeding, zfloor, color='0.6', lw=1.2, ls='--', zorder=1)
ax.scatter(gap, v_speeding, np.full_like(gap, zfloor), color='0.6', s=12, depthshade=False, zorder=1)
ax.scatter(gap, v_speeding, accel, color='black', s=45, depthshade=False, zorder=5, label='measured $dv/dt$')
ax.scatter(gap, v_speeding, results.fittedvalues, facecolors='none', edgecolors='tab:red', s=45, lw=1.3, depthshade=False, zorder=5, label='fitted (on plane)')
ax.set_zlim(zfloor, accel.max() + 0.25)
ax.set_xlabel('gap $x$ (m)', labelpad=8)
ax.set_ylabel('speed $v$ (m/s)', labelpad=8)
ax.set_zlabel('$dv/dt$ (m/s$^2$)', labelpad=6)
ax.view_init(elev=20, azim=35)
ax.set_box_aspect((1.3, 1.3, 0.9))
ax.tick_params(labelsize=7)
ax.set_title('(a) Data and fitted plane', fontsize=10, pad=0)
ax.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(-0.08, 0.98), framealpha=0.9)

# --------------- scatter plot -----------
ax2 = fig.add_subplot(1, 2, 2)
sc = ax2.scatter(gap, v_speeding, c=accel, cmap='viridis', s=60, edgecolor='k', zorder=3)
m_, b_ = np.polyfit(gap, v_speeding, 1)
xs = np.array([gap.min(), gap.max()])
ax2.plot(xs, m_ * xs + b_, 'r--', lw=1.3, zorder=2)
ax2.set_xlabel('gap $x$ (m)')
ax2.set_ylabel('speed $v$ (m/s)')
ax2.set_title('(b) Regressors are nearly collinear', fontsize=10)
ax2.grid(True, linestyle=':')
ax2.legend(fontsize=8, loc='upper right')
plt.colorbar(sc, ax=ax2, label='$dv/dt$ (m/s$^2$)')
plt.tight_layout()
plt.show()