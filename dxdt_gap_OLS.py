import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# 1. EDR Data Setup
time = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
# Velocity IS dx/dt
v_speeding = np.array([30.83, 32.78, 34.72, 36.39, 38.06, 39.17, 40.56, 41.94, 43.06, 44.17, 45.00]) # (m/s)

# 2. Simulated Lead Car (Reference) Setup
# Assume the reference car maintains the speeding car's initial cruising speed of 30.83 m/s
v_ref = np.full_like(v_speeding, 30.83) 

# Relative velocity (Reference speed - Speeding car speed)
# Since the speeding car is accelerating, v_rel becomes increasingly negative (gap is shrinking)
v_rel = v_ref - v_speeding 

# 3. Integrate relative velocity to get the Gap Variance
# We assume the gap error starts at 0 meters at t = -5.0
gap = cumulative_trapezoid(v_rel, time, initial=0)

# 4. OLS Regression: dx/dt = \epsilon * gap + w
# sm.add_constant adds a column of 1s to fit the disturbance intercept (w)
X = sm.add_constant(gap)
# Using v_speeding (dx/dt) as the dependent Y variable
model = sm.OLS(v_speeding, X)
results = model.fit()

w_hat = results.params[0]
eps_hat = results.params[1]

print("--- First-Order Simulated Lead Car Gap Regression ---")
print(results.summary())
print(f"\nExtracted \u03B5 (Slope): {eps_hat:.5f}")
print(f"Extracted w (Intercept): {w_hat:.5f}")

# 5. Plotting
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(gap, v_speeding, color='black', label='Synthesized EDR Data')

gap_range = np.linspace(min(gap), max(gap), 100)
ax.plot(gap_range, eps_hat * gap_range + w_hat, color='red', linestyle='--', label='OLS Fit')

ax.set_xlabel('Relative Gap Error (m)')
ax.set_ylabel('Speeding Car Velocity $dx/dt$ (m/s)')
ax.set_title('First-Order Simulated Lead Car Model: $dx/dt = \epsilon \cdot gap + w$')
ax.grid(True, linestyle=':')
ax.legend(fontsize='small')

# Strict border enforcement: No gap to the border of the graphs
ax.margins(0)
ax.set_xlim(min(gap), max(gap))

plt.tight_layout()
plt.show()