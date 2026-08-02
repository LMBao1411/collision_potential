import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

time = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
v = np.array([30.83, 32.78, 34.72, 36.39, 38.06, 39.17, 40.56, 41.94, 43.06, 44.17, 45.00]) # (m/s)

# Calculate acceleration (dv/dt) using finite difference
dt = 0.5        # The time step between every intervals = 0.5 seconds
dvdt = np.gradient(v, dt)

# Calculate displacement (x) by integrating velocity over time
x = cumulative_trapezoid(v, time, initial=0)

# STATSMODELS OLS REGRESSION EXECUTION
def fit_and_evaluate_model(dependent_y, independent_x, model_name):
    """ Fits an OLS regression model and extracts performance metrics. """
    # sm.add_constant adds a column of 1s so the model fits an intercept (w)
    X = sm.add_constant(independent_x)
    model = sm.OLS(dependent_y, X)
    results = model.fit()
    
    # Extract coefficients and performance metrics
    w_hat = results.params[0]
    eps_hat = results.params[1]
    w_se = results.bse[0]
    eps_se = results.bse[1]
    
    print(f"--- Model: {model_name} ---")
    print(results.summary())
    print("\nExtracted Parameters:")
    print(f"Epsilon (\u03B5): {eps_hat:.5f} \u00B1 {eps_se:.5f}")
    print(f"Disturbance (w): {w_hat:.5f} \u00B1 {w_se:.5f}")
    print(f"R-squared: {results.rsquared:.4f}\n")
    
    return results, eps_hat, w_hat

# Model 0: dv/dt = \epsilon * v + w
res_0, eps_0, w_0 = fit_and_evaluate_model(dvdt, v, "dv/dt = \u03B5*v + w")

# Model 1: dx/dt = \epsilon * x + w (Note: dx/dt is velocity v)
res_1, eps_1, w_1 = fit_and_evaluate_model(v, x, "dx/dt = \u03B5*x + w")

# Model 2: dv/dt = \epsilon * x + w
res_2, eps_2, w_2 = fit_and_evaluate_model(dvdt, x, "dv/dt = \u03B5*x + w")

# PLOTTING REGRESSION FITS
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Common Plotting Function ensuring strict margin compliance
def format_regression_plot(ax, x_data, y_data, w_hat, eps_hat, x_label, y_label, title):
    ax.scatter(x_data, y_data, color='black', label='EDR Data')
    x_range = np.linspace(min(x_data), max(x_data), 100)
    ax.plot(x_range, eps_hat * x_range + w_hat, color='red', linestyle='--', label='OLS Fit')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=':')
    ax.legend(fontsize='small')
    
    # Strict border enforcement: No gap to the border of the graphs
    ax.margins(0)
    ax.set_xlim(min(x_data), max(x_data))
    
# Plot 1: dv/dt vs v
format_regression_plot(axs[0], v, dvdt, w_0, eps_0, 
                       'Velocity $v$ (m/s)', 'Acceleration $dv/dt$ (m/s$^2$)', 
                       'Model 0: $dv/dt = \epsilon v + w$')

# Plot 2: v vs x
format_regression_plot(axs[1], x, v, w_1, eps_1, 
                       'Displacement $x$ (m)', 'Velocity $v$ (m/s)', 
                       'Model 1: $dx/dt = \epsilon x + w$')

# Plot 3: dv/dt vs x
format_regression_plot(axs[2], x, dvdt, w_2, eps_2, 
                       'Displacement $x$ (m)', 'Acceleration $dv/dt$ (m/s$^2$)', 
                       'Model 2: $dv/dt = \epsilon x + w$')

plt.tight_layout()
plt.show()