import numpy as np
import matplotlib.pyplot as plt

# Define the smooth SoftPlus activation function
def softplus_activation_np(x, beta=50.0):
    return np.log(1 + np.exp(beta * x)) / beta

# Define the custom piecewise activation function
def piecewise_activation(x, d=0.05):
    return np.piecewise(
        x,
        [x <= 0, (x > 0) & (x < d), x >= d],
        [0, lambda x: x**2 / (2 * d), lambda x: x - d / 2]
    )

# Generate input data
x_values = np.linspace(-2, 3, 500)

# Calculate function values
softplus_values_np = softplus_activation_np(x_values)
piecewise_values_np = piecewise_activation(x_values)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, softplus_values_np, label="SoftPlus (beta=1)", color="blue")
plt.plot(x_values, piecewise_values_np, label="Piecewise Activation", color="red", linestyle="--")
plt.title("Comparison of SoftPlus and Custom Piecewise Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
