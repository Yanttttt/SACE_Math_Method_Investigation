import numpy as np
import matplotlib.pyplot as plt

def g_upper(x):
    y = np.zeros_like(x)
    mask1 = (x >= -1.319) & (x <= 0)
    mask2 = (x > 0) & (x <= 1.576)
    y[mask1] = (25/64)/(1.6)*(x[mask1]+2.56)**2 + np.log(0.08) + 2.2
    y[mask2] = np.log(0.1*(x[mask2]+0.8)) + 3.8
    return y

def g_lower(x):
    y = np.zeros_like(x)
    mask1 = (x >= -1.319) & (x <= -0.188)
    mask2 = (x > -0.188) & (x <= 1.576)
    y[mask1] = 0.05
    y[mask2] = 2.6/(1 + np.exp(-3.4*(x[mask2]-0.9)))
    return y

x_vals = np.linspace(-1.319, 1.576, 500)
y_upper = g_upper(x_vals)
y_lower = g_lower(x_vals)

plt.figure(figsize=(6,4))
plt.plot(x_vals, y_upper, label="g_upper(x)", color='blue')
plt.plot(x_vals, y_lower, label="g_lower(x)", color='red')
plt.fill_between(x_vals, y_lower, y_upper, color='gray', alpha=0.3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Upper and Lower Curves")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

np.save("D2/g_x_vals.npy", x_vals)
np.save("D2/g_y_upper.npy", y_upper)
np.save("D2/g_y_lower.npy", y_lower)