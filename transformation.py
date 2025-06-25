import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x-1)**2*(x-5)**2+6

theta= np.pi / 6

trans = {
        "rot": np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
        "scale": np.array([[4, 0], [0, 0.2]]),
        "shear": np.array([[1, 0.5], [0, 1]]),
        "none": np.array([[1, 0], [0, 1]])
    }
A=trans["shear"]
b = np.array([0, 0])

x_vals = np.linspace(1,5.5, 500)
f_vals = f(x_vals)
f2_vals = f(x_vals)-2

points = np.vstack((x_vals, f_vals))  # shape: (2, N)
points2 = np.vstack((x_vals, f2_vals))

points_covariant = A @ points + b[:, None]
points2_covariant = A @ points2 + b[:, None]

origin = np.array([[0], [0]])
i_hat = np.array([[5], [0]])
j_hat = np.array([[0], [5]])
i_trans = A @ i_hat
j_trans = A @ j_hat

plt.figure(figsize=(6, 6))

plt.plot(points[0], points[1], label="P(t)", color='blue')
plt.plot(points2[0], points2[1], label="P(t)", color='blue')
plt.plot(points_covariant[0], points_covariant[1], label="T(P(t))", color='green')
plt.plot(points2_covariant[0], points2_covariant[1], label="T(P(t))", color='green')

plt.quiver(*origin, *i_hat, angles='xy', scale_units='xy', scale=1, color='grey', label="i-hat")
plt.quiver(*origin, *j_hat, angles='xy', scale_units='xy', scale=1, color='grey', label="j-hat")
plt.quiver(*(origin+b[:, None]), *i_trans, angles='xy', scale_units='xy', scale=1, color='purple', label="Transformed i-hat")
plt.quiver(*(origin+b[:, None]), *j_trans, angles='xy', scale_units='xy', scale=1, color='orange', label="Transformed j-hat")

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("transformation_shear.png", dpi=300)
