import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return (x-1)**2*(x-5)**2+6

def f2(x):
    return (x-1)**2*(x-5)**2+4

x_vals = np.linspace(1,5.5, 500)
f1_vals = f1(x_vals)
f2_vals = f2(x_vals)

points1 = np.vstack((x_vals, f1_vals))
points2 = np.vstack((x_vals, f2_vals))

plt.plot(points1[0], points1[1], label=r"$f_1(x)$", color='blue')
plt.plot(points2[0], points2[1], label=r"$f_2(x)$", color='green')
plt.fill_between(x_vals, f1_vals, f2_vals, where=f1_vals >= f2_vals, color='red', alpha=0.3, label=r"$D_0$")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("original.png", dpi=300)