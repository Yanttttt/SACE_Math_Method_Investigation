import numpy as np
import matplotlib.pyplot as plt

def apply_affine(points, A=np.eye(2), b=np.zeros(2)):
    return A @ points + b[:, None]

def rot_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

transforms = {
    "D1": {"A": np.array([[1, 0.3], [0, 1]]), "b": np.array([0, 0])},
    "D2": {"A": rot_matrix(np.pi/11) @ np.array([[1.7, 0], [0, 1.7]]), "b": np.array([0.5, 1.7])},
    "D3": {"A": np.array([[0.04, 0], [0, 0.04]]), "b": np.array([-2, 1.3])},
    "D4": {"A": np.array([[1, 0.3], [0, 1]]) @ np.array([[0.045, 0], [0, 0.05]]), "b": np.array([2, 8])},
    "D4p": {"A": np.array([[1, -0.2], [0, 1]]) @ np.array([[0.05, 0], [0, 0.05]]) , "b": np.array([-11, -1.5])},
    "D5": {"A": rot_matrix(-np.pi/2) @ np.array([[0.018, 0], [0, 0.018]]), "b": np.array([6.3, 1.7])}
}

plt.figure(figsize=(6,6))

# ===================== D1 =======================

D1 = np.load("D1/curves.npy", allow_pickle=True).item()
colors = ["blue","lightblue","green","lightgreen","yellow","lightyellow","red","pink"]

A, b = transforms["D1"]["A"], transforms["D1"]["b"]
for i, key in enumerate(D1.keys()):
    pts=np.vstack(D1[key])
    # pts = apply_affine(np.vstack(D1[key]), A, b)
    plt.plot(pts[0], pts[1], label=key, color=colors[i])

# ======================D2=======================

D2 = {
    'x_vals': np.load("D2/g_x_vals.npy"),
    'y_lower': np.load("D2/g_y_lower.npy"),
    'y_upper': np.load("D2/g_y_upper.npy")
}

A, b = transforms["D2"]["A"], transforms["D2"]["b"]
x = apply_affine(np.vstack((D2['x_vals'], D2['y_upper'])), A, b)
y = apply_affine(np.vstack((D2['x_vals'], D2['y_lower'])), A, b)
plt.plot(x[0], x[1], 'm-', label="D2 Upper Boundary")
plt.plot(y[0], y[1], 'r-', label="D2 Lower Boundary")

#===================== D3 ===================

D3 = np.load("D3/bezier.npz")
A, b = transforms["D3"]["A"], transforms["D3"]["b"]
for i, key in enumerate(D3.files, start=1):
    pts = apply_affine(D3[key].T, A, b)
    plt.plot(pts[0], pts[1], linewidth=2, label=f"Curve {i}")

# ======================D4=======================

D4 = {
    'x_vals': np.load("D4/x_vals.npy"),
    'y_upper': -np.load("D4/y_lower.npy"),
    'y_lower': -np.load("D4/y_upper.npy")
}

A, b = transforms["D4"]["A"], transforms["D4"]["b"]
upper = apply_affine(np.vstack((D4['x_vals'], D4['y_upper'])), A, b)
lower = apply_affine(np.vstack((D4['x_vals'], D4['y_lower'])), A, b)
plt.plot(upper[0], upper[1], 'm-', label="D4 Upper Boundary")
plt.plot(lower[0], lower[1], 'r-', label="D4 Lower Boundary")
# plt.fill_between(upper[0], lower[1], upper[1], color='pink', alpha=0.3)

A, b = transforms["D4p"]["A"], transforms["D4p"]["b"]
upperp = apply_affine(np.vstack((D4['x_vals'], D4['y_upper'])), A, b)
lowerp = apply_affine(np.vstack((D4['x_vals'], D4['y_lower'])), A, b)
plt.plot(upperp[0], upperp[1], 'm-', label="D4' Upper Boundary")
plt.plot(lowerp[0], lowerp[1], 'r-', label="D4' Lower Boundary")

# ======================D5=======================

D5 = np.load("D5/fourier.npz")
z_recon = D5["z_recon"]
A, b = transforms["D5"]["A"], transforms["D5"]["b"]
z_pts = apply_affine(np.vstack((np.real(z_recon), -np.imag(z_recon))), A, b)
plt.plot(z_pts[0], z_pts[1], color='red', label="D5 Fourier")

# =============================================================================

plt.grid(True)  
plt.axis('equal')
# plt.legend();
plt.tight_layout()
# plt.show()
plt.savefig("assemble_all.png", dpi=300, bbox_inches='tight')


