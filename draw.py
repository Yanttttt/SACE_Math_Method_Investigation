import numpy as np
import matplotlib.pyplot as plt

def apply_affine(points, A=np.eye(2), b=np.zeros(2)):
    return A @ points + b[:, None]

def rot_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

transforms = {
    "D1": {"A": np.array([[1, 0], [0, 1]]), "b": np.array([0, 0])},
    "D2": {"A": rot_matrix(np.pi/11) @ np.array([[1.7, 0], [0, 1.7]]), "b": np.array([0.5, 1.7])},
    "D3": {"A": np.array([[0.04, 0], [0, 0.04]]), "b": np.array([-2, 1.3])},
    "D4": {"A": np.array([[1, 0.3], [0, 1]]) @ np.array([[0.045, 0], [0, 0.05]]), "b": np.array([2, 8])},
    "D4p": {"A": np.array([[1, -0.2], [0, 1]]) @ np.array([[0.05, 0], [0, 0.05]]) , "b": np.array([-11, -1.5])},
    "D5": {"A": rot_matrix(-np.pi/2) @ np.array([[0.018, 0], [0, 0.018]]), "b": np.array([6.3, 1.7])}
}

plt.figure(figsize=(6,6))

# ===================== D1 =======================

D1 = np.load("D1/curves.npy", allow_pickle=True).item()
A, b = transforms["D1"]["A"], transforms["D1"]["b"]

keys = sorted(D1.keys(), key=lambda k: int(k.strip("P")))

for i in range(0, len(keys), 2):
    outer_key = keys[i]
    inner_key = keys[i+1]
    outer = apply_affine(np.vstack(D1[outer_key]), A, b)
    inner = apply_affine(np.vstack(D1[inner_key]), A, b)

    x_ring = np.concatenate([outer[0], inner[0][::-1]])
    y_ring = np.concatenate([outer[1], inner[1][::-1]])
    if i == 0:
        plt.fill(x_ring, y_ring, color="blue", alpha=0.6, label=f"$D_1$")
    # else:
    #     plt.fill(x_ring, y_ring, color="blue", alpha=0.6)
# # ======================D2=======================

# D2 = {
#     'x_vals': np.load("D2/g_x_vals.npy"),
#     'y_lower': np.load("D2/g_y_lower.npy"),
#     'y_upper': np.load("D2/g_y_upper.npy")
# }
# A, b = transforms["D2"]["A"], transforms["D2"]["b"]
# upper = apply_affine(np.vstack((D2['x_vals'], D2['y_upper'])), A, b)
# lower = apply_affine(np.vstack((D2['x_vals'], D2['y_lower'])), A, b)
# x_poly = np.concatenate([upper[0], lower[0][::-1]])
# y_poly = np.concatenate([upper[1], lower[1][::-1]])
# plt.fill(x_poly, y_poly, color="red", alpha=0.6, label=f"$D_2$")

# #===================== D3 ===================

# D3 = np.load("D3/bezier.npz")
# A, b = transforms["D3"]["A"], transforms["D3"]["b"]

# all_pts = []
# for key in D3.files:
#     pts = apply_affine(D3[key].T, A, b)
#     all_pts.append(pts.T)  # (N, 2)

# all_pts = np.vstack(all_pts)
# plt.fill(all_pts[:,0], all_pts[:,1], color="green", alpha=0.6, label=f"$D_3$")

# # ======================D4=======================

# D4 = {
#     'x_vals': np.load("D4/x_vals.npy"),
#     'y_upper': -np.load("D4/y_lower.npy"),
#     'y_lower': -np.load("D4/y_upper.npy")
# }

# key="D4"

# A, b = transforms[key]["A"], transforms[key]["b"]
# upper = apply_affine(np.vstack((D4['x_vals'], D4['y_upper'])), A, b)
# lower = apply_affine(np.vstack((D4['x_vals'], D4['y_lower'])), A, b)
# x_poly = np.concatenate([upper[0], lower[0][::-1]])
# y_poly = np.concatenate([upper[1], lower[1][::-1]])
# plt.fill(x_poly, y_poly, color="purple", alpha=0.6, label=f"$D_4$")

# key="D4p"
# A, b = transforms[key]["A"], transforms[key]["b"]
# upper = apply_affine(np.vstack((D4['x_vals'], D4['y_upper'])), A, b)
# lower = apply_affine(np.vstack((D4['x_vals'], D4['y_lower'])), A, b)
# x_poly = np.concatenate([upper[0], lower[0][::-1]])
# y_poly = np.concatenate([upper[1], lower[1][::-1]])
# plt.fill(x_poly, y_poly, color="grey", alpha=0.6, label=f"$D_4^'$")

# # ======================D5=======================

# D5 = np.load("D5/fourier.npz")
# z_recon = D5["z_recon"]
# A, b = transforms["D5"]["A"], transforms["D5"]["b"]
# z_pts = apply_affine(np.vstack((np.real(z_recon), -np.imag(z_recon))), A, b)

# plt.fill(z_pts[0], z_pts[1], color="orange", alpha=0.6, label=f"$D_5$")

# # =============================================================================

plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.savefig("D1_chunk.png", dpi=300, bbox_inches='tight')
