import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def apply_affine(points, A=np.eye(2), b=np.zeros(2)):
    return A @ points + b[:, None]

def rot_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def monte_carlo_area(x_ring, y_ring, num_points=200000):
    polygon = Path(np.vstack([x_ring, y_ring]).T)
    xmin, xmax = x_ring.min(), x_ring.max()
    ymin, ymax = y_ring.min(), y_ring.max()
    xs = np.random.uniform(xmin, xmax, num_points)
    ys = np.random.uniform(ymin, ymax, num_points)
    points = np.vstack([xs, ys]).T
    inside = polygon.contains_points(points)
    rect_area = (xmax - xmin) * (ymax - ymin)
    area = rect_area * np.sum(inside) / num_points
    return area

transforms = {
    "D1": {"A": np.eye(2), "b": np.zeros(2)},
    "D2": {"A": rot_matrix(np.pi/11) @ np.array([[1.7, 0], [0, 1.7]]), "b": np.array([0.5, 1.7])},
    "D3": {"A": np.array([[0.04, 0], [0, 0.04]]), "b": np.array([-2, 1.3])},
    "D4": {"A": np.array([[1, 0.3], [0, 1]]) @ np.array([[0.045, 0], [0, 0.05]]), "b": np.array([2, 8])},
    "D4p": {"A": np.array([[1, -0.2], [0, 1]]) @ np.array([[0.05, 0], [0, 0.05]]), "b": np.array([-11, -1.5])},
    "D5": {"A": rot_matrix(-np.pi/2) @ np.array([[0.018, 0], [0, 0.018]]), "b": np.array([6.3, 1.7])}
}

plt.figure(figsize=(8,8))

# ====================== D1 ======================
D1 = np.load("D1/curves.npy", allow_pickle=True).item()
A, b = transforms["D1"]["A"], transforms["D1"]["b"]
keys = sorted(D1.keys(), key=lambda k: int(k.strip("P")))
total_area_D1 = 0.0
for i in range(0, len(keys), 2):
    outer_key, inner_key = keys[i], keys[i+1]
    outer = apply_affine(np.vstack(D1[outer_key]), A, b)
    inner = apply_affine(np.vstack(D1[inner_key]), A, b)
    x_ring = np.concatenate([outer[0], inner[0][::-1]])
    y_ring = np.concatenate([outer[1], inner[1][::-1]])
    plt.fill(x_ring, y_ring, color="blue", alpha=0.6, label=f"$D_1$" if i==0 else "")
    area = monte_carlo_area(x_ring, y_ring)
    print(f"D1 ring {i//2+1} area ≈ {area:.6f}")
    total_area_D1 += area
print(f"Total D1 area ≈ {total_area_D1:.6f}")

# ====================== D2 ======================
D2 = {
    'x_vals': np.load("D2/g_x_vals.npy"),
    'y_lower': np.load("D2/g_y_lower.npy"),
    'y_upper': np.load("D2/g_y_upper.npy")
}
A, b = transforms["D2"]["A"], transforms["D2"]["b"]
upper = apply_affine(np.vstack((D2['x_vals'], D2['y_upper'])), A, b)
lower = apply_affine(np.vstack((D2['x_vals'], D2['y_lower'])), A, b)
x_poly = np.concatenate([upper[0], lower[0][::-1]])
y_poly = np.concatenate([upper[1], lower[1][::-1]])
plt.fill(x_poly, y_poly, color="red", alpha=0.6, label=f"$D_2$")
area_D2 = monte_carlo_area(x_poly, y_poly)
print(f"Total D2 area ≈ {area_D2:.6f}")

# ====================== D3 ======================
D3 = np.load("D3/bezier.npz")
A, b = transforms["D3"]["A"], transforms["D3"]["b"]
all_pts = []
for key in D3.files:
    pts = apply_affine(D3[key].T, A, b)
    all_pts.append(pts.T)
all_pts = np.vstack(all_pts)
plt.fill(all_pts[:,0], all_pts[:,1], color="green", alpha=0.6, label=f"$D_3$")
area_D3 = monte_carlo_area(all_pts[:,0], all_pts[:,1])
print(f"Total D3 area ≈ {area_D3:.6f}")

# ====================== D4 ======================
D4 = {
    'x_vals': np.load("D4/x_vals.npy"),
    'y_upper': -np.load("D4/y_lower.npy"),
    'y_lower': -np.load("D4/y_upper.npy")
}
for key in ["D4", "D4p"]:
    A, b = transforms[key]["A"], transforms[key]["b"]
    upper = apply_affine(np.vstack((D4['x_vals'], D4['y_upper'])), A, b)
    lower = apply_affine(np.vstack((D4['x_vals'], D4['y_lower'])), A, b)
    x_poly = np.concatenate([upper[0], lower[0][::-1]])
    y_poly = np.concatenate([upper[1], lower[1][::-1]])
    plt.fill(x_poly, y_poly, color="purple" if key=="D4" else "grey", alpha=0.6, label=f"${key}$")
    area = monte_carlo_area(x_poly, y_poly)
    print(f"{key} area ≈ {area:.6f}")

# ====================== D5 ======================
D5 = np.load("D5/fourier.npz")
z_recon = D5["z_recon"]
A, b = transforms["D5"]["A"], transforms["D5"]["b"]
z_pts = apply_affine(np.vstack((np.real(z_recon), -np.imag(z_recon))), A, b)
plt.fill(z_pts[0], z_pts[1], color="orange", alpha=0.6, label=f"$D_5$")
area_D5 = monte_carlo_area(z_pts[0], z_pts[1])
print(f"Total D5 area ≈ {area_D5:.6f}")

# ====================== Finish ======================
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.savefig("all_domains.png", dpi=300, bbox_inches='tight')
plt.show()
