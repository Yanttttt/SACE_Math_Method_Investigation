import numpy as np
import matplotlib.pyplot as plt

# 原函数定义
def f(x):
    return (x-1)**2*(x-5)**2+6

# 仿射变换 Ax + b

theta= np.pi / 6  # 旋转角度

trans = {
        "rot": np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
        "scale": np.array([[2, 0], [0, 1]]),
        "shear": np.array([[1, 0.5], [0, 1]]),
        "none": np.array([[1, 0], [0, 1]])
    }
A=trans["none"]
b = np.array([0, -2])

# 定义绘图区间
x_vals = np.linspace(1,5.5, 500)
f_vals = f(x_vals)

points = np.vstack((x_vals, f_vals))  # shape: (2, N)

# --- 2. 协变：变换图像点 (x, f(x)) → A(x, f(x)) + b ---
points_covariant = A @ points + b[:, None]

# --- 4. 坐标轴变换可视化（显示单位坐标轴经 A 变换后的方向） ---
origin = np.array([[0], [0]])
i_hat = np.array([[5], [0]])
j_hat = np.array([[0], [5]])
i_trans = A @ i_hat
j_trans = A @ j_hat

# --- 绘图 ---
plt.figure(figsize=(6, 6))

# 原图像
plt.plot(points[0], points[1], label="P(t)", color='blue')

# 协变图像（图像整体变换）
plt.plot(points_covariant[0], points_covariant[1], label="T(P(t))", color='green')

# 显示仿射变换后的坐标轴方向
plt.quiver(*origin, *i_hat, angles='xy', scale_units='xy', scale=1, color='grey', label="i-hat")
plt.quiver(*origin, *j_hat, angles='xy', scale_units='xy', scale=1, color='grey', label="j-hat")

plt.quiver(*(origin+b[:, None]), *i_trans, angles='xy', scale_units='xy', scale=1, color='purple', label="Transformed i-hat")
plt.quiver(*(origin+b[:, None]), *j_trans, angles='xy', scale_units='xy', scale=1, color='orange', label="Transformed j-hat")

# 设置图像属性
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("transformation_translate.png", dpi=300)
