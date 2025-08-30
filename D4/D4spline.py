import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# === Step 1: 读取灰度图 ===
img = cv2.imread("D4/M.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("图片路径不对或文件不存在")

# 二值化（假设封闭图形比背景暗）
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# === Step 2: 提取轮廓 ===
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if not contours:
    raise ValueError("没有找到轮廓，请检查阈值或图像颜色方向")

# 找到最大轮廓
contour = max(contours, key=cv2.contourArea)
contour = contour.squeeze()  # (N,2) 形式

# === Step 3: 分割为上下边界 ===
# 轮廓点按 x 排序
contour = contour[np.argsort(contour[:,0])]

# 对于每个 x，取最小 y 和最大 y
x_vals = np.unique(contour[:,0])
y_min = []
y_max = []
for x in x_vals:
    ys = contour[contour[:,0] == x][:,1]
    y_min.append(np.min(ys))
    y_max.append(np.max(ys))

x_vals = x_vals.astype(float)
y_min = np.array(y_min, dtype=float)
y_max = np.array(y_max, dtype=float)

# === Step 4: 采样点 (减少点数避免震荡) ===
num_samples = 30  # 你可以调节这个参数，比如 6~20
sample_idx = np.linspace(0, len(x_vals)-1, num_samples, dtype=int)

x_sample = x_vals[sample_idx]
y_min_sample = y_min[sample_idx]
y_max_sample = y_max[sample_idx]

# === Step 5: 三次样条拟合（基于采样点） ===
cs_min = CubicSpline(x_sample, y_min_sample, bc_type="natural")
cs_max = CubicSpline(x_sample, y_max_sample, bc_type="natural")

x_fit = np.linspace(x_vals.min(), x_vals.max(), 400)
y_fit_min = cs_min(x_fit)
y_fit_max = cs_max(x_fit)

# === Step 6: 绘制结果 ===
plt.figure(figsize=(6,4))
plt.imshow(img, cmap='gray')

plt.plot(x_vals, y_min, 'g.', alpha=0.3)
plt.plot(x_vals, y_max, 'b.', alpha=0.3)

plt.plot(x_sample, y_min_sample, 'go', markersize=8, label="upper sample points")
plt.plot(x_sample, y_max_sample, 'bo', markersize=8, label="lower sample points")

plt.plot(x_fit, y_fit_min, 'r-', linewidth=2, label="upper boundary function")
plt.plot(x_fit, y_fit_max, 'm-', linewidth=2, label="lower boundary function")

plt.legend()
plt.show()
