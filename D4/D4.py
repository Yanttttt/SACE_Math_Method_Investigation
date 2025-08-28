import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# === Step 1: 读取灰度图 ===
img = cv2.imread("M.png", cv2.IMREAD_GRAYSCALE)

# 二值化（假设封闭图形比背景暗）
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# === Step 2: 提取轮廓 ===
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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

# === Step 4: 拉格朗日插值拟合 ===
# 为避免高阶插值不稳定，可以只取一部分点
sample_idx = np.linspace(0, len(x_vals)-1, 15, dtype=int)  # 选15个采样点
x_sample = x_vals[sample_idx]

lagrange_min = lagrange(x_sample, y_min[sample_idx])
lagrange_max = lagrange(x_sample, y_max[sample_idx])

x_fit = np.linspace(x_vals.min(), x_vals.max(), 400)
y_fit_min = lagrange_min(x_fit)
y_fit_max = lagrange_max(x_fit)

# === Step 5: 绘制结果 ===
plt.figure(figsize=(8,6))
plt.imshow(img, cmap='gray')
plt.plot(x_vals, y_min, 'g.', label="下边界点")
plt.plot(x_vals, y_max, 'b.', label="上边界点")
plt.plot(x_fit, y_fit_min, 'r-', label="下边界插值")
plt.plot(x_fit, y_fit_max, 'm-', label="上边界插值")
plt.legend()
plt.title("封闭图形上下边界 + 拉格朗日插值")
plt.show()
