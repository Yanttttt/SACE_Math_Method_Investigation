import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
# 1. 读取图像与轮廓处理
# -----------------------------
NUM_POINTS = 5000  # 动画时点数可以小一点，加快速度
file = "flower.jpg"
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
img_inv = cv2.bitwise_not(img)
_, binary = cv2.threshold(img_inv, 130, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea).squeeze()

if not np.array_equal(contour[0], contour[-1]):
    contour = np.vstack([contour, contour[0]])

indices = np.linspace(0, len(contour) - 1, NUM_POINTS, dtype=int)
contour_sampled = contour[indices]

# 转复数表示
z = contour_sampled[:, 0] + 1j * contour_sampled[:, 1]

# -----------------------------
# 2. 傅里叶变换
# -----------------------------
Z = fft(z) / NUM_POINTS  # 归一化
N = len(Z)
freqs = np.fft.fftfreq(N)  # 对应频率
# 排序，低频到高频
indices_sorted = np.argsort(np.abs(freqs))
Z_sorted = Z[indices_sorted]
freqs_sorted = freqs[indices_sorted]

# -----------------------------
# 3. 动画设置
# -----------------------------
fig, ax = plt.subplots(figsize=(5,5))
ax.set_aspect('equal')
ax.set_xlim(np.min(np.real(z))-10, np.max(np.real(z))+10)
ax.set_ylim(-(np.max(np.imag(z))+10), -(np.min(np.imag(z))-10))
ax.grid(True)

# 绘制线段和轨迹
lines = [ax.plot([], [], color='blue')[0] for _ in range(10)]  # 最多10个可见线段
trace, = ax.plot([], [], color='red')  # 轨迹
points_x, points_y = [], []

# -----------------------------
# 4. 动画函数
# -----------------------------
def update(frame):
    t = frame / NUM_POINTS * 2 * np.pi  # 0到2π
    pos = 0+0j
    positions = [pos]

    # 依次叠加前10个最大的傅里叶系数
    for k in range(10):
        coef = Z_sorted[-(k+1)]
        freq = freqs_sorted[-(k+1)]
        pos += coef * np.exp(2j * np.pi * freq * frame)
        positions.append(pos)

    # 更新线段
    for i, line in enumerate(lines):
        if i < len(positions)-1:
            line.set_data([np.real(positions[i]), np.real(positions[i+1])],
                          [-np.imag(positions[i]), -np.imag(positions[i+1])])
        else:
            line.set_data([], [])

    # 更新轨迹
    points_x.append(np.real(positions[-1]))
    points_y.append(-np.imag(positions[-1]))
    trace.set_data(points_x, points_y)
    return lines + [trace]

# -----------------------------
# 5. 创建动画
# -----------------------------
anim = FuncAnimation(fig, update, frames=NUM_POINTS, interval=20, blit=True)
anim.save("fourier_epicycles.gif", writer=PillowWriter(fps=30))
plt.show()
