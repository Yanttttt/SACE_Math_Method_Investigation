import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft 

NUM_POINTS = 1024

file= "badapple2.png"
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
img_inv=img.copy()
# img_inv = cv2.bitwise_not(img)  # 让黑区域变成白块（用于提取）
_, binary = cv2.threshold(img_inv, 200, 255, cv2.THRESH_BINARY)

# 提取外轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea).squeeze()

# 可视化检查


# -------- 闭合路径处理 --------
if not np.array_equal(contour[0], contour[-1]):
    contour = np.vstack([contour, contour[0]])  # 保证首尾闭合

# -------- 均匀采样 --------
# 等间距索引下采样：在完整路径上均匀取点
indices = np.linspace(0, len(contour) - 1, NUM_POINTS, dtype=int)
contour_sampled = contour[indices]

# -------- 转复数形式 --------
z = contour_sampled[:, 0] + 1j * contour_sampled[:, 1]

Z = fft(z)

plt.figure(figsize=(8, 8))
plt.scatter(np.real(z), -np.imag(z), color='black', s=0.2, label="Sampled Points")
plt.gca().set_aspect('equal')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.savefig(file+"_contour.png", bbox_inches='tight', pad_inches=0.1)

for FOURIER_KEEP in [2,10, 50, 100, 200, 1000]:

    # -------- 傅里叶变换与压缩 --------
    Z_filtered = np.zeros_like(Z)
    Z_filtered[:FOURIER_KEEP] = Z[:FOURIER_KEEP]
    Z_filtered[-FOURIER_KEEP+1:] = Z[-FOURIER_KEEP+1:]  # 保持共轭对称

    z_recon = ifft(Z_filtered)

    plt.figure(figsize=(8, 8))
    plt.plot(np.real(z_recon), -np.imag(z_recon), color='red')
    plt.gca().set_aspect('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(file+"_"+str(FOURIER_KEEP)+".png", bbox_inches='tight', pad_inches=0.1)
