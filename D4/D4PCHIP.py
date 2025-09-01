import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====== PCHIP 实现 ======
def pchip_slopes(h, delta):
    """计算 PCHIP 各节点的导数"""
    n = len(h) + 1
    d = np.zeros(n)

    # 端点导数
    d[0] = ((2*h[0] + h[1]) * delta[0] - h[0]*delta[1]) / (h[0] + h[1])
    d[-1] = ((2*h[-1] + h[-2]) * delta[-1] - h[-1]*delta[-2]) / (h[-1] + h[-2])

    # 中间节点
    for k in range(1, n-1):
        if delta[k-1] * delta[k] > 0:  # 单调一致
            d[k]=(delta[k-1]+delta[k])/2
        else:
            d[k] = 0.0
        # d[k]=(delta[k-1]+delta[k])/2
    return d

def pchip_interpolate(x, y, x_new):
    """分段 Hermite 插值"""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = pchip_slopes(h, delta)

    x_new = np.asarray(x_new)
    y_new = np.zeros_like(x_new)

    for i in range(n-1):
        idx = (x_new >= x[i]) & (x_new <= x[i+1])
        xi, xi1 = x[i], x[i+1]
        hi = h[i]
        t = (x_new[idx] - xi) / hi
        h00 = (1 + 2*t) * (1 - t)**2
        h10 = t * (1 - t)**2
        h01 = t**2 * (3 - 2*t)
        h11 = t**2 * (t - 1)
        y_new[idx] = (h00 * y[i] +
                      h10 * hi * d[i] +
                      h01 * y[i+1] +
                      h11 * hi * d[i+1])
    return y_new


# ====== 主程序 ======
def fit_image_boundaries(img_path, num_samples=10):
    # Step 1: 读图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("图片路径不对或文件不存在")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Step 2: 轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("没有找到轮廓")

    contour = max(contours, key=cv2.contourArea).squeeze()

    # Step 3: 上下边界
    contour = contour[np.argsort(contour[:,0])]
    x_vals = np.unique(contour[:,0])
    y_min, y_max = [], []
    for x in x_vals:
        ys = contour[contour[:,0] == x][:,1]
        y_min.append(np.min(ys))
        y_max.append(np.max(ys))
    x_vals = x_vals.astype(float)
    y_min = np.array(y_min, dtype=float)
    y_max = np.array(y_max, dtype=float)

    # Step 4: 采样
    sample_idx = np.linspace(0, len(x_vals)-1, num_samples, dtype=int)
    x_sample = x_vals[sample_idx]
    y_min_sample = y_min[sample_idx]
    y_max_sample = y_max[sample_idx]

    # Step 5: PCHIP 拟合
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 400)
    y_fit_min = pchip_interpolate(x_sample, y_min_sample, x_fit)
    y_fit_max = pchip_interpolate(x_sample, y_max_sample, x_fit)

    # Step 6: 绘制
    plt.figure(figsize=(6,4))
    plt.imshow(img, cmap='gray')
    plt.plot(x_sample, y_min_sample, 'go', label="upper sample points")
    plt.plot(x_sample, y_max_sample, 'bo', label="lower sample points")
    plt.plot(x_fit, y_fit_min, 'r-', label="upper boundary PCHIP")
    plt.plot(x_fit, y_fit_max, 'm-', label="lower boundary PCHIP")
    plt.legend()
    plt.tight_layout()
    plt.savefig("D4/PCHIP_.png")


# ====== 调用 ======
# fit_image_boundaries("D4/M.png", num_samples=30) # 23
# fit_image_boundaries("D4/M.png", num_samples=15) # 23
fit_image_boundaries("D4/M.png", num_samples=23) # 23