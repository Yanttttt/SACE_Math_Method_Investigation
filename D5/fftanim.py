# fftanim_multi.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft

NUM_POINTS = 2000
FOURIER_LIST = [2, 10, 200, 1000] 
FRAMES = 800
INTERVAL_MS = 20
USE_BLIT = False
OUT_FILE = "D5/epicycles.mp4"
FPS = 30

# ---------------- load + contour ----------------
img = cv2.imread(FILE, cv2.IMREAD_GRAYSCALE)

img_inv = cv2.bitwise_not(img)
_, binary = cv2.threshold(img_inv, 130, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contour = max(contours, key=cv2.contourArea).squeeze()
if contour.ndim == 1 and contour.size == 2:
    contour = contour[np.newaxis, :]
if not np.array_equal(contour[0], contour[-1]):
    contour = np.vstack([contour, contour[0]])

indices = np.linspace(0, len(contour) - 1, NUM_POINTS, dtype=int)
contour_sampled = contour[indices]

z = contour_sampled[:, 0] + 1j * (-contour_sampled[:, 1])
z = z - z.mean()
span_x = z.real.max() - z.real.min()
span_y = z.imag.max() - z.imag.min()
max_span = max(span_x, span_y)
scale = 2.0 / max_span if max_span != 0 else 1.0
z = z * scale
N = len(z)

Z = fft(z) / N
all_coeffs = []
for k, c in enumerate(Z):
    freq = k if k <= N // 2 else k - N
    all_coeffs.append((freq, c))
all_coeffs.sort(key=lambda x: -abs(x[1]))

# ---------------- figure and subplots ----------------
nplots = len(FOURIER_LIST)
cols = 2
rows = (nplots + 1) // 2
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
axes = axes.flatten()

pad = 0.3
x_min, x_max = z.real.min(), z.real.max()
y_min, y_max = z.imag.min(), z.imag.max()

plots_data = []

for idx, keep in enumerate(FOURIER_LIST):
    ax = axes[idx]
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_title(f"{keep} terms", fontsize=14)

    coeffs = all_coeffs[:keep]
    vector_lines = [ax.plot([], [], lw=1, alpha=0.9)[0] for _ in coeffs]
    trace_line, = ax.plot([], [], lw=1.5)
    endpoint_dot, = ax.plot([], [], 'o', ms=4)
    trace = []

    plots_data.append({
        "coeffs": coeffs,
        "vector_lines": vector_lines,
        "trace_line": trace_line,
        "endpoint_dot": endpoint_dot,
        "trace": trace
    })

for j in range(nplots, len(axes)):
    axes[j].axis("off")

# ---------------- animation functions ----------------
def init():
    artists = []
    for pd in plots_data:
        pd["trace"].clear()
        for vl in pd["vector_lines"]:
            vl.set_data([], [])
        pd["trace_line"].set_data([], [])
        pd["endpoint_dot"].set_data([], [])
        artists += [pd["trace_line"], pd["endpoint_dot"]] + pd["vector_lines"]
    return artists

def update(frame):
    t = (frame % FRAMES) / FRAMES
    artists = []
    for pd in plots_data:
        x, y = 0.0, 0.0
        for i, (freq, c) in enumerate(pd["coeffs"]):
            angle = 2 * np.pi * freq * t + np.angle(c)
            dx = abs(c) * np.cos(angle)
            dy = abs(c) * np.sin(angle)
            nx, ny = x + dx, y + dy
            pd["vector_lines"][i].set_data([x, nx], [y, ny])
            x, y = nx, ny

        pd["trace"].append((x, y))
        if len(pd["trace"]) > N:
            pd["trace"].pop(0)

        tx = [p[0] for p in pd["trace"]]
        ty = [p[1] for p in pd["trace"]]
        pd["trace_line"].set_data(tx, ty)
        pd["endpoint_dot"].set_data([x], [y])
        artists += [pd["trace_line"], pd["endpoint_dot"]] + pd["vector_lines"]
    return artists

ani = FuncAnimation(
    fig, update, frames=range(FRAMES), init_func=init,
    blit=USE_BLIT, interval=INTERVAL_MS, repeat=True
)

ani.save(OUT_FILE, writer="ffmpeg", fps=FPS)
