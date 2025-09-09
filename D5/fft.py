import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

NUM_POINTS = 250000
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

z = contour_sampled[:, 0] + 1j * contour_sampled[:, 1]
Z = fft(z)

plt.figure(figsize=(6, 3.5))
plt.scatter(np.real(z), -np.imag(z), color='black', s=0.2, label="Sampled Points")
plt.gca().set_aspect('equal')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.savefig(file + "_contour.png", bbox_inches='tight', pad_inches=0.1)

FOURIER_KEEP = 1000
Z_filtered = np.zeros_like(Z)
Z_filtered[:FOURIER_KEEP] = Z[:FOURIER_KEEP]
Z_filtered[-FOURIER_KEEP+1:] = Z[-FOURIER_KEEP+1:]

z_recon = ifft(Z_filtered)
np.savez("D5/fourier.npz", z_recon=z_recon, Z_filtered=Z_filtered)

plt.figure(figsize=(4, 4))
plt.plot(np.real(z_recon), -np.imag(z_recon), color='red')
plt.gca().set_aspect('equal')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

z_points = z_recon
dz = np.roll(z_points, -1) - z_points  # z_{k+1} - z_k
area = 0.5 * np.sum(np.imag(np.conj(z_points) * dz))
print("Area =", area)