import cv2
import numpy as np
import matplotlib.pyplot as plt

def pchip_slopes(h, delta):
    n = len(h) + 1
    d = np.zeros(n)
    d[0] = ((2*h[0] + h[1])*delta[0] - h[0]*delta[1]) / (h[0] + h[1])
    d[-1] = ((2*h[-1] + h[-2])*delta[-1] - h[-1]*delta[-2]) / (h[-1] + h[-2])
    for k in range(1, n-1):
        if delta[k-1]*delta[k] > 0:
            d[k] = (delta[k-1] + delta[k])/2
        else:
            d[k] = 0.0
    return d

def hermite_formulas(i, xi, yi, xi1, yi1, di, di1):
    h_val = xi1 - xi
    hermite_expr = (
        f"p_{i} (x)=h_(00) (t)y_{i}+h_(10) (t) h d_{i}+"
        f"h_(01) (t)y_{i+1}+h_(11) (t)y_{i+1} (h={h_val:.4f}, t=(x-{xi:.4f})/h)"
    )
    a = (2*(yi - yi1) + h_val*(di + di1)) / (h_val**3)
    b = (3*(yi1 - yi) - h_val*(2*di + di1)) / (h_val**2)
    c = di
    d_val = yi

    def fmt(num, first=False):
        if first:
            return f"{num:.6f}" if num >= 0 else f"{num:.6f}"
        return f"+{num:.6f}" if num >=0 else f"{num:.6f}"

    poly_expr = f"={fmt(a, first=True)}x^3{fmt(b)}x^2{fmt(c)}x{fmt(d_val)}"
    return f"{hermite_expr}{poly_expr}"

def pchip_interpolate(x, y, x_new):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y)/h
    d = pchip_slopes(h, delta)
    x_new = np.asarray(x_new)
    y_new = np.zeros_like(x_new)
    for i in range(n-1):
        idx = (x_new >= x[i]) & (x_new <= x[i+1])
        xi, xi1 = x[i], x[i+1]
        hi = h[i]
        t = (x_new[idx]-xi)/hi
        h00 = (1+2*t)*(1-t)**2
        h10 = t*(1-t)**2
        h01 = t**2*(3-2*t)
        h11 = t**2*(t-1)
        y_new[idx] = h00*y[i] + h10*hi*d[i] + h01*y[i+1] + h11*hi*d[i+1]
    return y_new, d

def fit_image_boundaries(img_path, num_samples=23, return_arrays=True):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")

    contour = max(contours, key=cv2.contourArea).squeeze()
    contour = contour[np.argsort(contour[:,0])]
    x_vals = np.unique(contour[:,0])
    
    y_min = [np.min(contour[contour[:,0]==x][:,1]) for x in x_vals]
    y_max = [np.max(contour[contour[:,0]==x][:,1]) for x in x_vals]

    x_vals = x_vals.astype(float)
    y_min = np.array(y_min, dtype=float)
    y_max = np.array(y_max, dtype=float)

    sample_idx = np.linspace(0, len(x_vals)-1, num_samples, dtype=int)
    x_sample = x_vals[sample_idx]
    y_min_sample = y_min[sample_idx]
    y_max_sample = y_max[sample_idx]

    y_min_fit, d_min = pchip_interpolate(x_sample, y_min_sample, x_vals)
    y_max_fit, d_max = pchip_interpolate(x_sample, y_max_sample, x_vals)

    print("=== Sample Points ===")
    print("Upper: " + ",".join([f"({x_sample[i]:.4f},{y_max_sample[i]:.4f})" for i in range(len(x_sample))]))
    print("Lower: " + ",".join([f"({x_sample[i]:.4f},{y_min_sample[i]:.4f})" for i in range(len(x_sample))]))

    print("\n=== Piecewise PCHIP Polynomials (Upper Boundary) ===")
    for i in range(1, len(x_sample)):
        formula = hermite_formulas(i, x_sample[i-1], y_max_sample[i-1], x_sample[i], y_max_sample[i], d_max[i-1], d_max[i])
        print(formula)

    print("\n=== Piecewise PCHIP Polynomials (Lower Boundary) ===")
    for i in range(1, len(x_sample)):
        formula = hermite_formulas(i, x_sample[i-1], y_min_sample[i-1], x_sample[i], y_min_sample[i], d_min[i-1], d_min[i])
        print(formula)

    plt.figure(figsize=(6,4))
    plt.imshow(img, cmap='gray')
    plt.plot(x_sample, y_min_sample, 'go', label="lower sample points")
    plt.plot(x_sample, y_max_sample, 'bo', label="upper sample points")
    plt.plot(x_vals, y_min_fit, 'r-', label="lower PCHIP")
    plt.plot(x_vals, y_max_fit, 'm-', label="upper PCHIP")
    plt.legend()
    plt.tight_layout()
    plt.savefig("D4/PCHIP_.png")

    if return_arrays:
        return x_vals, y_min_fit, y_max_fit


x_vals, y_lower, y_upper = fit_image_boundaries("D4/M.png", num_samples=23)

np.save("D4/x_vals.npy", x_vals)
np.save("D4/y_lower.npy", y_lower)
np.save("D4/y_upper.npy", y_upper)
