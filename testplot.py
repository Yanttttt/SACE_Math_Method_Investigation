import matplotlib.pyplot as plt
import numpy as np

t_vals = np.linspace(0, 1, 500)

def B1(t):
    x = -131.66*t**3 +108.72*t**2 +154.59*t -135.39
    y = -239.05*t**3 +490.74*t**2 -405.96*t +171.46
    return x, y

def B2(t):
    x = -123.0*t**3 +222.54*t**2 -28.98*t -3.74
    y = -197.3*t**3 +353.04*t**2 -179.01*t +17.19
    return x, y

def B3(t):
    x = -12.6*t**2 +24.4*t +66.82
    y = -9.74*t**2 +1.58*t -6.08
    return x, y

def B4(t):
    x = 0.22*t**2 -12.32*t +78.62
    y = -4.15*t**2 +4.06*t -14.24
    return x, y

def B5(t):
    x = -137.21*t**3 +182.25*t**2 -109.11*t +66.52
    y = -4.68*t**3 +4.86*t**2 -38.91*t -14.33
    return x, y

def B6(t):
    x = 14.09*t**2 +10.24*t +2.45
    y = 0.5*t**2 -21.82*t -53.06
    return x, y

def B7(t):
    x = 27.44*t**2 -153.52*t +26.78
    y = 48.55*t**2 -38.0*t -74.38
    return x, y

def B8(t):
    x = 872.4*t**3 -1497.24*t**2 +588.75*t -99.3
    y = 353.52*t**3 -304.98*t**2 +186.75*t -63.83
    return x, y

curves = [B1, B2, B3, B4, B5, B6, B7, B8]

plt.figure(figsize=(4,4))
colors = plt.cm.tab10(np.arange(8))

for i, B in enumerate(curves):
    x_vals, y_vals = B(t_vals)
    plt.plot(x_vals, y_vals, label=f'B{i+1}', color=colors[i])

plt.grid(True, linestyle=":")
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()
