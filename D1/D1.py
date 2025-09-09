import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x-1)**2*(x-5)**2+6

theta= np.pi

trans = {
        "rot": np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
        "scale": np.array([[4, 0], [0, 0.2]]),
        "shear": np.array([[1, 0.5], [0, 1]]),
        "none": np.array([[1, 0], [0, 1]]),
        "flip": np.array([[-1, 0], [0, 1]])
    }

def affineTrans(A,b,points,points2,label1,label2,label1T,label2T,savename,show=False):
    points_T = A @ points + b[:, None]
    points2_T = A @ points2 + b[:, None]
    if show:
        origin = np.array([[0], [0]])
        i_hat = np.array([[5], [0]])
        j_hat = np.array([[0], [5]])
        i_trans = A @ i_hat
        j_trans = A @ j_hat

        plt.figure(figsize=(6, 3))

        plt.plot(points[0], points[1], label=label1, color='blue')
        plt.plot(points2[0], points2[1], label=label2, color='blue')
        plt.plot(points_T[0], points_T[1], label=label1T, color='green')
        plt.plot(points2_T[0], points2_T[1], label=label2T, color='green')

        # plt.quiver(*origin, *i_hat, angles='xy', scale_units='xy', scale=1, color='grey', label="i-hat")
        # plt.quiver(*origin, *j_hat, angles='xy', scale_units='xy', scale=1, color='grey', label="j-hat")
        # plt.quiver(*(origin), *i_trans, angles='xy', scale_units='xy', scale=1, color='purple', label="Transformed i-hat")
        # plt.quiver(*(origin), *j_trans, angles='xy', scale_units='xy', scale=1, color='orange', label="Transformed j-hat")

        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(savename, dpi=300)
    return points_T, points2_T;

def nonLinear(points):
    x = points[0]
    y = points[1]
    angle = -np.pi * x / 36 + np.pi / 2
    x_new = y * np.cos(angle)
    y_new = y * np.sin(angle)
    return np.vstack((x_new, y_new))

def mark_start(P, label, color, dx=1.0, dy=1.0):
    plt.scatter(P[0, 0], P[1, 0], color=color, s=20, zorder=5)
    plt.annotate(
        label,
        xy=(P[0, 0], P[1, 0]),
        xytext=(P[0, 0] + dx, P[1, 0] + dy),
        fontsize=14,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color)
    )

b = np.array([44, 0])
b2 = np.array([22, 623/80])

x_vals = np.linspace(1,5.5, 500)
f_vals = f(x_vals)
f2_vals = f(x_vals)-2

points = np.vstack((x_vals, f_vals))  # shape: (2, N)
points2 = np.vstack((x_vals, f2_vals))

points_T1, points2_T1=affineTrans(trans["scale"],np.array([0,0]), points, points2, "P1(t)", "P2(t)", "T(P1(t))", "T(P2(t))", "./D1/trans1.png",True)

points_T2, points2_T2=affineTrans(trans["flip"],b2, points_T1, points2_T1, "P1(t)", "P2(t)", "T(P1(t))", "T(P2(t))", "./D1/trans2.png",True)

points3, points4=affineTrans(trans["flip"],np.array([0,0]), points_T2, points2_T2, "P1(t)", "P2(t)", "P3(t)", "P4(t)", "./D1/trans3.png",True)

P1=nonLinear(points_T2)
P2=nonLinear(points2_T2)
P3=nonLinear(points3)
P4=nonLinear(points4)

x_grid = np.linspace(-18, 18, 7)
y_grid = np.linspace(-18, 18, 7)

plt.figure(figsize=(6, 3))

for y in y_grid:
    xs = np.linspace(-18, 18, 37)
    ys = np.full_like(xs, y)
    grid_line = np.vstack((xs, ys))
    grid_line_trans = nonLinear(grid_line)
    plt.plot(grid_line_trans[0], grid_line_trans[1], color='gray', linewidth=0.5)

for x in x_grid:
    ys = np.linspace(-18, 18, 37)
    xs = np.full_like(ys, x)
    grid_line = np.vstack((xs, ys))
    grid_line_trans = nonLinear(grid_line)
    plt.plot(grid_line_trans[0], grid_line_trans[1], color='gray', linewidth=0.5)

plt.plot(P1[0], P1[1], label="P1(t)", color='blue')
plt.plot(P2[0], P2[1], label="P2(t)", color='blue')
plt.plot(P3[0], P3[1], label="P3(t)", color='green')
plt.plot(P4[0], P4[1], label="P4(t)", color='green')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.axis('equal')
plt.ylim(ymin=0, ymax=12)
plt.xlim(xmin=-12, xmax=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig("./D1/trans_non_linear.png", dpi=300)

P5=trans["rot"] @ P1
P6=trans["rot"] @ P2
P7=trans["rot"] @ P3
P8=trans["rot"] @ P4

plt.figure(figsize=(6, 6))
plt.plot(P1[0], P1[1], label="P1(t)", color='blue')
plt.plot(P2[0], P2[1], label="P2(t)", color='lightblue')
plt.plot(P3[0], P3[1], label="P3(t)", color='green')
plt.plot(P4[0], P4[1], label="P4(t)", color='lightgreen')
plt.plot(P5[0], P5[1], label="P5(t)", color='yellow')
plt.plot(P6[0], P6[1], label="P6(t)", color='lightyellow')
plt.plot(P7[0], P7[1], label="P7(t)", color='red')
plt.plot(P8[0], P8[1], label="P8(t)", color='pink')

mark_start(P1, "P1(1)=P7(1)", 'black', -2, -2)
mark_start(P2, "P2(1)=P8(1)", 'black', -5, 2)
mark_start(P3, "P3(1)=P5(1)", 'black',-4, -2)
mark_start(P4, "P4(1)=P6(1)", 'black',1,0)

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./D1/D1.png", dpi=300)

curves = {
    "P1": P1,
    "P2": P2,
    "P3": P3,
    "P4": P4,
    "P5": P5,
    "P6": P6,
    "P7": P7,
    "P8": P8
}
np.save("D1/curves.npy", curves)