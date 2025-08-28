from typing import Iterable, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

ArrayLike2D = Sequence[Sequence[float]]  # [[x1,y1], [x2,y2], ...]

def bezier_points(control_points: ArrayLike2D, num: int = 400):
    P = np.asarray(control_points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] < 2:
        raise ValueError("control_points must be an Nx2 array with N>=2")
    t = np.linspace(0.0, 1.0, num=num, endpoint=True)           # (num,)
    Q = np.broadcast_to(P, (t.size, ) + P.shape).copy()         # (num, N, 2)
    for _ in range(1, P.shape[0]):
        Q = (1 - t)[:, None, None] * Q[:, :-1, :] + t[:, None, None] * Q[:, 1:, :]
    return Q[:, 0, :]                                           # (num, 2)

def plot_beziers(control_sets: Iterable[Tuple[int, ArrayLike2D]], *, 
                 num: int = 400, figsize: Tuple[float, float] = (7, 5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    used_points = []  # 保存已标记过的点坐标

    for cid, cps in control_sets:
        pts = bezier_points(cps, num=num)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=2)  # 曲线
        cps_arr = np.asarray(cps, dtype=float)
        ax.scatter(cps_arr[:, 0], cps_arr[:, 1])    # 控制点

        for m, (x, y) in enumerate(cps_arr, start=0):
            offset = (0.0, 0.0)
            for (ux, uy) in used_points:
                if np.hypot(x - ux, y - uy) < 1e-6:  # 与已有点重合
                    offset = (-15, -15)  # 往右上挪一点
                    break
            used_points.append((x, y))
            ax.text(x + offset[0], y + offset[1], 
                    rf"$C^{{({cid})}}_{{{m+1}}}$", 
                    fontsize=10, ha="left", va="bottom")

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.12)
    plt.show()


if __name__ == "__main__":
    curves = [
        # (1, [(-135.39,171.46), (-83.86,36.14), (3.91,64.40), (-3.74,17.19)]),
        # (2, [(-3.74,17.19), (-13.4,-42.48), (51.12,15.53), (66.82,-6.08)]),
        # (3, [(66.82,-6.08), (79.02,-5.29), (78.62,-14.24)]),
        # (4, [(78.62,-14.24), (72.46,-12.21), (66.52,-14.33)]),
        (5, [(66.52,-14.33), (30.15,-27.30), (54.53,-38.65), (2.45,-53.06)]),
        # (6, [(2.45,-53.06), (7.57,-63.97), (26.78,-74.38)]),
        # (7, [(26.78,-74.38), (-49.98,-93.38), (-99.30,-63.83)]),
        # (8, [(-99.30,-63.83), (96.95,-1.58), (-205.88,-40.99), (-135.39,171.46)])
    ]
    plot_beziers(curves, num=500, figsize=(3,2))
