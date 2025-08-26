from typing import Iterable, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

ArrayLike2D = Sequence[Sequence[float]]  # [[x1,y1], [x2,y2], ...]

def bezier_points(control_points: ArrayLike2D, num: int = 400):
    """
    Evaluate a 2D Bezier curve using a vectorized De Casteljau algorithm.
    Works for quadratic (3 points), cubic (4 points), or any order >= 2.
    Returns an (num, 2) ndarray of sampled points along the curve.
    """
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
    """
    Plot many Bezier curves on one axes.
    - control_sets: iterable of (curve_id, control point list)
    - num: sampling resolution
    - Control points are shown as circles and labeled C^{(id)}_{m}.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for cid, cps in control_sets:
        pts = bezier_points(cps, num=num)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=2)  # curve
        cps_arr = np.asarray(cps, dtype=float)
        ax.scatter(cps_arr[:, 0], cps_arr[:, 1])    # control points
        for m, (x, y) in enumerate(cps_arr, start=0):
            ax.text(x, y, rf"$C^{{({cid})}}_{{{m+1}}}$", 
                    fontsize=10, ha="left", va="bottom")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    curves = [
        # [(-3.74,17.19),(-13.4, -42.48), (51.12,15.53), (66.82,-6.08)],
        (2, [(-135.39,171.46), (-83.86,36.14), (3.91,64.40), (-3.74,17.19)]),
        # [(-135.39,171.46),(-205.88,-40.99),(96.95,-1.58),(-99.30,-63.83)],
        # [(-99.30,-63.83), (-49.98,-93.38), (26.78,-74.38)],
        # [(26.78,-74.38),(7.57,-63.97), (2.45,-53.06)],
        # [(66.52,-14.33), (30.15,-27.30), (54.53,-38.65), (2.45,-53.06)],
        # [(66.52,-14.33), (72.46, -12.21), (78.62,-14.24)],
        # [(78.62,-14.24), (79.02,-5.29), (66.82,-6.08)]
    ]
    plot_beziers(curves, num=500, figsize=(3,2))
