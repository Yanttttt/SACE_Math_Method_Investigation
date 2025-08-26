import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, Optional, Dict, Any, List

FuncType = Union[Callable[[np.ndarray], Union[np.ndarray, tuple]], str]

def _allowed_namespace(u: np.ndarray) -> Dict[str, object]:
    """Allowed names when evaluating string expressions."""
    allowed = {
        "x": u, "t": u,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
        "abs": np.abs, "pi": np.pi, "e": np.e,
        "floor": np.floor, "ceil": np.ceil, "round": np.round,
        "where": np.where, "maximum": np.maximum, "minimum": np.minimum,
        "heaviside": np.heaviside, "sign": np.sign,
    }
    allowed["sigmoid"] = lambda x_: 1.0 / (1.0 + np.exp(-x_))
    return allowed

def plot_functions(
    curves: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
    show_grid: bool = True,
):
    """
    Plot multiple curves (ordinary functions or parametric equations) on one figure.

    Each curve is a dict with keys:
      - "func": callable or string
          If returns y: plot y = f(x) over [x_min, x_max]
          If returns (X, Y): parametric curve with t in [t_min, t_max]
      - "x_min", "x_max": domain for normal functions
      - "t_min", "t_max": domain for parametric functions
      - "num_points": number of points (default 1000)
      - "label": legend label
      - "style": dict, passed to plt.plot (e.g. {"color": "r", "linestyle": "--"})
    """
    plt.figure()

    for i, cfg in enumerate(curves):
        f = cfg.get("func")
        num_points = cfg.get("num_points", 1000)

        # decide domain variable
        if "x_min" in cfg and "x_max" in cfg:
            u = np.linspace(cfg["x_min"], cfg["x_max"], num_points)
        elif "t_min" in cfg and "t_max" in cfg:
            u = np.linspace(cfg["t_min"], cfg["t_max"], num_points)
        else:
            raise ValueError(f"Curve {i} must specify x_min/x_max or t_min/t_max.")

        # evaluate function
        if isinstance(f, str):
            val = eval(f, {"__builtins__": {}}, _allowed_namespace(u))
        else:
            val = f(u)

        # decide coordinates
        if isinstance(val, (tuple, list)) and len(val) == 2:
            X, Y = val
        else:
            X, Y = u, val

        label = cfg.get("label", f"curve{i+1}")
        style = cfg.get("style", {})

        plt.plot(X, Y, label=label, **style)

    # plt.xlabel("x")
    # plt.ylabel("y")
    if title:
        plt.title(title)
    if show_grid:
        plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
