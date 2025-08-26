import numpy as np
from plot_functions import plot_functions

plot_functions(
    curves=[
        {
            "func": "sin(x)",
            "x_min": -6, "x_max": 6,
            "label": "sin(x)",
            "style": {"color": "blue", "linestyle": "--"}
        }
    ]
)
