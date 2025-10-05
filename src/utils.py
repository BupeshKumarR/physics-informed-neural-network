from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_center_slice(T: np.ndarray, title: str = "Center Z-Slice", extent=None, cmap: str = "hot") -> None:
    zc = T.shape[2] // 2
    sl = T[:, :, zc]
    plt.imshow(sl.T, origin="lower", cmap=cmap, extent=extent)
    plt.colorbar(label="Temperature (°C)")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")


