import time
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from src.utils import plot_center_slice


def main() -> None:
    # Grid spacing derived from config
    dx, dy, dz = cfg.LX / cfg.NX, cfg.LY / cfg.NY, cfg.LZ / cfg.NZ

    # Stability-limited time step for explicit scheme
    dt = cfg.STABILITY_FACTOR * (
        1.0 / (2.0 * cfg.ALPHA * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2))
    )
    nt = int(cfg.T_FINAL / dt)

    # Initialize field and Dirichlet boundaries
    T = np.full((cfg.NX, cfg.NY, cfg.NZ), cfg.T_INITIAL, dtype=np.float64)
    T[0, :, :] = cfg.T_BOUNDARY
    T[-1, :, :] = cfg.T_BOUNDARY
    T[:, 0, :] = cfg.T_BOUNDARY
    T[:, -1, :] = cfg.T_BOUNDARY
    T[:, :, 0] = cfg.T_BOUNDARY
    T[:, :, -1] = cfg.T_BOUNDARY

    # Source region (central cube)
    cx, cy, cz = cfg.NX // 2, cfg.NY // 2, cfg.NZ // 2
    hw = cfg.SRC_HALF_WIDTH
    xs, xe = cx - hw, cx + hw
    ys, ye = cy - hw, cy + hw
    zs, ze = cz - hw, cz + hw

    print(f"Running simulation for {nt} time steps with dt = {dt:.6e} s")
    start = time.time()

    for n in range(nt):
        T_new = T.copy()

        lap = (
            (T[2:, 1:-1, 1:-1] - 2.0 * T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) / (dx**2)
            + (T[1:-1, 2:, 1:-1] - 2.0 * T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) / (dy**2)
            + (T[1:-1, 1:-1, 2:] - 2.0 * T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2]) / (dz**2)
        )

        T_new[1:-1, 1:-1, 1:-1] = T[1:-1, 1:-1, 1:-1] + cfg.ALPHA * dt * lap

        # Add internal heat source
        T_new[xs:xe, ys:ye, zs:ze] += cfg.Q_VALUE * dt

        # Re-apply Dirichlet boundaries
        T_new[0, :, :] = cfg.T_BOUNDARY
        T_new[-1, :, :] = cfg.T_BOUNDARY
        T_new[:, 0, :] = cfg.T_BOUNDARY
        T_new[:, -1, :] = cfg.T_BOUNDARY
        T_new[:, :, 0] = cfg.T_BOUNDARY
        T_new[:, :, -1] = cfg.T_BOUNDARY

        T = T_new

        if nt > 0 and (n + 1) % max(1, nt // 10) == 0:
            print(f"Step {n + 1}/{nt} completed.")

    dur = time.time() - start
    print(f"Simulation finished in {dur:.2f} seconds.")

    # Save field
    np.save(cfg.GROUND_TRUTH_PATH, T)
    print(f"Ground truth data saved to {cfg.GROUND_TRUTH_PATH}")

    # Visualize center slice
    plt.figure(figsize=(8, 6))
    plot_center_slice(T, title="FDM Ground Truth: Center Z-Slice at T_final", extent=[0, cfg.LX, 0, cfg.LY])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


