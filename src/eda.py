import os
import numpy as np
import matplotlib.pyplot as plt
import config as cfg


def analyze_fdm_data(filepath: str = None, save_dir: str | None = None) -> None:
    if filepath is None:
        filepath = cfg.GROUND_TRUTH_PATH
    print(f"--- EDA for {filepath} ---")

    try:
        T = np.load(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{filepath}'.")
        print("Please run src/fdm_solver.py first to generate the data.")
        return

    # 1) Basic statistics
    print("\n--- Basic Statistics ---")
    print(f"Data shape: {T.shape} (X, Y, Z)")
    print(f"Min Temperature: {T.min():.2f} °C")
    print(f"Max Temperature: {T.max():.2f} °C")
    print(f"Mean Temperature: {T.mean():.2f} °C")

    # 2) Spatial distribution (center slice + histogram)
    print("\n--- Spatial Distribution ---")
    os.makedirs(save_dir, exist_ok=True) if save_dir else None
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(T.flatten(), bins=50, color="orangered")
    plt.title("Histogram of Final Temperatures")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    center_slice_z = T[:, :, T.shape[2] // 2]
    im = plt.imshow(center_slice_z.T, origin="lower", cmap="hot")
    plt.colorbar(im, label="Temperature (°C)")
    plt.title("Center Z-Slice Visualization")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.tight_layout()
    if save_dir:
        out_path = os.path.join(save_dir, "spatial_distribution.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close()
    else:
        plt.show()

    # 3) Gradient analysis
    print("\n--- Gradient Analysis ---")
    grad_x, grad_y, grad_z = np.gradient(T)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    print(f"Max Gradient Magnitude: {gradient_magnitude.max():.2f} °C/m")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(gradient_magnitude.flatten(), bins=50, color="skyblue", log=True)
    plt.title("Histogram of Gradient Magnitudes (Log Scale)")
    plt.xlabel("Gradient Magnitude (°C/m)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    center_slice_grad = gradient_magnitude[:, :, T.shape[2] // 2]
    im_grad = plt.imshow(center_slice_grad.T, origin="lower", cmap="viridis")
    plt.colorbar(im_grad, label="Gradient Mag. (°C/m)")
    plt.title("Gradient Magnitude at Center Z-Slice")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.tight_layout()
    if save_dir:
        out_path = os.path.join(save_dir, "gradient_analysis.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    analyze_fdm_data(cfg.GROUND_TRUTH_PATH, save_dir=os.path.join("outputs", "eda"))


