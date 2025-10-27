import numpy as np
import matplotlib.pyplot as plt
import config_heatsink as cfg
from pathlib import Path

def plot_ground_truth_slice():
    """Generates a heatmap visualization of the FEniCS ground truth data slice."""
    
    Path("plots").mkdir(exist_ok=True)
    Path("plots/final_presentations").mkdir(exist_ok=True, parents=True)
    
    print("Loading ground truth data...")
    data = np.load(cfg.GROUND_TRUTH_PATH)
    coords = data[:, :3]  # x, y, z
    temps_true = data[:, 3]  # T
    
    # --- Select points for a specific slice (e.g., Y = midpoint) ---
    y_slice_value = (cfg.Y_MAX - cfg.Y_MIN) / 2.0
    slice_indices = np.where(np.abs(coords[:, 1] - y_slice_value) < 0.01)[0] # Adjust tolerance if needed
    
    if len(slice_indices) == 0:
        print(f"Warning: No points found near Y = {y_slice_value}. Adjust slice value or tolerance.")
        # Fallback: plot all points projected onto XZ plane
        slice_coords_x = coords[:, 0]
        slice_coords_z = coords[:, 2]
        slice_temps = temps_true
        title = 'Ground Truth (All Points Projected on XZ)'
    else:
        slice_coords_x = coords[slice_indices, 0]
        slice_coords_z = coords[slice_indices, 2]
        slice_temps = temps_true[slice_indices]
        title = f'Ground Truth (Y = {y_slice_value:.2f} Slice)'

    print(f"Plotting {len(slice_temps)} points for the slice...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use tricontourf for potentially unstructured FEniCS points
    contour = ax.tricontourf(slice_coords_x, slice_coords_z, slice_temps, levels=50, cmap='hot')
    
    ax.set_xlabel('X coordinate', fontsize=18, fontweight='bold')
    ax.set_ylabel('Z coordinate', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.set_aspect('equal', adjustable='box') # Ensure correct aspect ratio
    ax.tick_params(axis='both', labelsize=14)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Temperature (°C)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    save_path = "plots/final_presentations/ground_truth_slice.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Ground truth slice plot saved to {save_path}")
    # plt.show() # Uncomment to display locally

if __name__ == "__main__":
    plot_ground_truth_slice()
