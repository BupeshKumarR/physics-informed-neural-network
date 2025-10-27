import numpy as np
import matplotlib.pyplot as plt
import config_heatsink as cfg
from pathlib import Path

def plot_error_map(results_file=cfg.RESULTS_PATH):
    """Generates a heatmap visualization of the absolute prediction error."""

    Path("plots").mkdir(exist_ok=True)
    
    print(f"Loading results from {results_file}...")
    try:
        results = np.load(results_file)
        coords = results['coords']
        temps_true = results['temps_true']
        temps_pred = results['temps_pred']
        mae = results['mae']
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file}")
        return
    except KeyError as e:
         print(f"Error: Key {e} not found in results file.")
         return
         
    # Calculate absolute error
    abs_error = np.abs(temps_pred - temps_true)

    # --- Select points for a specific slice (e.g., Y = midpoint) ---
    y_slice_value = (cfg.Y_MAX - cfg.Y_MIN) / 2.0
    slice_indices = np.where(np.abs(coords[:, 1] - y_slice_value) < 0.01)[0] 

    if len(slice_indices) == 0:
        print(f"Warning: No points found near Y = {y_slice_value}. Plotting all points projected.")
        slice_coords_x = coords[:, 0]
        slice_coords_z = coords[:, 2]
        slice_error = abs_error
        title_suffix = '(All Points Projected on XZ)'
    else:
        slice_coords_x = coords[slice_indices, 0]
        slice_coords_z = coords[slice_indices, 2]
        slice_error = abs_error[slice_indices]
        title_suffix = f'(Y = {y_slice_value:.2f} Slice)'

    print(f"Plotting error map for {len(slice_error)} points...")

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use tricontourf for potentially unstructured points
    contour = ax.tricontourf(slice_coords_x, slice_coords_z, slice_error, levels=50, cmap='Reds')
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Z coordinate')
    ax.set_title(f'Absolute Prediction Error (MAE: {mae:.2f}°C) {title_suffix}')
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(contour, label='Absolute Error (°C)')
    
    save_path = "plots/absolute_error_map.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error map plot saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    plot_error_map()
