import numpy as np
import matplotlib.pyplot as plt
import config_heatsink as cfg
from pathlib import Path

def plot_side_by_side(results_file=cfg.RESULTS_PATH):
    """Generates a side-by-side comparison of Ground Truth and PINN Prediction."""

    Path("plots").mkdir(exist_ok=True)
    Path("plots/final_presentations").mkdir(exist_ok=True, parents=True)
    
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
         print(f"Error: Key {e} not found in results file. Did the training save correctly?")
         return

    # --- Select points for a specific slice (e.g., Y = midpoint) ---
    y_slice_value = (cfg.Y_MAX - cfg.Y_MIN) / 2.0
    slice_indices = np.where(np.abs(coords[:, 1] - y_slice_value) < 0.01)[0]

    if len(slice_indices) == 0:
        print(f"Warning: No points found near Y = {y_slice_value}. Plotting all points projected.")
        slice_coords_x = coords[:, 0]
        slice_coords_z = coords[:, 2]
        slice_temps_true = temps_true
        slice_temps_pred = temps_pred
        title_suffix = '(All Points Projected on XZ)'
    else:
        slice_coords_x = coords[slice_indices, 0]
        slice_coords_z = coords[slice_indices, 2]
        slice_temps_true = temps_true[slice_indices]
        slice_temps_pred = temps_pred[slice_indices]
        title_suffix = f'(Y = {y_slice_value:.2f} Slice)'

    print(f"Plotting comparison for {len(slice_temps_true)} points...")

    # Determine common color limits
    vmin = min(np.min(slice_temps_true), np.min(slice_temps_pred), cfg.T_AIR)
    vmax = max(np.max(slice_temps_true), np.max(slice_temps_pred), cfg.BASE_TEMP)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Global title
    fig.suptitle('PINN Prediction vs Ground Truth Comparison', fontsize=24, fontweight='bold', y=0.98)

    # Ground Truth Plot
    contour1 = axes[0].tricontourf(slice_coords_x, slice_coords_z, slice_temps_true, 
                                   levels=50, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('X coordinate', fontsize=18, fontweight='bold')
    axes[0].set_ylabel('Z coordinate', fontsize=18, fontweight='bold')
    axes[0].set_title(f'Ground Truth {title_suffix}', fontsize=20, fontweight='bold')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].tick_params(axis='both', labelsize=14)
    cbar1 = fig.colorbar(contour1, ax=axes[0])
    cbar1.set_label('Temperature (°C)', fontsize=16, fontweight='bold')
    cbar1.ax.tick_params(labelsize=14)

    # PINN Prediction Plot
    contour2 = axes[1].tricontourf(slice_coords_x, slice_coords_z, slice_temps_pred, 
                                   levels=50, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('X coordinate', fontsize=18, fontweight='bold')
    axes[1].set_ylabel('Z coordinate', fontsize=18, fontweight='bold')
    axes[1].set_title(f'PINN Prediction (MAE: {mae:.2f}°C) {title_suffix}', fontsize=20, fontweight='bold')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].tick_params(axis='both', labelsize=14)
    cbar2 = fig.colorbar(contour2, ax=axes[1])
    cbar2.set_label('Temperature (°C)', fontsize=16, fontweight='bold')
    cbar2.ax.tick_params(labelsize=14)
    
    save_path = "plots/final_presentations/prediction_vs_truth.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Side-by-side comparison plot saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    plot_side_by_side()
