"""
Plot 4: Enhanced Error Map
Target: Technical audience (skeptics need proof)
Goal: Show where the model fails and where it succeeds
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import config_heatsink as cfg
from src.train_phase2 import HeatSinkPINN

def create_enhanced_error_map():
    """Creates a detailed error map showing prediction accuracy spatially"""
    
    Path("plots").mkdir(exist_ok=True)
    Path("plots/final_presentations").mkdir(exist_ok=True, parents=True)
    
    # Load ground truth
    print("Loading ground truth data...")
    data = np.load(cfg.GROUND_TRUTH_PATH)
    coords = data[:, :3]
    temps_true = data[:, 3]
    
    # Load and evaluate model
    print("Loading best model...")
    device = torch.device("cpu")
    model = HeatSinkPINN(num_layers=cfg.MLP_LAYERS, hidden_size=cfg.MLP_HIDDEN)
    model.load_state_dict(torch.load("models/heatsink_pinn_model.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        # Normalize coordinates
        x_norm = (torch.tensor(coords[:, 0], dtype=torch.float32).reshape(-1, 1) - cfg.COORD_NORM_MIN) / (cfg.COORD_NORM_MAX - cfg.COORD_NORM_MIN)
        y_norm = (torch.tensor(coords[:, 1], dtype=torch.float32).reshape(-1, 1) - cfg.COORD_NORM_MIN) / (cfg.COORD_NORM_MAX - cfg.COORD_NORM_MIN)
        z_norm = (torch.tensor(coords[:, 2], dtype=torch.float32).reshape(-1, 1) - cfg.COORD_NORM_MIN) / (cfg.COORD_NORM_MAX - cfg.COORD_NORM_MIN)
        t_norm = torch.zeros_like(x_norm)
        
        T_norm = model(t_norm, x_norm, y_norm, z_norm)
        
        # Denormalize
        temps_pred = T_norm.numpy().flatten() * (cfg.T_NORM_MAX - cfg.T_NORM_MIN) + cfg.T_NORM_MIN
    
    # Calculate errors
    abs_error = np.abs(temps_pred - temps_true)
    rel_error = abs_error / (temps_true + 1e-10) * 100  # Percentage error
    
    # Calculate metrics
    mae = np.mean(abs_error)
    mse = np.mean((temps_pred - temps_true)**2)
    max_error = np.max(abs_error)
    
    print(f"Model Performance:")
    print(f"  MAE: {mae:.2f}°C")
    print(f"  MSE: {mse:.2f}")
    print(f"  Max Error: {max_error:.2f}°C")
    print(f"  Mean Rel Error: {np.mean(rel_error):.2f}%")
    
    # Select a slice for visualization
    y_slice_value = (cfg.Y_MAX - cfg.Y_MIN) / 2.0
    slice_indices = np.where(np.abs(coords[:, 1] - y_slice_value) < 0.01)[0]
    
    if len(slice_indices) < 10:
        slice_indices = np.arange(len(coords))
        title_suffix = '(All Points)'
    else:
        title_suffix = f'(Y = {y_slice_value:.2f} Slice)'
    
    slice_coords_x = coords[slice_indices, 0]
    slice_coords_z = coords[slice_indices, 2]
    slice_abs_error = abs_error[slice_indices]
    slice_rel_error = rel_error[slice_indices]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Absolute Error Map
    contour1 = axes[0, 0].tricontourf(slice_coords_x, slice_coords_z, slice_abs_error, 
                                      levels=50, cmap='Reds')
    axes[0, 0].set_xlabel('X coordinate', fontsize=18, fontweight='bold')
    axes[0, 0].set_ylabel('Z coordinate', fontsize=18, fontweight='bold')
    axes[0, 0].set_title(f'Absolute Error Map {title_suffix}\n(MAE: {mae:.2f}°C)', 
                        fontsize=20, fontweight='bold')
    axes[0, 0].set_aspect('equal', adjustable='box')
    axes[0, 0].tick_params(axis='both', labelsize=14)
    cbar1 = fig.colorbar(contour1, ax=axes[0, 0])
    cbar1.set_label('Absolute Error (°C)', fontsize=16, fontweight='bold')
    cbar1.ax.tick_params(labelsize=14)
    
    # Plot 2: Relative Error Map
    contour2 = axes[0, 1].tricontourf(slice_coords_x, slice_coords_z, slice_rel_error, 
                                      levels=50, cmap='YlOrRd')
    axes[0, 1].set_xlabel('X coordinate', fontsize=18, fontweight='bold')
    axes[0, 1].set_ylabel('Z coordinate', fontsize=18, fontweight='bold')
    axes[0, 1].set_title(f'Relative Error Map {title_suffix}\n(Mean Rel Error: {np.mean(slice_rel_error):.2f}%)', 
                        fontsize=20, fontweight='bold')
    axes[0, 1].set_aspect('equal', adjustable='box')
    axes[0, 1].tick_params(axis='both', labelsize=14)
    cbar2 = fig.colorbar(contour2, ax=axes[0, 1])
    cbar2.set_label('Relative Error (%)', fontsize=16, fontweight='bold')
    cbar2.ax.tick_params(labelsize=14)
    
    # Plot 3: Error Distribution Histogram
    axes[1, 0].hist(abs_error, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Absolute Error (°C)', fontsize=18, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=18, fontweight='bold')
    axes[1, 0].set_title('Error Distribution', fontsize=20, fontweight='bold')
    axes[1, 0].tick_params(axis='both', labelsize=14)
    axes[1, 0].axvline(mae, color='green', linestyle='--', linewidth=2, label=f'MAE: {mae:.2f}°C')
    axes[1, 0].axvline(max_error, color='red', linestyle='--', linewidth=2, label=f'Max: {max_error:.2f}°C')
    axes[1, 0].legend(fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error vs Temperature Scatter
    scatter = axes[1, 1].scatter(temps_true, abs_error, c=temps_true, 
                               cmap='hot', alpha=0.6, s=10, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('True Temperature (°C)', fontsize=18, fontweight='bold')
    axes[1, 1].set_ylabel('Absolute Error (°C)', fontsize=18, fontweight='bold')
    axes[1, 1].set_title('Error vs True Temperature', fontsize=20, fontweight='bold')
    axes[1, 1].tick_params(axis='both', labelsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    cbar3 = fig.colorbar(scatter, ax=axes[1, 1])
    cbar3.set_label('Temperature (°C)', fontsize=16, fontweight='bold')
    cbar3.ax.tick_params(labelsize=14)
    
    # Add overall title
    fig.suptitle(f'Model Error Analysis - MAE: {mae:.2f}°C, MSE: {mse:.2f}', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = "plots/final_presentations/error_map_enhanced.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Enhanced error map saved to {save_path}")
    
    return {'mae': mae, 'mse': mse, 'max_error': max_error}

if __name__ == "__main__":
    metrics = create_enhanced_error_map()
    print(f"\nFinal Metrics:")
    print(f"  MAE: {metrics['mae']:.2f}°C")
    print(f"  MSE: {metrics['mse']:.2f}")
    print(f"  Max Error: {metrics['max_error']:.2f}°C")

