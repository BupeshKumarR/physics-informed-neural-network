import numpy as np
import matplotlib.pyplot as plt
import config_heatsink as cfg
from pathlib import Path

def plot_losses(results_file=cfg.RESULTS_PATH):
    """Plots the training loss curve from the saved results."""

    Path("plots").mkdir(exist_ok=True)
    Path("plots/final_presentations").mkdir(exist_ok=True, parents=True)
    
    print(f"Loading results from {results_file}...")
    try:
        results = np.load(results_file)
        losses = results['losses'] # Assumes 'losses' key holds the total weighted loss per epoch
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file}")
        return
    except KeyError as e:
         print(f"Error: Key {e} not found in results file. Ensure 'losses' are saved.")
         return

    print(f"Plotting loss curve for {len(losses)} epochs...")

    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(losses, label='Total Weighted Loss', linewidth=2)
    # If you saved individual raw losses, plot them too:
    # if 'pde_losses_raw' in results: ax.plot(results['pde_losses_raw'], label='PDE Loss (Raw)', alpha=0.7)
    # if 'bc_hot_losses_raw' in results: ax.plot(results['bc_hot_losses_raw'], label='BC Hot Loss (Raw)', alpha=0.7)
    # if 'bc_conv_losses_raw' in results: ax.plot(results['bc_conv_losses_raw'], label='BC Conv Loss (Raw)', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=18, fontweight='bold')
    ax.set_ylabel('Loss (log scale)', fontsize=18, fontweight='bold')
    ax.set_title('Training Loss Curve', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # if 'bc_cold_losses_raw' in results: ax.plot(results['bc_cold_losses_raw'], label='BC Cold Loss (Raw)', alpha=0.7)


    ax.set_yscale('log') # Use log scale to see details
    
    save_path = "plots/final_presentations/training_loss_curve.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss curve plot saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    # Note: You might need to modify train_phase2.py to save individual raw losses
    # to the results.npz file if you want to plot them separately.
    plot_losses()
