import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_workflow_comparison():
    """Creates a workflow comparison diagram showing FEM vs PINN approaches."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Define colors
    fem_color = '#ff9999'
    pinn_color = '#99ccff'
    text_color = '#333333'
    
    # FEM Workflow (Left)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_title('Traditional FEM Approach', fontsize=20, fontweight='bold', pad=15)
    
    # FEM Steps
    fem_steps = [
        ("Setup Geometry\n& Parameters", 5, 10.5, 2, 1),
        ("Generate Mesh\n(Complex)", 5, 8.5, 2, 1),
        ("Solve PDE\n(FEniCS)", 5, 6.5, 2, 1),
        ("Post-process\nResults", 5, 4.5, 2, 1),
        ("Design Iteration\n(Repeat All)", 5, 2.5, 2, 1)
    ]
    
    fem_times = ["1 hour", "3 hours", "15 min", "5 min", "~4.25 hrs total"]
    
    for i, (text, x, y, w, h) in enumerate(fem_steps):
        # Draw box
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                           boxstyle="round,pad=0.1", 
                           facecolor=fem_color, 
                           edgecolor='black',
                           linewidth=1.5)
        ax1.add_patch(box)
        
        # Add text
        ax1.text(x, y, text, ha='center', va='center', 
                fontsize=14, fontweight='bold', color=text_color)
        
        # Add time estimate
        ax1.text(x, y-0.7, f"({fem_times[i]})", ha='center', va='center', 
                fontsize=12, style='italic', color='#666666')
        
        # Add arrow (except for last step)
        if i < len(fem_steps) - 1:
            ax1.arrow(x, y-0.8, 0, -0.4, head_width=0.1, head_length=0.1, 
                     fc='black', ec='black')
    
    ax1.text(5, 0.5, "Per Design Iteration", ha='center', va='center', 
            fontsize=16, fontweight='bold', color='red')
    
    # PINN Workflow (Right)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title('PINN Approach', fontsize=20, fontweight='bold', pad=15)
    
    # PINN Steps
    pinn_steps = [
        ("Train Model\n(One Time)", 5, 10.5, 2, 1),
        ("Input New Design\nParameters", 5, 8.5, 2, 1),
        ("Instant Prediction\n(< 1 second)", 5, 6.5, 2, 1),
        ("Design Iteration\n(Repeat Steps 2-3)", 5, 4.5, 2, 1)
    ]
    
    pinn_times = ["8 hours", "< 1 sec", "< 1 sec", "< 1 sec per design"]
    
    for i, (text, x, y, w, h) in enumerate(pinn_steps):
        # Draw box
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                           boxstyle="round,pad=0.1", 
                           facecolor=pinn_color, 
                           edgecolor='black',
                           linewidth=1.5)
        ax2.add_patch(box)
        
        # Add text
        ax2.text(x, y, text, ha='center', va='center', 
                fontsize=14, fontweight='bold', color=text_color)
        
        # Add time estimate
        ax2.text(x, y-0.7, f"({pinn_times[i]})", ha='center', va='center', 
                fontsize=12, style='italic', color='#666666')
        
        # Add arrow (except for last step)
        if i < len(pinn_steps) - 1:
            ax2.arrow(x, y-0.8, 0, -0.4, head_width=0.1, head_length=0.1, 
                     fc='black', ec='black')
    
    ax2.text(5, 2.5, "Massive Speedup for\nMultiple Designs", ha='center', va='center', 
            fontsize=16, fontweight='bold', color='blue')
    
    # Add speedup calculation
    ax2.text(5, 0.5, "~15,000x faster per design", ha='center', va='center', 
            fontsize=14, fontweight='bold', color='green',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.7))
    
    # Remove axes
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # Create final_presentations directory
    Path("plots/final_presentations").mkdir(exist_ok=True, parents=True)
    
    # Save the plot
    save_path = "plots/final_presentations/workflow_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Workflow comparison diagram saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    create_workflow_comparison()
