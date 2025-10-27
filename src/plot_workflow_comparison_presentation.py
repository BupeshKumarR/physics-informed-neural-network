"""
Enhanced Workflow Comparison: FEM vs PINN
Optimized for presentation readability
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path

def create_presentation_workflow():
    """Creates a clean, impactful workflow comparison for presentations"""
    
    Path("plots").mkdir(exist_ok=True)
    Path("plots/final_presentations").mkdir(exist_ok=True, parents=True)
    
    # Create figure with white background
    fig = plt.figure(figsize=(22, 12), facecolor='white')
    
    # Create two subplots with more spacing
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Professional colors
    fem_color = '#FFCDD2'        # Light red
    fem_border = '#D32F2F'       # Dark red
    pinn_color = '#C5E1A5'       # Light green
    pinn_border = '#558B2F'      # Dark green
    
    # ==================== FEM WORKFLOW (LEFT) ====================
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 14)
    ax1.axis('off')
    
    # Title
    ax1.text(5, 13, 'Traditional FEM Approach', 
            ha='center', va='center', fontsize=26, fontweight='bold')
    ax1.text(5, 12.3, '(Finite Element Method)', 
            ha='center', va='center', fontsize=18, style='italic', color='#666')
    
    # FEM Steps with larger boxes
    fem_steps = [
        ("Setup Geometry\n& Parameters", 5, 10.2),
        ("Generate Mesh\n(Complex Process)", 5, 8),
        ("Solve PDE\n(FEniCS/ANSYS)", 5, 5.8),
        ("Post-process\nResults", 5, 3.6),
    ]
    
    fem_times = ["~1 hour", "~3 hours", "~15 min", "~5 min"]
    
    box_width = 4
    box_height = 1.4
    
    for i, (text, x, y) in enumerate(fem_steps):
        # Draw box
        box = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height, 
                           boxstyle="round,pad=0.15", 
                           facecolor=fem_color, 
                           edgecolor=fem_border,
                           linewidth=3)
        ax1.add_patch(box)
        
        # Main text
        ax1.text(x, y+0.2, text, ha='center', va='center', 
                fontsize=17, fontweight='bold', color='#1a1a1a')
        
        # Time estimate
        ax1.text(x, y-0.45, fem_times[i], ha='center', va='center', 
                fontsize=15, style='italic', color='#D32F2F', fontweight='bold')
        
        # Arrow to next step
        if i < len(fem_steps) - 1:
            arrow = FancyArrowPatch((x, y-box_height/2-0.1), 
                                   (x, fem_steps[i+1][2]+box_height/2+0.1),
                                   arrowstyle='->', mutation_scale=35, 
                                   lw=3.5, color='#D32F2F')
            ax1.add_patch(arrow)
    
    # Iteration loop arrow (curved back)
    loop_arrow = FancyArrowPatch((6.5, 3.6), (6.5, 10.2),
                                arrowstyle='->', mutation_scale=35,
                                lw=3.5, color='#D32F2F', linestyle='--',
                                connectionstyle="arc3,rad=1.5")
    ax1.add_patch(loop_arrow)
    ax1.text(7.8, 6.9, 'Repeat for\neach design', ha='center', va='center',
            fontsize=15, fontweight='bold', color='#D32F2F',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#D32F2F', linewidth=2))
    
    # Total time box
    total_box = Rectangle((1.5, 1.2), 7, 1.3, 
                          facecolor='#FFEBEE', 
                          edgecolor='#D32F2F', linewidth=3)
    ax1.add_patch(total_box)
    ax1.text(5, 2.15, 'Total Per Design:', ha='center', va='center',
            fontsize=17, fontweight='bold', color='#1a1a1a')
    ax1.text(5, 1.6, '~4.25 hours', ha='center', va='center',
            fontsize=20, fontweight='bold', color='#D32F2F')
    
    # Bottom note
    ax1.text(5, 0.3, '❌ Slow for design optimization', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#D32F2F')
    
    # ==================== PINN WORKFLOW (RIGHT) ====================
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 14)
    ax2.axis('off')
    
    # Title
    ax2.text(5, 13, 'PINN Approach', 
            ha='center', va='center', fontsize=26, fontweight='bold')
    ax2.text(5, 12.3, '(Physics-Informed Neural Network)', 
            ha='center', va='center', fontsize=18, style='italic', color='#666')
    
    # One-time training box (highlighted differently)
    training_box = FancyBboxPatch((2.5, 10.2-box_height/2), 5, box_height, 
                                 boxstyle="round,pad=0.15", 
                                 facecolor='#FFF9C4',  # Light yellow
                                 edgecolor='#F57F17',  # Dark yellow
                                 linewidth=3)
    ax2.add_patch(training_box)
    ax2.text(5, 10.4, '⚙️ Train Model (One Time)', ha='center', va='center', 
            fontsize=17, fontweight='bold', color='#1a1a1a')
    ax2.text(5, 9.75, '~8 hours (once)', ha='center', va='center', 
            fontsize=15, style='italic', color='#F57F17', fontweight='bold')
    
    # Arrow down
    arrow1 = FancyArrowPatch((5, 10.2-box_height/2-0.1), (5, 8+box_height/2+0.1),
                            arrowstyle='->', mutation_scale=35, 
                            lw=3.5, color='#558B2F')
    ax2.add_patch(arrow1)
    
    # Fast iteration steps
    pinn_steps = [
        ("Input New\nDesign Parameters", 5, 8),
        ("Instant Prediction\n⚡ < 1 second", 5, 5.8),
        ("Analyze Results", 5, 3.6),
    ]
    
    pinn_times = ["< 1 sec", "< 1 sec", "< 1 sec"]
    
    for i, (text, x, y) in enumerate(pinn_steps):
        # Draw box
        box = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height, 
                           boxstyle="round,pad=0.15", 
                           facecolor=pinn_color, 
                           edgecolor=pinn_border,
                           linewidth=3)
        ax2.add_patch(box)
        
        # Main text
        ax2.text(x, y+0.2, text, ha='center', va='center', 
                fontsize=17, fontweight='bold', color='#1a1a1a')
        
        # Time estimate
        ax2.text(x, y-0.45, pinn_times[i], ha='center', va='center', 
                fontsize=15, style='italic', color='#558B2F', fontweight='bold')
        
        # Arrow to next step
        if i < len(pinn_steps) - 1:
            arrow = FancyArrowPatch((x, y-box_height/2-0.1), 
                                   (x, pinn_steps[i+1][2]+box_height/2+0.1),
                                   arrowstyle='->', mutation_scale=35, 
                                   lw=3.5, color='#558B2F')
            ax2.add_patch(arrow)
    
    # Fast iteration loop (curved back)
    loop_arrow2 = FancyArrowPatch((6.5, 3.6), (6.5, 8),
                                 arrowstyle='->', mutation_scale=35,
                                 lw=3.5, color='#558B2F', linestyle='--',
                                 connectionstyle="arc3,rad=1.2")
    ax2.add_patch(loop_arrow2)
    ax2.text(7.6, 5.8, 'Rapid\niteration', ha='center', va='center',
            fontsize=15, fontweight='bold', color='#558B2F',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#558B2F', linewidth=2))
    
    # Total time box (per design)
    total_box2 = Rectangle((1.5, 1.2), 7, 1.3, 
                           facecolor='#E8F5E9', 
                           edgecolor='#558B2F', linewidth=3)
    ax2.add_patch(total_box2)
    ax2.text(5, 2.15, 'Per New Design:', ha='center', va='center',
            fontsize=17, fontweight='bold', color='#1a1a1a')
    ax2.text(5, 1.6, '< 1 second', ha='center', va='center',
            fontsize=20, fontweight='bold', color='#558B2F')
    
    # Bottom note with speedup
    ax2.text(5, 0.3, '✅ ~15,000× faster per design!', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#558B2F')
    
    plt.tight_layout()
    
    save_path = "plots/final_presentations/workflow_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Presentation-ready workflow comparison saved to {save_path}")

if __name__ == "__main__":
    create_presentation_workflow()

