"""
Plot 2: PINN Conceptual Architecture
Target: Technical audience (AI/Physics Experts)
Goal: Show how a PINN works - the secret sauce
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

def create_pinn_architecture_diagram():
    """Creates a flowchart explaining how a PINN works"""
    
    Path("plots").mkdir(exist_ok=True)
    Path("plots/final_presentations").mkdir(exist_ok=True, parents=True)
    
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    
    # Define colors
    input_color = '#e3f2fd'  # Light blue
    network_color = '#fff3e0'  # Light orange
    physics_color = '#f3e5f5'  # Light purple
    loss_color = '#ffebee'  # Light red
    output_color = '#e8f5e9'  # Light green
    
    # 1. Input Layer (Coordinates)
    input_box = FancyBboxPatch((0.5, 5), 1.5, 2, 
                              boxstyle="round,pad=0.2", 
                              facecolor=input_color, 
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 6.8, 'Input\nCoordinates', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    ax.text(1.25, 6.2, '(x, y, z)', ha='center', va='center', 
            fontsize=12, style='italic')
    
    # 2. Neural Network (MLP)
    network_box = FancyBboxPatch((4, 4), 3, 4, 
                                 boxstyle="round,pad=0.3", 
                                 facecolor=network_color, 
                                 edgecolor='black',
                                 linewidth=2.5)
    ax.add_patch(network_box)
    ax.text(5.5, 7.3, 'Neural Network\n(MLP)', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    ax.text(5.5, 6.8, 'Adaptive Sine', ha='center', va='center', 
            fontsize=12, style='italic')
    ax.text(5.5, 6.3, 'Fourier Features', ha='center', va='center', 
            fontsize=12, style='italic')
    
    # Inside the network box - show layers
    layer_y_positions = [5.8, 5.3, 4.8, 4.3]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    for i, (y, name) in enumerate(zip(layer_y_positions, layer_names)):
        ax.text(4.3, y, f"{i+1}. {name}", ha='left', va='center', fontsize=11)
    
    # 3. Temperature Output
    output_box = FancyBboxPatch((9, 5), 1.5, 2, 
                               boxstyle="round,pad=0.2", 
                               facecolor=output_color, 
                               edgecolor='black',
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(9.75, 6.8, 'Temperature', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    ax.text(9.75, 6.2, 'T(x,y,z)', ha='center', va='center', 
            fontsize=12, style='italic')
    
    # 4. Data Loss Path (Top)
    bc_box = FancyBboxPatch((11, 8.5), 2, 1.5, 
                           boxstyle="round,pad=0.2", 
                           facecolor=physics_color, 
                           edgecolor='black',
                           linewidth=2)
    ax.add_patch(bc_box)
    ax.text(12, 9.7, 'Boundary', ha='center', va='center', 
            fontsize=13, fontweight='bold')
    ax.text(12, 9.2, 'Conditions', ha='center', va='center', 
            fontsize=13, fontweight='bold')
    ax.text(12, 8.7, 'BC: T = T₀', ha='center', va='center', 
            fontsize=11, style='italic')
    
    data_loss_box = FancyBboxPatch((11, 6.5), 2, 1.2, 
                                   boxstyle="round,pad=0.2", 
                                   facecolor=loss_color, 
                                   edgecolor='red',
                                   linewidth=2)
    ax.add_patch(data_loss_box)
    ax.text(12, 7.4, 'Data Loss', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='red')
    ax.text(12, 6.8, 'L_data', ha='center', va='center', 
            fontsize=12, style='italic')
    
    # 5. Physics Loss Path (Bottom)
    deriv_box = FancyBboxPatch((11, 3), 2, 1.5, 
                              boxstyle="round,pad=0.2", 
                              facecolor=physics_color, 
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(deriv_box)
    ax.text(12, 4.2, 'Derivatives', ha='center', va='center', 
            fontsize=13, fontweight='bold')
    ax.text(12, 3.6, '∂T/∂x, ∂²T/∂x²', ha='center', va='center', 
            fontsize=11, style='italic')
    ax.text(12, 3.2, 'Auto-diff', ha='center', va='center', 
            fontsize=10)
    
    pde_box = FancyBboxPatch((11, 1), 2, 1.5, 
                             boxstyle="round,pad=0.2", 
                             facecolor=physics_color, 
                             edgecolor='black',
                             linewidth=2)
    ax.add_patch(pde_box)
    ax.text(12, 2.2, 'PDE Residual', ha='center', va='center', 
            fontsize=13, fontweight='bold')
    ax.text(12, 1.6, '∇²T = 0', ha='center', va='center', 
            fontsize=12, style='italic', fontweight='bold')
    ax.text(12, 1.2, 'Heat Equation', ha='center', va='center', 
            fontsize=11)
    
    physics_loss_box = FancyBboxPatch((11, -0.3), 2, 1, 
                                      boxstyle="round,pad=0.2", 
                                      facecolor=loss_color, 
                                      edgecolor='red',
                                      linewidth=2)
    ax.add_patch(physics_loss_box)
    ax.text(12, 0.5, 'Physics Loss', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='red')
    ax.text(12, -0.1, 'L_physics', ha='center', va='center', 
            fontsize=12, style='italic')
    
    # 6. Combined Loss
    combined_loss_box = FancyBboxPatch((5.5, -0.3), 3, 1.5, 
                                       boxstyle="round,pad=0.3", 
                                       facecolor='#ffcdd2', 
                                       edgecolor='red',
                                       linewidth=3)
    ax.add_patch(combined_loss_box)
    ax.text(7, 0.8, 'Combined Loss', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='red')
    ax.text(7, 0.3, 'L_total = λ₁L_data + λ₂L_physics', ha='center', va='center', 
            fontsize=12, style='italic')
    
    # Add arrows
    # Input to Network
    arrow1 = FancyArrowPatch((2, 6), (4, 6), 
                             arrowstyle='->', mutation_scale=25, 
                             lw=2, color='black')
    ax.add_patch(arrow1)
    
    # Network to Output
    arrow2 = FancyArrowPatch((7, 6), (9, 6), 
                             arrowstyle='->', mutation_scale=25, 
                             lw=2, color='black')
    ax.add_patch(arrow2)
    
    # Output to BC (top path)
    arrow3 = FancyArrowPatch((9.75, 7), (11, 8.5), 
                             arrowstyle='->', mutation_scale=20, 
                             lw=2, color='blue')
    ax.add_patch(arrow3)
    ax.text(9.3, 7.8, 'L_data', ha='left', fontsize=9, color='blue')
    
    # Output to Derivatives (bottom path)
    arrow4 = FancyArrowPatch((9.75, 5), (11, 4.5), 
                             arrowstyle='->', mutation_scale=20, 
                             lw=2, color='purple')
    ax.add_patch(arrow4)
    ax.text(9.3, 4.3, 'Derivatives', ha='left', fontsize=9, color='purple')
    
    # BC to Data Loss
    arrow5 = FancyArrowPatch((12, 8.5), (12, 6.5), 
                             arrowstyle='->', mutation_scale=20, 
                             lw=2, color='blue')
    ax.add_patch(arrow5)
    
    # PDE to Physics Loss
    arrow6 = FancyArrowPatch((12, 1), (12, -0.3), 
                             arrowstyle='->', mutation_scale=20, 
                             lw=2, color='purple')
    ax.add_patch(arrow6)
    
    # Both losses to combined
    arrow7 = FancyArrowPatch((10.5, -0.1), (8.5, 0.6), 
                             arrowstyle='->', mutation_scale=25, 
                             lw=2, color='red')
    ax.add_patch(arrow7)
    arrow8 = FancyArrowPatch((10.5, -0.1), (5.5, 0.6), 
                             arrowstyle='->', mutation_scale=25, 
                             lw=2, color='red')
    ax.add_patch(arrow8)
    
    # Title
    ax.text(8, 12, 'Physics-Informed Neural Network (PINN) Architecture', 
            ha='center', va='center', fontsize=22, fontweight='bold')
    
    # Subtitle
    ax.text(8, 11.2, 'Combining neural networks with physics equations to learn faster', 
            ha='center', va='center', fontsize=16, style='italic')
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=input_color, edgecolor='black', label='Input'),
        plt.Rectangle((0,0),1,1, facecolor=network_color, edgecolor='black', label='Neural Network'),
        plt.Rectangle((0,0),1,1, facecolor=output_color, edgecolor='black', label='Output'),
        plt.Rectangle((0,0),1,1, facecolor=physics_color, edgecolor='black', label='Physics Enforcement'),
        plt.Rectangle((0,0),1,1, facecolor=loss_color, edgecolor='red', label='Loss Functions')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.02), fontsize=12)
    
    plt.tight_layout()
    save_path = "plots/final_presentations/pinn_architecture.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ PINN architecture diagram saved to {save_path}")

if __name__ == "__main__":
    create_pinn_architecture_diagram()

