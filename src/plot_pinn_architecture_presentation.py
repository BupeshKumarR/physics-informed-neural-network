"""
Enhanced PINN Architecture Diagram for Presentations
Optimized for readability in PowerPoint slides
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def create_presentation_pinn_diagram():
    """Creates a clean, readable PINN architecture diagram for presentations"""
    
    Path("plots").mkdir(exist_ok=True)
    Path("plots/final_presentations").mkdir(exist_ok=True, parents=True)
    
    # Larger figure for better resolution
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Professional color scheme
    input_color = '#E3F2FD'      # Light blue
    network_color = '#FFF3E0'    # Light orange
    physics_color = '#F3E5F5'    # Light purple
    loss_color = '#FFEBEE'       # Light red
    output_color = '#E8F5E9'     # Light green
    
    # Title
    ax.text(10, 11.2, 'Physics-Informed Neural Network Architecture', 
            ha='center', va='center', fontsize=28, fontweight='bold')
    ax.text(10, 10.5, 'Combining Neural Networks with Physics Laws', 
            ha='center', va='center', fontsize=20, style='italic', color='#555')
    
    # 1. INPUT - Left side
    input_box = FancyBboxPatch((1, 4.5), 2.5, 2.5, 
                              boxstyle="round,pad=0.15", 
                              facecolor=input_color, 
                              edgecolor='#1976D2',
                              linewidth=3)
    ax.add_patch(input_box)
    ax.text(2.25, 6.2, 'Input', ha='center', va='center', 
            fontsize=20, fontweight='bold')
    ax.text(2.25, 5.5, 'Coordinates', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    ax.text(2.25, 4.9, '(x, y, z)', ha='center', va='center', 
            fontsize=16, style='italic', color='#555')
    
    # 2. NEURAL NETWORK - Center
    network_box = FancyBboxPatch((5.5, 3.5), 4, 4.5, 
                                 boxstyle="round,pad=0.2", 
                                 facecolor=network_color, 
                                 edgecolor='#F57C00',
                                 linewidth=3)
    ax.add_patch(network_box)
    ax.text(7.5, 7.3, 'Neural Network', ha='center', va='center', 
            fontsize=22, fontweight='bold')
    ax.text(7.5, 6.7, '(MLP)', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    ax.text(7.5, 6.1, 'Adaptive Sine', ha='center', va='center', 
            fontsize=16, style='italic', color='#555')
    ax.text(7.5, 5.6, 'Fourier Features', ha='center', va='center', 
            fontsize=16, style='italic', color='#555')
    
    # Network layers
    layer_y = [5.0, 4.5, 4.0]
    layers = ['Hidden Layers', 'with Physics-', 'Aware Activations']
    for y, text in zip(layer_y, layers):
        ax.text(7.5, y, text, ha='center', va='center', fontsize=14, color='#666')
    
    # 3. OUTPUT - Right of network
    output_box = FancyBboxPatch((11, 4.5), 2.5, 2.5, 
                               boxstyle="round,pad=0.15", 
                               facecolor=output_color, 
                               edgecolor='#388E3C',
                               linewidth=3)
    ax.add_patch(output_box)
    ax.text(12.25, 6.2, 'Predicted', ha='center', va='center', 
            fontsize=19, fontweight='bold')
    ax.text(12.25, 5.6, 'Temperature', ha='center', va='center', 
            fontsize=19, fontweight='bold')
    ax.text(12.25, 4.95, 'T(x, y, z)', ha='center', va='center', 
            fontsize=17, style='italic', color='#555')
    
    # 4. BOUNDARY CONDITIONS - Top right
    bc_box = FancyBboxPatch((15, 7), 3.5, 2, 
                           boxstyle="round,pad=0.15", 
                           facecolor=physics_color, 
                           edgecolor='#7B1FA2',
                           linewidth=2.5)
    ax.add_patch(bc_box)
    ax.text(16.75, 8.5, 'Boundary', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    ax.text(16.75, 8.0, 'Conditions', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    ax.text(16.75, 7.4, 'T = T₀', ha='center', va='center', 
            fontsize=16, style='italic', color='#555')
    
    # 5. DATA LOSS - Below BC
    data_loss_box = FancyBboxPatch((15, 4.8), 3.5, 1.5, 
                                   boxstyle="round,pad=0.15", 
                                   facecolor=loss_color, 
                                   edgecolor='#D32F2F',
                                   linewidth=2.5)
    ax.add_patch(data_loss_box)
    ax.text(16.75, 5.85, 'Data Loss', ha='center', va='center', 
            fontsize=19, fontweight='bold', color='#C62828')
    ax.text(16.75, 5.2, 'ℒ_data', ha='center', va='center', 
            fontsize=17, style='italic')
    
    # 6. PDE RESIDUAL - Bottom right
    pde_box = FancyBboxPatch((15, 2), 3.5, 2, 
                             boxstyle="round,pad=0.15", 
                             facecolor=physics_color, 
                             edgecolor='#7B1FA2',
                             linewidth=2.5)
    ax.add_patch(pde_box)
    ax.text(16.75, 3.5, 'PDE Residual', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    ax.text(16.75, 2.9, 'Heat Equation', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    ax.text(16.75, 2.35, '∇²T = 0', ha='center', va='center', 
            fontsize=17, style='italic', color='#555')
    
    # 7. PHYSICS LOSS - Below PDE
    physics_loss_box = FancyBboxPatch((15, 0.3), 3.5, 1.3, 
                                      boxstyle="round,pad=0.15", 
                                      facecolor=loss_color, 
                                      edgecolor='#D32F2F',
                                      linewidth=2.5)
    ax.add_patch(physics_loss_box)
    ax.text(16.75, 1.2, 'Physics Loss', ha='center', va='center', 
            fontsize=19, fontweight='bold', color='#C62828')
    ax.text(16.75, 0.6, 'ℒ_physics', ha='center', va='center', 
            fontsize=17, style='italic')
    
    # 8. COMBINED LOSS - Bottom center
    combined_box = FancyBboxPatch((5.5, 0.3), 6, 1.8, 
                                  boxstyle="round,pad=0.2", 
                                  facecolor='#FFCDD2', 
                                  edgecolor='#D32F2F',
                                  linewidth=4)
    ax.add_patch(combined_box)
    ax.text(8.5, 1.5, 'Total Loss', ha='center', va='center', 
            fontsize=22, fontweight='bold', color='#B71C1C')
    ax.text(8.5, 0.8, 'ℒ_total = λ₁·ℒ_data + λ₂·ℒ_physics', ha='center', va='center', 
            fontsize=17, style='italic')
    
    # ARROWS with labels
    # Input → Network
    arrow1 = FancyArrowPatch((3.5, 5.75), (5.5, 5.75), 
                             arrowstyle='->', mutation_scale=35, 
                             lw=3, color='#1976D2')
    ax.add_patch(arrow1)
    
    # Network → Output
    arrow2 = FancyArrowPatch((9.5, 5.75), (11, 5.75), 
                             arrowstyle='->', mutation_scale=35, 
                             lw=3, color='#F57C00')
    ax.add_patch(arrow2)
    
    # Output → BC (top path)
    arrow3 = FancyArrowPatch((13.5, 6.5), (15, 8), 
                             arrowstyle='->', mutation_scale=30, 
                             lw=2.5, color='#7B1FA2',
                             connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow3)
    ax.text(14, 7.5, 'Compare', ha='center', fontsize=14, 
            color='#7B1FA2', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # BC → Data Loss
    arrow4 = FancyArrowPatch((16.75, 7), (16.75, 6.3), 
                             arrowstyle='->', mutation_scale=30, 
                             lw=2.5, color='#D32F2F')
    ax.add_patch(arrow4)
    
    # Output → PDE (bottom path with auto-diff)
    arrow5 = FancyArrowPatch((13.5, 5), (15, 3), 
                             arrowstyle='->', mutation_scale=30, 
                             lw=2.5, color='#7B1FA2',
                             connectionstyle="arc3,rad=-0.3")
    ax.add_patch(arrow5)
    ax.text(14, 3.7, 'Auto-diff', ha='center', fontsize=14, 
            color='#7B1FA2', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # PDE → Physics Loss
    arrow6 = FancyArrowPatch((16.75, 2), (16.75, 1.6), 
                             arrowstyle='->', mutation_scale=30, 
                             lw=2.5, color='#D32F2F')
    ax.add_patch(arrow6)
    
    # Losses → Combined (curved arrows)
    arrow7 = FancyArrowPatch((15, 5.5), (11.5, 1.8), 
                             arrowstyle='->', mutation_scale=35, 
                             lw=3, color='#D32F2F',
                             connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow7)
    
    arrow8 = FancyArrowPatch((15, 0.9), (11.5, 0.9), 
                             arrowstyle='->', mutation_scale=35, 
                             lw=3, color='#D32F2F')
    ax.add_patch(arrow8)
    
    # Backprop arrow
    arrow9 = FancyArrowPatch((5.5, 1.2), (3, 4), 
                             arrowstyle='->', mutation_scale=35, 
                             lw=3, color='#FF6F00',
                             linestyle='--',
                             connectionstyle="arc3,rad=-0.4")
    ax.add_patch(arrow9)
    ax.text(3.5, 2.5, 'Backprop\n& Update', ha='center', fontsize=15, 
            color='#FF6F00', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', 
                     edgecolor='#FF6F00', linewidth=2))
    
    plt.tight_layout()
    save_path = "plots/final_presentations/pinn_architecture.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Presentation-ready PINN diagram saved to {save_path}")

if __name__ == "__main__":
    create_presentation_pinn_diagram()

