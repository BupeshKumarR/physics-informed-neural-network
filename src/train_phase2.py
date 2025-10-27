# src/train_phase2.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config_heatsink as cfg
from src.pinn_model import PINN
import meshio
import trimesh
import random

def normalize_coords(coords, min_val=cfg.COORD_NORM_MIN, max_val=cfg.COORD_NORM_MAX, norm_min=cfg.NORM_INPUT_MIN, norm_max=cfg.NORM_INPUT_MAX):
    return norm_min + (coords - min_val) * (norm_max - norm_min) / (max_val - min_val)

def normalize_temp(temps, min_val=cfg.T_NORM_MIN, max_val=cfg.T_NORM_MAX, norm_min=cfg.NORM_OUTPUT_MIN, norm_max=cfg.NORM_OUTPUT_MAX):
    return norm_min + (temps - min_val) * (norm_max - norm_min) / (max_val - min_val)

def denormalize_temp(norm_temps, min_val=cfg.T_NORM_MIN, max_val=cfg.T_NORM_MAX, norm_min=cfg.NORM_OUTPUT_MIN, norm_max=cfg.NORM_OUTPUT_MAX):
    return min_val + (norm_temps - norm_min) * (max_val - min_val) / (norm_max - norm_min)

# Loss calculation functions for the new PINN architecture
def calculate_pde_loss(model, x, y, z):
    """CORRECTED PDE Loss: Apply Laplacian with proper scaling - FIXED GRADIENT COMPUTATION"""
    print("\n--- DEBUG: Entering calculate_pde_loss ---")
    
    # Ensure input tensors are float32 and require gradients
    x = x.float().requires_grad_(True)
    y = y.float().requires_grad_(True)
    z = z.float().requires_grad_(True)

    # Normalize coordinates FIRST before setting requires_grad
    x_norm = normalize_coords(x).requires_grad_(True)
    y_norm = normalize_coords(y).requires_grad_(True)
    z_norm = normalize_coords(z).requires_grad_(True)
    t_pde = torch.zeros_like(x_norm)  # Time input

    # Get NORMALIZED output from base model forward pass
    T_norm = model(t_pde, x_norm, y_norm, z_norm)
    # # # # # # # # # # # # # # print(f"DEBUG: T_norm range: {T_norm.min().item():.3e} to {T_norm.max().item():.3e}")

    # Calculate derivatives w.r.t NORMALIZED coordinates
    try:
        # First derivatives - CORRECTED with proper grad_outputs
        first_derivatives = torch.autograd.grad(
            outputs=T_norm,  # Don't use .sum() here
            inputs=[x_norm, y_norm, z_norm],
            grad_outputs=torch.ones_like(T_norm),  # ESSENTIAL for proper gradients
            create_graph=True, 
            retain_graph=True  # ESSENTIAL for second derivatives
        )
        
        # CORRECT indexing: first_derivatives[0] = grad w.r.t x_norm, etc.
        grad_norm_T_x = first_derivatives[0]
        grad_norm_T_y = first_derivatives[1] 
        grad_norm_T_z = first_derivatives[2]
        
        # # # # # # # # # # # # # # print(f"DEBUG: grad_norm_T_x range: {grad_norm_T_x.min().item():.3e} to {grad_norm_T_x.max().item():.3e}")
        # # # # # # # # # # # # # # print(f"DEBUG: grad_norm_T_y range: {grad_norm_T_y.min().item():.3e} to {grad_norm_T_y.max().item():.3e}")
        # # # # # # # # # # # # # # print(f"DEBUG: grad_norm_T_z range: {grad_norm_T_z.min().item():.3e} to {grad_norm_T_z.max().item():.3e}")
        
        if torch.isnan(grad_norm_T_x).any(): 
            print("!!! DEBUG: NaN in grad_norm_T_x !!!")
        if torch.isnan(grad_norm_T_y).any(): 
            print("!!! DEBUG: NaN in grad_norm_T_y !!!")
        if torch.isnan(grad_norm_T_z).any(): 
            print("!!! DEBUG: NaN in grad_norm_T_z !!!")

        # Second derivatives - CORRECTED with proper grad_outputs
        grad_norm_T_xx = torch.autograd.grad(
            outputs=grad_norm_T_x, 
            inputs=x_norm, 
            grad_outputs=torch.ones_like(grad_norm_T_x),  # ESSENTIAL
            create_graph=True, 
            retain_graph=True
        )[0]
        # # # # # # # # # # # # # # print(f"DEBUG: grad_norm_T_xx range: {grad_norm_T_xx.min().item():.3e} to {grad_norm_T_xx.max().item():.3e}")
        if torch.isnan(grad_norm_T_xx).any(): 
            print("!!! DEBUG: NaN in grad_norm_T_xx !!!")

        grad_norm_T_yy = torch.autograd.grad(
            outputs=grad_norm_T_y, 
            inputs=y_norm, 
            grad_outputs=torch.ones_like(grad_norm_T_y),  # ESSENTIAL
            create_graph=True, 
            retain_graph=True
        )[0]
        # # # # # # # # # # # # # # print(f"DEBUG: grad_norm_T_yy range: {grad_norm_T_yy.min().item():.3e} to {grad_norm_T_yy.max().item():.3e}")
        if torch.isnan(grad_norm_T_yy).any(): 
            print("!!! DEBUG: NaN in grad_norm_T_yy !!!")

        grad_norm_T_zz = torch.autograd.grad(
            outputs=grad_norm_T_z, 
            inputs=z_norm, 
            grad_outputs=torch.ones_like(grad_norm_T_z),  # ESSENTIAL
            create_graph=False,  # CHANGED: Don't create graph for final derivative
            retain_graph=False  # CHANGED: Don't retain graph
        )[0]
        # # # # # # # # # # # # # # print(f"DEBUG: grad_norm_T_zz range: {grad_norm_T_zz.min().item():.3e} to {grad_norm_T_zz.max().item():.3e}")
        # Detach to break computation graph and avoid conflicts
        grad_norm_T_xx = grad_norm_T_xx.detach().requires_grad_(True)
        grad_norm_T_yy = grad_norm_T_yy.detach().requires_grad_(True)
        grad_norm_T_zz = grad_norm_T_zz.detach().requires_grad_(True)
        if torch.isnan(grad_norm_T_zz).any(): 
            print("!!! DEBUG: NaN in grad_norm_T_zz !!!")

    except RuntimeError as e:
        print(f"!!! DEBUG: AUTOGRAD ERROR during derivative calculation: {e} !!!")
        return torch.tensor(1e6, device=x.device)

    # Calculate Laplacian in NORMALIZED coordinates
    laplacian_norm = grad_norm_T_xx + grad_norm_T_yy + grad_norm_T_zz
    
    # --- PDE residual is the normalized Laplacian ---
    pde_residual_norm = laplacian_norm 
    
    # --- USE HUBER LOSS on the Normalized Residual ---
    huber_loss_fn = nn.HuberLoss(delta=0.5)  # Delta controls transition, 0.5 is a common start
    loss = huber_loss_fn(pde_residual_norm, torch.zeros_like(pde_residual_norm))
    # --- End Change ---
    # # # # # # # # # # # # # # print(f"DEBUG: pde_loss calculated: {loss.item():.6e}")

    # Check for zero loss
    if torch.abs(loss) < 1e-12 and loss.requires_grad:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: PDE Loss became numerically zero!")
        print(f"  Residual mean squared: {torch.mean(pde_residual_norm**2).item():.3e}")
        print(f"  Residual abs mean: {torch.mean(torch.abs(pde_residual_norm)).item():.3e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # print("--- DEBUG: Exiting calculate_pde_loss ---")
    return loss

class HeatSinkPINN(PINN):
    """Heat sink specific PINN with specialized loss functions"""
    
    def pde_loss(self, x, y, z):
        """Heat equation: k * ∇²T = 0 (steady state) - CORRECTED SCALING, MSE Loss"""
        return calculate_pde_loss(self, x, y, z)
    
    def dirichlet_boundary_loss(self, x, y, z, target_temp):
        """Dirichlet boundary condition loss"""
        return calculate_dirichlet_boundary_loss(self, x, y, z, target_temp)
    
    def convective_boundary_loss_finite_diff(self, x, y, z, normals):
        """Convective boundary condition loss using finite difference"""
        # For now, return a placeholder - this function needs to be implemented
        return torch.tensor(0.0, device=x.device, requires_grad=True)

def calculate_dirichlet_boundary_loss(model, x_bc, y_bc, z_bc, target_temp):
    """Dirichlet boundary condition loss"""
    # Normalize inputs before calling model
    x_bc_norm = normalize_coords(x_bc)
    y_bc_norm = normalize_coords(y_bc)
    z_bc_norm = normalize_coords(z_bc)
    
    # Get normalized output and denormalize
    T_norm_pred = model(torch.zeros_like(x_bc_norm), x_bc_norm, y_bc_norm, z_bc_norm)
    T_phys_pred = denormalize_temp(T_norm_pred)
    
    return torch.mean((T_phys_pred - target_temp)**2)

def sample_mesh_boundary_points_with_normals(mesh_file="heatsink.msh", n_points_total=5000, device='cuda'):
    """
    Samples points directly from mesh faces tagged with physical groups
    and calculates their correct outward normal vectors.
    """
    mesh = meshio.read(mesh_file)
    
    # Check available cell types
    
    # Find the correct cell type for surface elements
    surface_cell_type = None
    for cell_type in ['triangle', 'tri', 'tri3', 'quad', 'quad4']:
        if cell_type in mesh.cells_dict:
            surface_cell_type = cell_type
            break
    
    if surface_cell_type is None:
        raise ValueError(f"No suitable surface cell type found. Available: {list(mesh.cells_dict.keys())}")
    
    
    # Get surface cells
    surface_cells = mesh.cells_dict[surface_cell_type]
    
    # Access physical tags correctly
    if 'gmsh:physical' not in mesh.cell_data_dict:
        raise ValueError("No gmsh:physical data found in mesh")
    
    if surface_cell_type not in mesh.cell_data_dict['gmsh:physical']:
        raise ValueError(f"No physical data found for {surface_cell_type}")
    
    triangle_physicals = mesh.cell_data_dict['gmsh:physical'][surface_cell_type]
    
    # Separate faces based on physical tags
    bottom_faces_indices = np.where(triangle_physicals == 1)[0]
    convective_faces_indices = np.where(triangle_physicals == 2)[0]
    
    
    if len(bottom_faces_indices) == 0:
        print("Warning: No faces found with physical tag 1 (bottom_surface)")
    if len(convective_faces_indices) == 0:
        print("Warning: No faces found with physical tag 2 (convection_surfaces)")
        
    bottom_faces = surface_cells[bottom_faces_indices]
    convective_faces = surface_cells[convective_faces_indices]

    # Use trimesh to handle sampling and normal calculation
    mesh_trimesh = trimesh.Trimesh(vertices=mesh.points, faces=surface_cells)

    # Calculate face areas for weighted sampling
    bottom_face_areas = mesh_trimesh.area_faces[bottom_faces_indices]
    convective_face_areas = mesh_trimesh.area_faces[convective_faces_indices]
    
    total_bottom_area = np.sum(bottom_face_areas)
    total_convective_area = np.sum(convective_face_areas)
    total_area = total_bottom_area + total_convective_area

    # Determine number of points per group based on relative area
    n_bottom = int(n_points_total * (total_bottom_area / total_area)) if total_area > 0 else 0
    n_convective = n_points_total - n_bottom


    # Sample points and get face indices for normals
    if n_bottom > 0 and len(bottom_faces) > 0:
        bottom_points, bottom_face_idx = trimesh.sample.sample_surface_even(
            trimesh.Trimesh(vertices=mesh.points, faces=bottom_faces), n_bottom
        )
        # Get normals for the sampled faces
        bottom_normals = mesh_trimesh.face_normals[bottom_faces_indices[bottom_face_idx]]
    else:
         bottom_points, bottom_normals = np.empty((0,3)), np.empty((0,3))

    if n_convective > 0 and len(convective_faces) > 0:
        convective_points, convective_face_idx = trimesh.sample.sample_surface_even(
            trimesh.Trimesh(vertices=mesh.points, faces=convective_faces), n_convective
        )
        # Get normals for the sampled faces
        convective_normals = mesh_trimesh.face_normals[convective_faces_indices[convective_face_idx]]
    else:
        convective_points, convective_normals = np.empty((0,3)), np.empty((0,3))

    # Convert to tensors
    x_conv = torch.tensor(convective_points[:, 0:1], dtype=torch.float32, device=device)
    y_conv = torch.tensor(convective_points[:, 1:2], dtype=torch.float32, device=device)
    z_conv = torch.tensor(convective_points[:, 2:3], dtype=torch.float32, device=device)
    normals_conv = torch.tensor(convective_normals, dtype=torch.float32, device=device)
    
    x_dirichlet = torch.tensor(bottom_points[:, 0:1], dtype=torch.float32, device=device)
    y_dirichlet = torch.tensor(bottom_points[:, 1:2], dtype=torch.float32, device=device)
    z_dirichlet = torch.tensor(bottom_points[:, 2:3], dtype=torch.float32, device=device)

    return {
        'convective': (x_conv, y_conv, z_conv, normals_conv),
        'dirichlet': (x_dirichlet, y_dirichlet, z_dirichlet)
    }

def get_avg_grad_l1_norm(model, loss):
    """Helper function to compute the average L1 norm of gradients."""
    model.train()
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
    valid_grads = [g.detach() for g in grads if g is not None and g.requires_grad]
    if not valid_grads:
        return 0.0
    # Calculate L1 norm for each grad tensor, sum them up, divide by total num elements
    total_l1_norm = sum(torch.sum(torch.abs(g)) for g in valid_grads)
    total_elements = sum(g.numel() for g in valid_grads)
    return (total_l1_norm / total_elements).item() if total_elements > 0 else 0.0

# Remove get_current_base_temp - no longer needed
def get_current_base_temp(epoch, total_epochs, start_temp, final_temp):
    """Placeholder - curriculum not used."""
    return final_temp

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    
    # Load ground truth data
    print("Loading heat sink ground truth data...")
    data = np.load(cfg.GROUND_TRUTH_PATH)
    coords = data[:, :3]  # x, y, z coordinates
    temps_true = data[:, 3]  # temperature values
    
    print(f"Loaded {len(coords)} data points")
    print(f"Temperature range: {temps_true.min():.1f}°C to {temps_true.max():.1f}°C")
    
    
    # Initialize model
    model = HeatSinkPINN(num_layers=cfg.MLP_LAYERS, hidden_size=cfg.MLP_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, factor=0.7)
    
    print("Starting Training (Simplified Architecture + Staged Weights)...")
    losses = []
    
    # Curriculum Parameters
    initial_base_temp = cfg.T_AIR  # Start bottom at ambient
    final_base_temp = cfg.BASE_TEMP  # Target 100°C

    for epoch in range(cfg.NUM_EPOCHS):
        optimizer.zero_grad()
        model.train() 

        # Generate PDE points
        x_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN
        y_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.Y_MAX - cfg.Y_MIN) + cfg.Y_MIN
        z_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN
        
        # Generate boundary points
        boundary_data = sample_mesh_boundary_points_with_normals(n_points_total=cfg.N_BOUNDARY_POINTS, device=device)
        x_conv, y_conv, z_conv, normals_conv = boundary_data['convective']
        x_bottom, y_bottom, z_bottom = boundary_data['dirichlet']

        # Debug prints
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Boundary point shapes:")
            print(f"  Convective: {x_conv.shape}, normals: {normals_conv.shape}")
            print(f"  Dirichlet: {x_bottom.shape}")
        
        # --- CALCULATE LOSSES using separate functions ---
        pde_loss = calculate_pde_loss(model, x_pde, y_pde, z_pde)  # Uses MSE
        
        # Hot BC Loss - Use fixed BASE_TEMP (no curriculum)
        loss_bc_hot = calculate_dirichlet_boundary_loss(
            model, x_bottom, y_bottom, z_bottom, cfg.BASE_TEMP
        ) 
        
        # Cold BC Loss
        loss_bc_cold = calculate_dirichlet_boundary_loss(
            model, x_conv, y_conv, z_conv, cfg.T_AIR
        )
        
        # Combine boundary losses
        bc_loss = loss_bc_hot + loss_bc_cold
        
        # Check for NaN values
        if torch.isnan(pde_loss) or torch.isnan(loss_bc_hot) or torch.isnan(loss_bc_cold):
            print(f"NaN detected at epoch {epoch}. PDE={pde_loss.item()}, BC_Hot={loss_bc_hot.item()}, BC_Cold={loss_bc_cold.item()}. Skipping update.")
            continue
        
        # --- Manual Staged Weights (Refined for better balance) ---
        stage_epochs = cfg.NUM_EPOCHS // 2
        if epoch < stage_epochs:  # Stage 1: Focus on Boundaries first
            pde_weight = 0.1  # Very low PDE weight initially
            bc_weight = 100.0  # Strong BC constraint
        else:  # Stage 2: Balance PDE and BC learning
            pde_weight = 300.0  # Reduced to prevent overfitting (not extreme)
            bc_weight = 30.0   # Reduced to prevent overfitting (not too weak)

        total_loss = (pde_weight * pde_loss) + (bc_weight * bc_loss)
        # --- End Staged Weights --- 
                     
        # Check for NaN 
        if torch.isnan(total_loss):
            print(f"Total Loss NaN at epoch {epoch}, skipping...")
            continue 

        # Debug prints
        if (epoch + 1) % 100 == 0:
            
            # Check model output range
            with torch.no_grad():
                # Normalize coordinates for the new model
                x_bottom_norm = normalize_coords(x_bottom)
                y_bottom_norm = normalize_coords(y_bottom)
                z_bottom_norm = normalize_coords(z_bottom)
                
                # Get normalized temperature predictions
                T_norm_pred_sample = model(torch.zeros_like(x_bottom_norm), x_bottom_norm, y_bottom_norm, z_bottom_norm)
                
                # Denormalize to physical temperatures
                T_pred_sample = denormalize_temp(T_norm_pred_sample)
            
            # Check learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            
            print(f"Model output range: {T_pred_sample.min().item():.2f} to {T_pred_sample.max().item():.2f}")
            print(f"Learning rate: {current_lr:.2e}")

        print(f"Epoch {epoch+1} completed.")
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        scheduler.step(total_loss)
        
        losses.append(total_loss.item())
        
        # --- Update Print Statement ---
        if (epoch + 1) % 100 == 0: 
            stage = 1 if epoch < stage_epochs else 2
            print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS} S:{stage}], Loss: {total_loss.item():.4f}, "
                  f"PDE(r): {pde_loss.item():.4e}, BC(r): {bc_loss.item():.4f}, "
                  f"Wts P/BC: {pde_weight:.1f}/{bc_weight:.1f}")
    
    # Save model
    torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
    print(f"Model saved to {cfg.MODEL_SAVE_PATH}")
    
    # Evaluate on ground truth
    print("Evaluating model...")
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    with torch.no_grad():
        # Normalize coordinates for the new model
        x_norm = normalize_coords(coords_tensor[:, 0:1])
        y_norm = normalize_coords(coords_tensor[:, 1:2])
        z_norm = normalize_coords(coords_tensor[:, 2:3])
        
        # Get normalized temperature predictions
        temps_norm_pred = model(torch.zeros(len(coords), 1, device=device), x_norm, y_norm, z_norm)
        
        # Denormalize to physical temperatures
        temps_pred = denormalize_temp(temps_norm_pred).cpu().numpy().flatten()
    
    # Calculate metrics
    mse = np.mean((temps_pred - temps_true)**2)
    mae = np.mean(np.abs(temps_pred - temps_true))
    max_error = np.max(np.abs(temps_pred - temps_true))
    
    print("Evaluation Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}°C")
    print(f"Max Error: {max_error:.6f}°C")
    
    # Save results
    np.savez(cfg.RESULTS_PATH, 
             coords=coords, 
             temps_true=temps_true, 
             temps_pred=temps_pred,
             losses=losses,
             mse=mse, mae=mae, max_error=max_error)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss curve
    axes[0,0].plot(losses)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_yscale('log')
    
    # Scatter plot: True vs Predicted
    axes[0,1].scatter(temps_true, temps_pred, alpha=0.5, s=1)
    axes[0,1].plot([temps_true.min(), temps_true.max()], 
                   [temps_true.min(), temps_true.max()], 'r--')
    axes[0,1].set_xlabel('True Temperature (°C)')
    axes[0,1].set_ylabel('Predicted Temperature (°C)')
    axes[0,1].set_title('True vs Predicted')
    
    # Error distribution
    errors = temps_pred - temps_true
    axes[1,0].hist(errors, bins=50, alpha=0.7)
    axes[1,0].set_xlabel('Prediction Error (°C)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Error Distribution')
    
    # 3D scatter plot
    ax = axes[1,1]
    scatter = ax.scatter(coords[:, 0], coords[:, 2], c=temps_pred, 
                        cmap='hot', s=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('PINN Prediction (Y=0.25 slice)')
    plt.colorbar(scatter, ax=ax, label='Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig('plots/phase2_results.png', dpi=150, bbox_inches='tight')
    print("Results visualization saved to plots/phase2_results.png")
    
    print("Phase 2 training completed successfully!")

if __name__ == "__main__":
    main()