import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config_heatsink as cfg
from src.pinn_model import PINN

class HeatSinkPINN(PINN):
    """PINN for heat sink steady-state heat transfer with finite difference BC"""
    
    def __init__(self):
        super().__init__(num_layers=cfg.MLP_LAYERS, hidden_size=cfg.MLP_HIDDEN)
        
    def pde_loss(self, x, y, z):
        """Heat equation: ∇²T = 0 (steady state)"""
        x.requires_grad_(True)
        y.requires_grad_(True) 
        z.requires_grad_(True)
        
        T = self.forward(torch.zeros_like(x), x, y, z)
        
        # First derivatives
        grad_T = torch.autograd.grad(
            T, [x, y, z], 
            grad_outputs=torch.ones_like(T), 
            create_graph=True, retain_graph=True
        )[0]
        
        grad_T_x = grad_T[:, 0:1]
        grad_T_y = grad_T[:, 1:2] 
        grad_T_z = grad_T[:, 2:3]
        
        # Second derivatives
        grad_T_xx = torch.autograd.grad(
            grad_T_x, x, 
            grad_outputs=torch.ones_like(grad_T_x),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_T_yy = torch.autograd.grad(
            grad_T_y, y, 
            grad_outputs=torch.ones_like(grad_T_y),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_T_zz = torch.autograd.grad(
            grad_T_z, z, 
            grad_outputs=torch.ones_like(grad_T_z),
            create_graph=True, retain_graph=True
        )[0]
        
        # Laplace equation: ∇²T = 0
        laplacian = grad_T_xx + grad_T_yy + grad_T_zz
        pde_residual = cfg.THERMAL_CONDUCTIVITY * laplacian
        
        return torch.mean(pde_residual**2)
    
    def convective_boundary_loss_finite_diff(self, x_bc, y_bc, z_bc, normals_bc, debug_epoch=None):
        """
        Convective boundary condition using finite difference approximation
        BC: -k * ∂T/∂n = h * (T - T_air)
        """
        epsilon = 1e-4  # Small distance for finite difference
        
        # Boundary points don't need gradients for sampling
        x_bc.requires_grad_(False)
        y_bc.requires_grad_(False)
        z_bc.requires_grad_(False)
        
        # Create interior points slightly offset from boundary along negative normal
        x_int = (x_bc - epsilon * normals_bc[:, 0:1]).requires_grad_(True)
        y_int = (y_bc - epsilon * normals_bc[:, 1:2]).requires_grad_(True)
        z_int = (z_bc - epsilon * normals_bc[:, 2:3]).requires_grad_(True)
        
        # Boundary points need gradients for temperature calculation
        x_bc_grad = x_bc.clone().requires_grad_(True)
        y_bc_grad = y_bc.clone().requires_grad_(True)
        z_bc_grad = z_bc.clone().requires_grad_(True)
        
        # Get model predictions
        T_bc = self.forward(torch.zeros_like(x_bc_grad), x_bc_grad, y_bc_grad, z_bc_grad)
        T_int = self.forward(torch.zeros_like(x_int), x_int, y_int, z_int)
        
        # DEBUG: Check for NaN/Inf in predictions
        if debug_epoch is not None and debug_epoch % 100 == 0:
            print(f"\n=== DEBUG EPOCH {debug_epoch} - CONVECTIVE BC ===")
            print(f"Boundary points shape: {x_bc.shape}")
            print(f"Boundary coords range: x=[{x_bc.min().item():.3f}, {x_bc.max().item():.3f}], y=[{y_bc.min().item():.3f}, {y_bc.max().item():.3f}], z=[{z_bc.min().item():.3f}, {z_bc.max().item():.3f}]")
            print(f"Normals sample: {normals_bc[0].cpu().numpy()}")
            
            if torch.isnan(T_bc).any() or torch.isinf(T_bc).any():
                print("!!! WARNING: NaN or Inf detected in T_bc !!!")
            if torch.isnan(T_int).any() or torch.isinf(T_int).any():
                print("!!! WARNING: NaN or Inf detected in T_int !!!")
                
            print(f"T_bc stats: min={T_bc.min().item():.2f}, max={T_bc.max().item():.2f}, mean={T_bc.mean().item():.2f}")
            print(f"T_int stats: min={T_int.min().item():.2f}, max={T_int.max().item():.2f}, mean={T_int.mean().item():.2f}")
        
        # Finite difference approximation of normal derivative
        normal_derivative_approx = (T_bc - T_int) / epsilon
        
        # DEBUG: Check derivative calculation
        if debug_epoch is not None and debug_epoch % 100 == 0:
            if torch.isnan(normal_derivative_approx).any() or torch.isinf(normal_derivative_approx).any():
                print("!!! WARNING: NaN or Inf detected in normal_derivative_approx !!!")
            print(f"NormDeriv stats: min={normal_derivative_approx.min().item():.2f}, max={normal_derivative_approx.max().item():.2f}, mean={normal_derivative_approx.mean().item():.2f}")
        
        # Convective boundary condition residual: k * dT/dn + h * T_bc - h * T_AIR = 0
        bc_residual = cfg.K_MATERIAL * normal_derivative_approx + cfg.H_CONVECTION * T_bc - cfg.H_CONVECTION * cfg.T_AIR
        
        # DEBUG: Check residual calculation
        if debug_epoch is not None and debug_epoch % 100 == 0:
            if torch.isnan(bc_residual).any() or torch.isinf(bc_residual).any():
                print("!!! WARNING: NaN or Inf detected in bc_residual !!!")
            print(f"BC Residual stats: min={bc_residual.min().item():.2f}, max={bc_residual.max().item():.2f}, mean={bc_residual.mean().item():.2f}")
        
        loss = torch.mean(bc_residual**2)
        
        # DEBUG: Final loss check
        if debug_epoch is not None and debug_epoch % 100 == 0:
            if torch.isnan(loss):
                print("!!! BC LOSS IS NaN !!!")
            print(f"Convective BC Loss: {loss.item():.6f}")
        
        return loss
    
    def dirichlet_boundary_loss(self, x_bc, y_bc, z_bc, target_temp, debug_epoch=None):
        """Simple Dirichlet boundary condition: T = target_temp"""
        T_bc = self.forward(torch.zeros_like(x_bc), x_bc, y_bc, z_bc)
        
        # DEBUG: Check Dirichlet BC
        if debug_epoch is not None and debug_epoch % 100 == 0:
            print(f"\n=== DEBUG EPOCH {debug_epoch} - DIRICHLET BC ===")
            print(f"Dirichlet points shape: {x_bc.shape}")
            print(f"Dirichlet coords range: x=[{x_bc.min().item():.3f}, {x_bc.max().item():.3f}], y=[{y_bc.min().item():.3f}, {y_bc.max().item():.3f}], z=[{z_bc.min().item():.3f}, {z_bc.max().item():.3f}]")
            print(f"Target temp: {target_temp}")
            
            if torch.isnan(T_bc).any() or torch.isinf(T_bc).any():
                print("!!! WARNING: NaN or Inf detected in T_bc (Dirichlet) !!!")
            print(f"T_bc stats: min={T_bc.min().item():.2f}, max={T_bc.max().item():.2f}, mean={T_bc.mean().item():.2f}")
        
        loss = torch.mean((T_bc - target_temp)**2)
        
        if debug_epoch is not None and debug_epoch % 100 == 0:
            if torch.isnan(loss):
                print("!!! DIRICHLET BC LOSS IS NaN !!!")
            print(f"Dirichlet BC Loss: {loss.item():.6f}")
        
        return loss

def sample_boundary_points_with_normals_debug(n_points, device, debug_epoch=None):
    """Sample boundary points with their outward normal vectors - DEBUG VERSION"""
    n_per_surface = n_points // 4  # 4 surfaces: top, sides
    
    # Top surface (z=Z_MAX): convective BC
    x_top = torch.rand(n_per_surface, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN
    y_top = torch.rand(n_per_surface, 1, device=device) * (cfg.Y_MAX - cfg.Y_MIN) + cfg.Y_MIN
    z_top = torch.full((n_per_surface, 1), cfg.Z_MAX, device=device)
    normals_top = torch.zeros(n_per_surface, 3, device=device)
    normals_top[:, 2] = 1.0  # Outward normal in +z direction
    
    # DEBUG: Check top surface sampling
    if debug_epoch is not None and debug_epoch % 100 == 0:
        print(f"\n=== DEBUG EPOCH {debug_epoch} - BOUNDARY SAMPLING ===")
        print(f"Top points Z check: min={z_top.min().item():.4f}, max={z_top.max().item():.4f}")  # Should be Z_MAX
        print(f"Top normals check (sample): {normals_top[0].cpu().numpy()}")  # Should be [0, 0, 1]
    
    # Side surfaces: convective BC
    n_side = n_per_surface // 4
    x_sides = []
    y_sides = []
    z_sides = []
    normals_sides = []
    
    # x=0 surface (normal in -x direction)
    x_sides.append(torch.zeros(n_side, 1, device=device))
    y_sides.append(torch.rand(n_side, 1, device=device) * (cfg.Y_MAX - cfg.Y_MIN) + cfg.Y_MIN)
    z_sides.append(torch.rand(n_side, 1, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN)
    normals = torch.zeros(n_side, 3, device=device)
    normals[:, 0] = -1.0  # -x direction
    normals_sides.append(normals)
    
    # x=X_MAX surface (normal in +x direction)
    x_sides.append(torch.full((n_side, 1), cfg.X_MAX, device=device))
    y_sides.append(torch.rand(n_side, 1, device=device) * (cfg.Y_MAX - cfg.Y_MIN) + cfg.Y_MIN)
    z_sides.append(torch.rand(n_side, 1, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN)
    normals = torch.zeros(n_side, 3, device=device)
    normals[:, 0] = 1.0  # +x direction
    normals_sides.append(normals)
    
    # y=0 surface (normal in -y direction)
    x_sides.append(torch.rand(n_side, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN)
    y_sides.append(torch.zeros(n_side, 1, device=device))
    z_sides.append(torch.rand(n_side, 1, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN)
    normals = torch.zeros(n_side, 3, device=device)
    normals[:, 1] = -1.0  # -y direction
    normals_sides.append(normals)
    
    # y=Y_MAX surface (normal in +y direction)
    x_sides.append(torch.rand(n_side, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN)
    y_sides.append(torch.full((n_side, 1), cfg.Y_MAX, device=device))
    z_sides.append(torch.rand(n_side, 1, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN)
    normals = torch.zeros(n_side, 3, device=device)
    normals[:, 1] = 1.0  # +y direction
    normals_sides.append(normals)
    
    # Concatenate
    x_sides = torch.cat(x_sides, dim=0)
    y_sides = torch.cat(y_sides, dim=0)
    z_sides = torch.cat(z_sides, dim=0)
    normals_sides = torch.cat(normals_sides, dim=0)
    
    # Bottom surface (z=0): Dirichlet BC
    x_bottom = torch.rand(n_per_surface, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN
    y_bottom = torch.rand(n_per_surface, 1, device=device) * (cfg.Y_MAX - cfg.Y_MIN) + cfg.Y_MIN
    z_bottom = torch.zeros(n_per_surface, 1, device=device)
    
    # DEBUG: Check bottom surface sampling
    if debug_epoch is not None and debug_epoch % 100 == 0:
        print(f"Bottom points Z check: min={z_bottom.min().item():.4f}, max={z_bottom.max().item():.4f}")  # Should be 0.0
        print(f"Side points shapes: x={x_sides.shape}, y={y_sides.shape}, z={z_sides.shape}")
        print(f"Side normals sample: {normals_sides[0].cpu().numpy()}")
    
    return {
        'convective': (x_top, y_top, z_top, normals_top, x_sides, y_sides, z_sides, normals_sides),
        'dirichlet': (x_bottom, y_bottom, z_bottom)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = HeatSinkPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5)
    
    # Training loop - SHORT DEBUG VERSION
    print("Starting Phase 2 training with DEBUG...")
    losses = []
    
    for epoch in range(500):  # Short run for debugging
        optimizer.zero_grad()
        
        # PDE points
        x_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN
        y_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.Y_MAX - cfg.Y_MIN) + cfg.Y_MIN
        z_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN
        
        # Boundary points with normals
        boundary_data = sample_boundary_points_with_normals_debug(cfg.N_BOUNDARY_POINTS, device, debug_epoch=epoch)
        convective_data = boundary_data['convective']
        dirichlet_data = boundary_data['dirichlet']
        
        x_top, y_top, z_top, normals_top, x_sides, y_sides, z_sides, normals_sides = convective_data
        x_bottom, y_bottom, z_bottom = dirichlet_data
        
        # Compute losses with debug info
        pde_loss = model.pde_loss(x_pde, y_pde, z_pde)
        
        # Convective boundary losses (finite difference)
        top_loss = model.convective_boundary_loss_finite_diff(x_top, y_top, z_top, normals_top, debug_epoch=epoch)
        sides_loss = model.convective_boundary_loss_finite_diff(x_sides, y_sides, z_sides, normals_sides, debug_epoch=epoch)
        
        # Dirichlet boundary loss (bottom surface)
        bottom_loss = model.dirichlet_boundary_loss(x_bottom, y_bottom, z_bottom, cfg.BASE_TEMP, debug_epoch=epoch)
        
        bc_loss = top_loss + sides_loss + bottom_loss
        
        # Check for NaN values
        if torch.isnan(pde_loss):
            print(f"PDE Loss NaN at epoch {epoch}, skipping...")
            continue
            
        if torch.isnan(bc_loss):
            print(f"BC Loss NaN at epoch {epoch}, checking components...")
            if torch.isnan(top_loss):
                print(f"  Top loss is NaN")
            if torch.isnan(sides_loss):
                print(f"  Sides loss is NaN")
            if torch.isnan(bottom_loss):
                print(f"  Bottom loss is NaN")
            continue
        
        total_loss = pde_loss + bc_loss
        
        # Check for NaN in total loss
        if torch.isnan(total_loss):
            print(f"Total Loss NaN at epoch {epoch}, skipping...")
            continue
            
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(total_loss)
        
        losses.append(total_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/500], "
                  f"Total Loss: {total_loss.item():.6f}, "
                  f"PDE Loss: {pde_loss.item():.6f}, "
                  f"BC Loss: {bc_loss.item():.6f}")
    
    print("Debug training completed!")

if __name__ == "__main__":
    main()
