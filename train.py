import torch
from pinn_model import PINN


# --- Training Parameters ---
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5000  # Can be increased on HPC
N_PDE_POINTS = 10000
N_BOUNDARY_POINTS = 2500
N_INITIAL_POINTS = 5000

# --- Problem Domain (must match FDM) ---
T_MIN, T_MAX = 0.0, 20.0
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize the Model
    model = PINN(num_layers=6, hidden_size=128).to(device)

    # 2. Initialize the Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        # PDE points (inside the domain)
        t_pde = torch.rand(N_PDE_POINTS, 1, device=device) * (T_MAX - T_MIN) + T_MIN
        x_pde = torch.rand(N_PDE_POINTS, 1, device=device) * (X_MAX - X_MIN) + X_MIN
        y_pde = torch.rand(N_PDE_POINTS, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
        z_pde = torch.rand(N_PDE_POINTS, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN

        # Placeholder forward and dummy loss
        preds = model(t_pde, x_pde, y_pde, z_pde)
        loss = torch.mean(preds ** 2)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}")

    print("Training finished.")


