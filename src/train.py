import torch
import config as cfg
from src.pinn_model import PINN


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    print("Starting training...")
    for epoch in range(cfg.NUM_EPOCHS):
        optimizer.zero_grad()

        t_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.T_MAX - cfg.T_MIN) + cfg.T_MIN
        x_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN
        y_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.Y_MAX - cfg.Y_MIN) + cfg.Y_MIN
        z_pde = torch.rand(cfg.N_PDE_POINTS, 1, device=device) * (cfg.Z_MAX - cfg.Z_MIN) + cfg.Z_MIN

        preds = model(t_pde, x_pde, y_pde, z_pde)
        loss = torch.mean(preds ** 2)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Loss: {loss.item():.6f}")

    print("Training finished.")


if __name__ == "__main__":
    main()


