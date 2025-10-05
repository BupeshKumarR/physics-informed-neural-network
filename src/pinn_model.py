import torch
import torch.nn as nn
import config as cfg


class PINN(nn.Module):
    def __init__(self, num_layers: int = cfg.MLP_LAYERS, hidden_size: int = cfg.MLP_HIDDEN) -> None:
        super().__init__()
        layers = [nn.Linear(4, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([t, x, y, z], dim=1)
        return self.net(inputs)


