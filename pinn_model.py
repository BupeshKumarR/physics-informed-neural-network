import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    A simple fully-connected neural network for the PINN.
    Maps (t, x, y, z) -> T.
    """

    def __init__(self, num_layers: int = 6, hidden_size: int = 128) -> None:
        """
        Initialize the neural network.

        Args:
            num_layers: The number of hidden layers.
            hidden_size: The number of neurons in each hidden layer.
        """
        super().__init__()

        layers = []
        # Input layer (4 -> hidden)
        layers.append(nn.Linear(4, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer (hidden -> 1)
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            t: Tensor of shape (N, 1) for time.
            x: Tensor of shape (N, 1) for x-coordinate.
            y: Tensor of shape (N, 1) for y-coordinate.
            z: Tensor of shape (N, 1) for z-coordinate.
        Returns:
            Tensor of shape (N, 1) representing predicted temperature.
        """
        inputs = torch.cat([t, x, y, z], dim=1)
        return self.net(inputs)


