# src/pinn_model.py
import torch
import torch.nn as nn
import numpy as np
import math
import config_heatsink as cfg 

# --- Layer Definitions ---
class AdaptiveSine(nn.Module):
    """Adaptive Sine activation with learnable frequency parameter 'a'."""
    def __init__(self, in_features):
        super(AdaptiveSine, self).__init__()
        self.a = nn.Parameter(torch.ones(1, in_features) * 10.0) 

    def forward(self, x):
        return torch.sin(self.a * x)

class FourierFeatureEmbedding(nn.Module):
    """Creates Fourier feature embedding."""
    def __init__(self, input_dims, embed_dims, scale=10.0):
        super(FourierFeatureEmbedding, self).__init__()
        self.input_dims = input_dims
        self.embed_dims = embed_dims
        self.scale = scale
        self.register_buffer('B', torch.randn(input_dims, embed_dims // 2) * self.scale)

    def forward(self, x):
        B_matrix = self.B.to(x.device) 
        x_proj = x @ B_matrix * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# --- Initialization Functions ---
def sine_init(m, w0=6.0):
    """Initializes weights for linear layers preceding sine activations with w0."""
    if isinstance(m, nn.Linear):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        w_std = math.sqrt(6.0 / fan_in) / w0
        nn.init.uniform_(m.weight, -w_std, w_std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def first_layer_sine_init(m):
    """Initializes weights specifically for the first linear layer."""
    if isinstance(m, nn.Linear):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        w_std = 1.0 / fan_in
        nn.init.uniform_(m.weight, -w_std, w_std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# --- The Upgraded Base PINN Class ---
class PINN(nn.Module):
    """Base PINN class with Fourier Features and Adaptive Sine."""
    def __init__(self, num_layers=cfg.MLP_LAYERS, hidden_size=cfg.MLP_HIDDEN):
        super(PINN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Input has 4 dimensions (t, x_norm, y_norm, z_norm)
        self.embedding = FourierFeatureEmbedding(input_dims=4, embed_dims=hidden_size) 
        
        layers = []
        # First layer (embedding -> hidden)
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(AdaptiveSine(hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(AdaptiveSine(hidden_size))
            
        # Final output layer (predicts NORMALIZED temperature)
        layers.append(nn.Linear(hidden_size, 1)) 
        
        self.net = nn.Sequential(*layers)
        
        # Apply SIREN initialization
        print("Applying SIREN-style initialization...")
        self.net[0].apply(first_layer_sine_init)  # Special init for first layer
        for i in range(2, len(layers), 2):  # Linear layers at even indices
            if i < len(layers) - 1:  # Skip the output layer
                self.net[i].apply(lambda m: sine_init(m, w0=6.0))  # w0=6 for hidden layers
        print("Initialization applied.")

    def forward(self, t, x_norm, y_norm, z_norm): # Expects NORMALIZED coords
        # Assumes t=0 input for steady state
        inputs = torch.cat([t, x_norm, y_norm, z_norm], dim=1)
        embedded_inputs = self.embedding(inputs)
        # Network predicts NORMALIZED temp [0,1]
        norm_output = self.net(embedded_inputs) 
        return norm_output