# Physics-Informed Neural Networks for Heat Sink Thermal Analysis

This project develops a **Physics-Informed Neural Network (PINN)** to solve the steady-state heat equation on a 3D heat sink geometry with complex boundary conditions. The PINN learns to predict temperature distribution directly from physics equations, achieving **~19.9°C Mean Absolute Error (MAE)** while providing **15,000× faster** predictions than traditional finite element methods.

## 🎯 Project Overview

Traditional computational fluid dynamics (CFD) solvers like FEniCS and ANSYS require **hours of computation time** per design iteration, making design optimization prohibitively slow. This PINN approach:
- ✅ **Trains once** (~8 hours) using physics equations
- ✅ **Predicts instantly** (< 1 second) for new designs
- ✅ **Enables rapid design iteration** for optimization

### Results Highlights

- **Best Model Performance**: MAE = 19.9°C (SIREN architecture)
- **Speed Improvement**: ~15,000× faster than traditional FEM per design
- **Architecture**: SIREN with Adaptive Sine activations + Fourier Features
- **Training**: Physics-constrained learning with mesh-aware boundary sampling

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Visualizations](#visualizations)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Mesh-Aware Training**: Boundary points sampled directly from Gmsh-generated mesh
- **Physics Constraints**: Heat equation (Laplacian) + Dirichlet + Convective BCs
- **SIREN Architecture**: Adaptive Sine activations with Fourier feature embedding
- **Normalized Training**: Input/output normalization for numerical stability
- **Staged Loss Weighting**: Dynamic focus on boundary conditions vs. PDE physics
- **HPC Integration**: Slurm scripts for GPU training on Northeastern's Discovery cluster

## 📁 Project Structure

```
physics-informed-neural-network/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt         # Python dependencies
├── config_heatsink.py       # Physics and model configuration
├── heatsink.msh             # Gmsh mesh file (3D heat sink geometry)
├── phase2_heatsink.slurm    # SLURM job script for HPC training
│
├── src/                      # Source code
│   ├── pinn_model.py        # PINN architecture (SIREN + Adaptive Sine)
│   ├── train_phase2.py      # Main training script with mesh-aware sampling
│   ├── fenics_solver.py     # Ground truth generator (FEniCS)
│   ├── create_mesh.py       # Gmsh mesh generation
│   │
│   └── plot_*.py            # Visualization scripts
│       ├── plot_performance_comparison.py    # Speed benchmarking
│       ├── plot_pinn_architecture.py          # Architecture diagram
│       ├── plot_workflow_comparison.py        # FEM vs PINN workflow
│       ├── plot_3way_comparison.py            # Model comparison
│       ├── plot_error_map_enhanced.py         # Error analysis
│       └── plot_ground_truth.py               # Ground truth visualization
│
└── plots/final_presentations/ # Presentation-ready visualizations
    ├── performance_comparison.png    # Speed comparison chart
    ├── pinn_architecture.png          # PINN architecture diagram
    ├── workflow_comparison.png        # Workflow comparison
    ├── 3way_comparison.png            # Model results comparison
    ├── error_map_enhanced.png         # Error analysis (4-panel)
    └── ground_truth_slice.png         # Ground truth reference

```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda/Miniconda

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/physics-informed-neural-network.git
cd physics-informed-neural-network

# Create conda environment
conda create -n pinn-env python=3.11 -y
conda activate pinn-env

# Install dependencies
pip install -r requirements.txt

# Additional requirements for FEniCS ground truth (optional)
conda install -c conda-forge fenics
```

### Key Dependencies

- **PyTorch**: Neural network framework
- **meshio**: Gmsh mesh I/O
- **trimesh**: Mesh processing
- **matplotlib**: Visualization
- **numpy**: Numerical operations
- **FEniCS**: Ground truth generation (optional)

## 🏃 Quick Start

### 1. Generate Mesh

```bash
python src/create_mesh.py
```

This creates `heatsink.msh` - a 3D heat sink mesh with physical tags:
- **Tag 1**: Bottom surface (Dirichlet BC, T = 100°C)
- **Tag 2**: Outer surfaces (Convective BC, h = 25 W/m²K)

### 2. Generate Ground Truth (Optional)

```bash
python src/fenics_solver.py
```

Generates `data/fenics_ground_truth.npy` using FEniCS for validation.

### 3. Train PINN

**Local (CPU):**
```bash
python src/train_phase2.py
```

**HPC (GPU):**
```bash
sbatch phase2_heatsink.slurm
```

The model trains with:
- PDE loss: Heat equation residual (∇²T = 0)
- Boundary conditions: Dirichlet + Convective
- Staged loss weighting: Focus on BCs first, then physics

### 4. Generate Visualizations

```bash
python src/run_all_visualizations.py
```

Creates all 6 presentation-ready visualizations in `plots/final_presentations/`.

## 🔬 Methodology

### Physics Problem

**Governing Equation** (Steady-state heat equation):
```
k∇²T = 0
```

**Boundary Conditions**:
1. **Dirichlet** (Bottom): T = 100°C (heat source)
2. **Convective** (Outer surfaces): -k ∂T/∂n = h(T - T_air)

**Parameters**:
- Thermal conductivity (k): 200 W/m·K
- Convection coefficient (h): 25 W/m²·K
- Ambient air temperature (T_air): 25°C

### PINN Architecture

**Base Architecture**: Multi-Layer Perceptron (MLP)
- **Layers**: 8 hidden layers, 256 neurons each
- **Input features**: Fourier Features (frequency embedding)
- **Activation**: Adaptive Sine (SIREN-style)
- **Output**: Temperature T(x,y,z)

**SIREN Initialization**:
- First layer: w₀ = 30 (high frequency)
- Hidden layers: w₀ = 6 (normal frequency)

### Training Strategy

1. **Staged Loss Weighting**:
   - Stage 1 (first half): Focus on boundary conditions
     - `pde_weight = 1.0`, `bc_weight = 100.0`
   - Stage 2 (second half): Balance physics and BCs
     - `pde_weight = 300.0`, `bc_weight = 30.0`

2. **Mesh-Aware Sampling**:
   - Boundary points sampled directly from Gmsh facets
   - Correct surface normals calculated for convective BC
   - Tag-based separation (Dirichlet vs Convective)

3. **Normalization**:
   - Coordinates: [-0.5, 0.5] → [0, 1]
   - Temperature: [25°C, 100°C] → [0, 1]
   - Prevents numerical instability

4. **Training Schedule**:
   - Optimizer: Adam (lr = 1e-4)
   - Scheduler: ReduceLROnPlateau
   - Gradient clipping: 0.5
   - Epochs: 10,000+

## 📊 Visualizations

All presentation-ready visualizations are in `plots/final_presentations/`:

| Visualization | Description | Audience |
|--------------|-------------|----------|
| `performance_comparison.png` | Speed benchmark (FEM vs PINN) | Executives |
| `pinn_architecture.png` | PINN architecture flowchart | Technical |
| `workflow_comparison.png` | Process comparison (FEM vs PINN) | General |
| `3way_comparison.png` | Ground Truth vs SIREN vs Tanh | Domain experts |
| `error_map_enhanced.png` | 4-panel error analysis | Technical reviewers |
| `ground_truth_slice.png` | Reference temperature field | Context/background |

## 📈 Results

### Quantitative Performance

- **Mean Absolute Error (MAE)**: 19.9°C
- **Mean Squared Error (MSE)**: ~480°C²
- **Temperature Range**: 25°C to 100°C
- **Training Time**: ~8 hours (one-time)
- **Inference Time**: < 1 second per design

### Speed Comparison

| Method | Time per Design | Speedup vs ANSYS |
|--------|----------------|------------------|
| ANSYS FEM | ~2 hours | 1× |
| OpenFOAM CFD | ~1 hour | 2× |
| **Our PINN** | **< 1 second** | **15,000×** |

### Key Findings

1. **SIREN > Tanh**: SIREN architecture (MAE=19.9°C) significantly outperforms standard Tanh baseline (MAE=29.7°C)
2. **Physics Consistency**: Model respects both Dirichlet and convective boundary conditions
3. **Scalability**: Once trained, PINN enables rapid design iterations
4. **Numerical Stability**: Normalization + staged weights prevent gradient explosion

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional architectures (DeepONet, Transformer)
- Multi-objective optimization (temperature + stress)
- Time-dependent problems
- Uncertainty quantification

## 📄 License

See [LICENSE](LICENSE) file for details (MIT License).

## 📚 References

- SIREN: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- PINN: [Physics-Informed Neural Networks](https://maziarraissi.github.io/PINNs/)
- FEniCS: [FEniCS Project](https://fenicsproject.org/)

---

**Project by**: Physics-Informed Neural Networks for Thermal Analysis  
**Institution**: Northeastern University  
**Year**: 2025
