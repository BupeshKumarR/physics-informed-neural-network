# Physics-Informed Neural Networks for Heat Transfer Analysis

**Authors:** Bupesh Kumar, Mahadharsan Ravichandran  
**Course:** DS5500 - Deep Learning, Fall 2025  
**Institution:** Northeastern University

---

## Project Summary

This project develops **Physics-Informed Neural Networks (PINNs)** to solve heat transfer problems in 3D geometries, achieving significant speedups over traditional finite element methods while maintaining reasonable accuracy. The work consists of two phases:

- **Phase 1**: Transient heat equation in a 3D cube with internal heat source (23.6% relative error)
- **Phase 2**: Steady-state heat equation on a complex 3D heat sink geometry (18.8°C MAE)

**Key Results:**
- 15,000× faster than traditional FEM solvers
- 18.8°C MAE on complex heat sink geometry (25-100°C range)
- SIREN architecture outperforms standard Tanh baseline by 8.9%
- Physics-constrained learning without labeled training data

**[Final Report PDF](report.pdf)** *(link to your report when available)*

---

## Repository Structure

```
physics-informed-neural-network/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── config.py                    # Phase 1 configuration
├── config_heatsink.py          # Phase 2 configuration
├── phase2_heatsink.slurm       # SLURM job script for HPC
│
├── src/                         # Source code
│   ├── pinn_model.py           # PINN architecture (SIREN + Adaptive Sine)
│   ├── train_phase2.py         # Phase 2 training script
│   ├── fenics_solver.py        # FEniCS ground truth generator
│   ├── create_mesh.py          # Gmsh mesh generation
│   ├── plot_*.py               # Visualization scripts
│   └── run_all_visualizations.py
│
├── data/                        # Data files
│   └── fenics_ground_truth.npy # Phase 2 ground truth data
│
├── models/                      # Trained models
│   └── heatsink_pinn_model.pth # Best Phase 2 model
│
├── results/                     # Training results
│   └── phase2_results.npz     # Phase 2 evaluation metrics
│
└── plots/                       # Visualizations
    └── final_presentations/     # Presentation-ready figures
        ├── performance_comparison.png
        ├── pinn_architecture.png
        ├── workflow_comparison.png
        ├── 3d_comparison.png
        ├── 3way_comparison.png
        ├── error_map_enhanced.png
        └── ground_truth_slice.png
```

---

## Installation

### Prerequisites

- **Python**: 3.8+ (tested on 3.11)
- **GPU**: CUDA-capable GPU recommended (NVIDIA V100/A100 for HPC)
- **Conda/Miniconda**: For environment management
- **HPC Access**: Northeastern Discovery cluster (for GPU training)

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/physics-informed-neural-network.git
cd physics-informed-neural-network

# 2. Create conda environment
conda create -n pinn-env python=3.11 -y
conda activate pinn-env

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install FEniCS (optional, for ground truth generation)
conda install -c conda-forge fenics
```

---

## Dependencies

Key Python packages with versions (see `requirements.txt` for complete list):

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Neural network framework |
| `numpy` | ≥1.24.0 | Numerical operations |
| `matplotlib` | ≥3.7.0 | Visualization |
| `meshio` | ≥5.3.0 | Gmsh mesh I/O |
| `trimesh` | ≥3.20.0 | Mesh processing |
| `gmsh` | ≥4.11.0 | Mesh generation |
| `fenics` | (optional) | Ground truth solver |

**Note:** FEniCS is optional and only needed if generating ground truth data. The project includes pre-computed ground truth in `data/fenics_ground_truth.npy`.

---

## Usage / Reproduction

### Phase 1: 3D Cube with Heat Source

**Problem Setup:**
- Domain: 1m × 1m × 1m cube
- Time: 0-20 seconds (transient)
- Heat source: Internal cube region (Q = 200 W/m³)
- Initial condition: T(t=0) = 25°C
- Boundary condition: T = 25°C on all surfaces

**Training:**
```bash
# Phase 1 training (if available)
python src/train_phase1.py
```

**Expected Output:**
- Trained model checkpoint
- Results: ~23.6% relative error
- Training time: ~6 hours on GPU

---

### Phase 2: 3D Heat Sink (Steady-State)

**Problem Setup:**
- Geometry: T-shaped heat sink (base: 500×500×100mm, fin: 100×500×400mm)
- Temperature range: 25°C to 100°C
- Boundary conditions:
  - Bottom surface: T = 100°C (Dirichlet)
  - All other surfaces: Convective (h = 25 W/m²·K, T_air = 25°C)

#### Step 1: Generate Mesh

```bash
python src/create_mesh.py
```

**Expected Output:** `heatsink.msh` (Gmsh mesh file with physical tags)

#### Step 2: Generate Ground Truth (Optional)

```bash
python src/fenics_solver.py
```

**Expected Output:** `data/fenics_ground_truth.npy` (coordinates + temperatures)

**Note:** Pre-computed ground truth is already included in the repository.

#### Step 3: Train PINN

**Local (CPU - slower):**
```bash
python src/train_phase2.py
```

**HPC (GPU - recommended):**
```bash
sbatch phase2_heatsink.slurm
```

**Expected Output:**
- Trained model: `models/heatsink_pinn_model.pth`
- Results: `results/phase2_results.npz` (MAE, MSE, loss curves)
- Training time: ~1 hour on NVIDIA V100 GPU
- Final MAE: ~18.8°C

#### Step 4: Generate Visualizations

```bash
python src/run_all_visualizations.py
```

**Expected Output:** All presentation-ready figures in `plots/final_presentations/`

---

## Technical Specifications

### PINN Architecture

**Base Model:**
- Type: Multi-Layer Perceptron (MLP)
- Layers: 8 hidden layers
- Hidden size: 256 neurons per layer
- Activation: Adaptive Sine (SIREN-style, per-neuron learnable frequency)
- Input embedding: Fourier Features (256-dim, scale=10.0)
- Output: Temperature T(x,y,z)

**SIREN Initialization:**
- First layer: `w_std = 1.0 / fan_in`
- Hidden layers: `w_std = sqrt(6.0 / fan_in) / w0` where `w0 = 6.0`

**Training Hyperparameters:**
- Optimizer: Adam
- Learning rate: 1e-5 (Phase 2)
- Epochs: 8000 (Phase 2)
- Batch sizes:
  - PDE points: 2000
  - Boundary points: 800
- Loss weighting: Staged (Stage 1: PDE=0.1, BC=100.0; Stage 2: PDE=300.0, BC=30.0)
- Gradient clipping: max_norm = 0.5
- Scheduler: ReduceLROnPlateau (patience=2000, factor=0.7)

---

## Problem Setup Details

### Phase 1: Transient Heat Equation

**Governing PDE:**
```
∂T/∂t = α∇²T + Q
```

**Parameters:**
- Thermal diffusivity (α): 0.01 m²/s
- Heat source (Q): 200 W/m³ (central cube region)
- Domain: [0, 1]³ meters
- Time: [0, 20] seconds
- Initial condition: T(x,y,z,0) = 25°C
- Boundary condition: T = 25°C on all surfaces

### Phase 2: Steady-State Heat Equation

**Governing PDE:**
```
k∇²T = 0
```

**Boundary Conditions:**
1. **Dirichlet** (Bottom surface, Tag 1): T = 100°C
2. **Convective** (All other surfaces, Tag 2): -k ∂T/∂n = h(T - T_air)

**Parameters:**
- Thermal conductivity (k): 200 W/m·K (Aluminum)
- Convection coefficient (h): 25 W/m²·K
- Ambient temperature (T_air): 25°C
- Temperature range: 25°C to 100°C

**Geometry:**
- Base: 0.5m × 0.5m × 0.1m
- Fin: 0.1m × 0.5m × 0.4m (centered on base)
- Mesh size: 0.05m (coarse mesh for faster training)

---

## Hardware Requirements

**Minimum (Local CPU):**
- CPU: 4+ cores
- RAM: 8GB
- Storage: 2GB
- Training time: ~24+ hours (not recommended)

**Recommended (HPC GPU):**
- GPU: NVIDIA V100 or A100 (16GB+ VRAM)
- CPU: 8+ cores
- RAM: 32GB
- Storage: 5GB
- Training time: ~1 hour for Phase 2

**HPC Cluster:**
- **System**: Northeastern Discovery Cluster
- **Partition**: `gpu-short` or `gpu`
- **Module**: CUDA 12.1.1, GCC 9.3.0
- **Job time limit**: 4 hours (sufficient for Phase 2)

---

## Training Times

| Phase | Training Time | Hardware | Notes |
|-------|---------------|----------|-------|
| Phase 1 | ~6 hours | NVIDIA V100 | Transient problem, more complex |
| Phase 2 | ~1 hour | NVIDIA V100 | Steady-state, optimized architecture |
| FEniCS (ground truth) | ~2-4 hours | CPU | Mesh generation + solving + post-processing |

**Note:** Training times are one-time costs. Inference takes < 1 second per new design.

---

## Key Results

### Phase 1: 3D Cube
- **Relative Error**: 23.6%
- **Temperature Range**: 25°C to ~39°C
- **Training Time**: ~6 hours

### Phase 2: Heat Sink
- **Mean Absolute Error (MAE)**: 18.8°C
- **Temperature Range**: 25°C to 100°C (75°C span)
- **Relative Error**: ~25% (18.8/75)
- **SIREN vs Tanh**: 8.9% improvement (18.8°C vs 20.7°C MAE)
- **Training Time**: ~1 hour
- **Inference Speed**: < 1 second for full 50K-point mesh
- **Speedup vs FEM**: 15,000× faster

---

## Known Limitations

1. **Accuracy Gap**: Current MAE (18.8°C) exceeds industry requirement (±5°C) but demonstrates feasibility
2. **Single Geometry**: Model trained on one heat sink configuration; generalization to other geometries requires retraining
3. **Simplified Boundary Conditions**: Convective BCs approximated as Dirichlet in implementation
4. **Single Run**: Results from single training run; variance across random seeds not quantified
5. **No Experimental Validation**: Results validated against FEniCS only; no physical experiments

---

## Acknowledgments

- **Northeastern University HPC**: Discovery cluster GPU resources
- **Prof. Bemis**: Course instructor and guidance
- **FEniCS Project**: Open-source finite element library
- **SIREN Authors**: Sitzmann et al. for periodic activation functions

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Key References

- **PINN**: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

- **SIREN**: Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. *Advances in Neural Information Processing Systems*, 33.

- **Fourier Features**: Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., ... & Barron, J. T. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *Advances in Neural Information Processing Systems*, 33.

- **FEniCS**: Alnæs, M., et al. (2015). The FEniCS project version 1.5. *Archive of Numerical Software*, 3(100).

---

## Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
- **Solution**: Reduce `N_PDE_POINTS` and `N_BOUNDARY_POINTS` in config file
- **Alternative**: Use CPU mode (slower but works)

**Issue: NaN losses during training**
- **Solution**: Check normalization ranges in config file
- **Solution**: Reduce learning rate or increase gradient clipping

**Issue: FEniCS import error**
- **Solution**: Install via conda: `conda install -c conda-forge fenics`
- **Note**: FEniCS is optional; pre-computed ground truth is included

**Issue: Mesh generation fails**
- **Solution**: Ensure Gmsh is installed: `pip install gmsh`
- **Solution**: Check geometry parameters in `create_mesh.py`

**Issue: Model predictions out of range**
- **Solution**: Verify normalization in training matches inference
- **Solution**: Check temperature bounds in config file

---

## Visualization Examples

Sample outputs available in `plots/final_presentations/`:

- **3D Comparison**: Ground truth vs SIREN vs Tanh models (3D heat sink visualization)
- **Error Analysis**: 4-panel error distribution and spatial maps
- **Performance Chart**: Speed comparison (FEM vs PINN)
- **Architecture Diagram**: PINN architecture flowchart
- **Workflow Comparison**: Traditional FEM vs PINN process

---

## Pre-trained Models

Pre-trained Phase 2 model available:
- **File**: `models/heatsink_pinn_model.pth`
- **Architecture**: SIREN (8 layers × 256 neurons)
- **Performance**: 18.8°C MAE
- **Usage**: Load with `torch.load()` and use `HeatSinkPINN` class

---

**Last Updated:** October 2025  
**Project Status:** Complete (DS5500 Fall 2025)
