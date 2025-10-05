# physics-informed-neural-network
This project proposes using Physics-Informed Neural Networks (PINNs) to model heat exchange with internal heating and convective boundary cooling.

## Workflow
The project follows a clear, multi-step workflow:

1.  **Configuration (`config.py`):** All physical, simulation, and model parameters are defined in a single, centralized file.
2.  **Data Generation (`src/fdm_solver.py`):** A Finite Difference Method (FDM) solver simulates the heat equation to generate a high-fidelity "ground truth" dataset for validation.
3.  **Data Analysis (`src/eda.py`):** An exploratory data analysis script inspects the ground truth data to understand its properties, such as temperature range and gradient magnitudes.
4.  **Model Training (`src/train.py`):** The core PINN model is trained. The loss function is a combination of the PDE residual, initial conditions, and boundary conditions.
5.  **Evaluation:** After training, the model is evaluated against the FDM ground truth to measure its accuracy.

## Project Structure

```
pinn-heat-transfer/
├── config.py              # Central configuration for all scripts
├── requirements.txt         # Python package dependencies
└── src/
    ├── fdm_solver.py        # Generates the ground truth data
    ├── eda.py               # Analyzes the generated data
    ├── pinn_model.py        # Defines the PyTorch PINN model
    └── train.py             # Main script to train the PINN
```
