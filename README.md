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
## Final Temperature Distribution
These plots show the final temperature state of the 3D cube after the simulation has run.

- Histogram of Final Temperatures (Left): This chart shows the distribution of temperatures across all points in the cube. The massive spike near 25-30°C tells us that the vast majority of the cube remains cool, close to the initial and boundary temperature. Only a small fraction of points reach higher temperatures, indicating a very localized heating effect.

- Center Z-Slice Visualization (Right): This heatmap is a 2D cross-section through the middle of the cube. It visually confirms what the histogram suggests: a distinct hotspot is located right in the center where the heat source is active. The temperature smoothly decreases outwards towards the cooler boundaries, creating a clear thermal gradient.

## Temperature Gradient Distribution
These plots analyze the rate of change of temperature, which is crucial for understanding the problem's complexity for the PINN.

- Histogram of Gradient Magnitudes (Left): This chart shows how "steep" the temperature changes are. The Y-axis is on a log scale, meaning the first bar is orders of magnitude larger than the others. This indicates that most of the cube is thermally stable (gradient is near zero). Significant temperature changes occur in a relatively small number of points.

- Gradient Magnitude at Center Z-Slice (Right): This heatmap shows where the temperature is changing most rapidly. Interestingly, the highest gradients (the bright yellow ring) are not at the absolute center but around the hotspot. This "ring" represents the area of maximum heat flux, which is the most challenging region for the neural network to accurately model.
