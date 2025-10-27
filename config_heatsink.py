# Heat sink configuration for Phase 2

# Heat sink geometry bounds
X_MIN, X_MAX = 0.0, 0.5  # Base width
Y_MIN, Y_MAX = 0.0, 0.5  # Base depth  
Z_MIN, Z_MAX = 0.0, 0.5  # Total height (base + fin)

# Time domain (steady state)
T_MIN, T_MAX = 0.0, 1.0

# PINN model hyperparameters (SIREN - stabilize for better accuracy)
MLP_LAYERS = 8  # Moderate size for stability
MLP_HIDDEN = 256  # Moderate capacity
LEARNING_RATE = 1e-5  # Lower LR for SIREN stability
NUM_EPOCHS = 8000  # Sufficient epochs with early stopping
N_PDE_POINTS = 2000  # Balanced for stability
N_BOUNDARY_POINTS = 800  # Balanced for stability
N_INITIAL_POINTS = 10000

# Heat sink specific parameters
BASE_TEMP = 100.0  # Bottom surface temperature
AMBIENT_TEMP = 25.0  # Ambient temperature
THERMAL_CONDUCTIVITY = 200.0  # Aluminum
CONVECTION_COEFF = 25.0
K_MATERIAL = 200.0  # Thermal conductivity
H_CONVECTION = 25.0  # Convection coefficient
T_AIR = 25.0  # Ambient temperature

# Normalization Ranges
T_NORM_MIN, T_NORM_MAX = 25.0, 100.0
COORD_NORM_MIN, COORD_NORM_MAX = 0.0, 0.5
NORM_OUTPUT_MIN, NORM_OUTPUT_MAX = 0.0, 1.0
NORM_INPUT_MIN, NORM_INPUT_MAX = -1.0, 1.0

# File paths
GROUND_TRUTH_PATH = "data/fenics_ground_truth.npy"
MODEL_SAVE_PATH = "models/heatsink_pinn_model.pth"
RESULTS_PATH = "results/phase2_results.npz"