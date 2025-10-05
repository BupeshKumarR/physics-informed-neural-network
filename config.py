# Central configuration for the 3D transient heat problem

# Material properties
ALPHA = 0.01  # Thermal diffusivity (m^2/s)

# Geometry
LX, LY, LZ = 1.0, 1.0, 1.0  # Domain size in meters

# Discretization
NX, NY, NZ = 32, 32, 32

# Time
T_FINAL = 20.0  # seconds
STABILITY_FACTOR = 0.1  # safety factor for explicit scheme

# Initial and boundary conditions
T_INITIAL = 25.0
T_BOUNDARY = 25.0

# Heat source (central cube region indices relative to grid)
Q_VALUE = 200.0  # deg C per second
SRC_HALF_WIDTH = 2  # half-size in voxels around center

# File paths
GROUND_TRUTH_PATH = "fdm_ground_truth.npy"

# PINN training domain bounds (must mirror FDM domain)
T_MIN, T_MAX = 0.0, T_FINAL
X_MIN, X_MAX = 0.0, LX
Y_MIN, Y_MAX = 0.0, LY
Z_MIN, Z_MAX = 0.0, LZ

# PINN model hyperparameters
MLP_LAYERS = 6
MLP_HIDDEN = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5000
N_PDE_POINTS = 10000
N_BOUNDARY_POINTS = 2500
N_INITIAL_POINTS = 5000


