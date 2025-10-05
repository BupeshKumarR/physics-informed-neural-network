import numpy as np
import matplotlib.pyplot as plt
import time


# --- Simulation Parameters ---
# Material Properties
ALPHA = 0.01  # Thermal diffusivity (m^2/s)

# Geometry
LX, LY, LZ = 1.0, 1.0, 1.0  # Domain size in meters

# Discretization
NX, NY, NZ = 32, 32, 32   # Number of grid points along each axis
DX, DY, DZ = LX / NX, LY / NY, LZ / NZ  # Grid spacing

# Time Stepping
T_FINAL = 20.0  # Total simulation time in seconds

# --- CFL Condition for Stability ---
# For 3D heat equation, dt <= 1 / (2 * alpha * (1/dx^2 + 1/dy^2 + 1/dz^2))
STABILITY_FACTOR = 0.1  # Use a factor less than 1 for safety
DT = STABILITY_FACTOR * (1.0 / (2.0 * ALPHA * (1.0 / DX**2 + 1.0 / DY**2 + 1.0 / DZ**2)))
NT = int(T_FINAL / DT)

# Initial and Boundary Conditions
T_INITIAL = 25.0  # Initial temperature of the cube (deg C)
T_BOUNDARY = 25.0  # Temperature at the walls (deg C)

# Heat Source
Q_VALUE = 200.0  # Heat generation rate (deg C per second)
# Define the source region (a small box in the center)
SRC_X_START, SRC_X_END = NX // 2 - 2, NX // 2 + 2
SRC_Y_START, SRC_Y_END = NY // 2 - 2, NY // 2 + 2
SRC_Z_START, SRC_Z_END = NZ // 2 - 2, NZ // 2 + 2


def main() -> None:
    # Initialize the 3D temperature grid
    T = np.full((NX, NY, NZ), T_INITIAL, dtype=np.float64)

    # Apply boundary conditions to the initial grid
    T[0, :, :] = T_BOUNDARY
    T[-1, :, :] = T_BOUNDARY
    T[:, 0, :] = T_BOUNDARY
    T[:, -1, :] = T_BOUNDARY
    T[:, :, 0] = T_BOUNDARY
    T[:, :, -1] = T_BOUNDARY

    print(f"Running simulation for {NT} time steps with dt = {DT:.6e} s")
    start_time = time.time()

    for n in range(NT):
        T_new = T.copy()

        # Vectorized 3D Laplacian on interior points
        laplacian = (
            (T[2:, 1:-1, 1:-1] - 2.0 * T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) / (DX**2)
            + (T[1:-1, 2:, 1:-1] - 2.0 * T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) / (DY**2)
            + (T[1:-1, 1:-1, 2:] - 2.0 * T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2]) / (DZ**2)
        )

        # Explicit update for interior
        T_new[1:-1, 1:-1, 1:-1] = T[1:-1, 1:-1, 1:-1] + ALPHA * DT * laplacian

        # Add the heat source term in the central region
        T_new[SRC_X_START:SRC_X_END, SRC_Y_START:SRC_Y_END, SRC_Z_START:SRC_Z_END] += Q_VALUE * DT

        # Enforce Dirichlet boundary conditions
        T_new[0, :, :] = T_BOUNDARY
        T_new[-1, :, :] = T_BOUNDARY
        T_new[:, 0, :] = T_BOUNDARY
        T_new[:, -1, :] = T_BOUNDARY
        T_new[:, :, 0] = T_BOUNDARY
        T_new[:, :, -1] = T_BOUNDARY

        T = T_new

        if NT > 0 and (n + 1) % max(1, NT // 10) == 0:
            print(f"Step {n + 1}/{NT} completed.")

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    # Save final 3D array
    np.save("fdm_ground_truth.npy", T)
    print("Ground truth data saved to fdm_ground_truth.npy")

    # Visualize center Z-slice
    plt.figure(figsize=(8, 6))
    center_slice_z = T[:, :, NZ // 2]
    plt.imshow(center_slice_z.T, origin='lower', extent=[0, LX, 0, LY], cmap='hot', aspect='auto')
    plt.colorbar(label='Temperature (°C)')
    plt.title('FDM Ground Truth: Center Z-Slice at T_final')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


