# src/create_mock_heatsink_data.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_mock_heatsink_data():
    """
    Create a realistic mock heat sink temperature distribution
    that mimics what FEniCS would produce for our geometry.
    """
    
    # Define the heat sink geometry (same as in create_mesh.py)
    base_dx, base_dy, base_dz = 0.5, 0.5, 0.1  # Base dimensions
    fin_dx, fin_dy, fin_dz = 0.1, 0.5, 0.4   # Fin dimensions
    
    # Create a grid of points within the heat sink volume
    # Base region: 0 <= x <= 0.5, 0 <= y <= 0.5, 0 <= z <= 0.1
    # Fin region: 0.2 <= x <= 0.3, 0 <= y <= 0.5, 0.1 <= z <= 0.5
    
    # Generate points in base
    x_base = np.linspace(0, base_dx, 20)
    y_base = np.linspace(0, base_dy, 20)
    z_base = np.linspace(0, base_dz, 5)
    
    # Generate points in fin
    x_fin = np.linspace(0.2, 0.3, 8)
    y_fin = np.linspace(0, base_dy, 20)
    z_fin = np.linspace(base_dz, base_dz + fin_dz, 15)
    
    # Create coordinate arrays
    coords_base = []
    for x in x_base:
        for y in y_base:
            for z in z_base:
                coords_base.append([x, y, z])
    
    coords_fin = []
    for x in x_fin:
        for y in y_fin:
            for z in z_fin:
                coords_fin.append([x, y, z])
    
    coords = np.array(coords_base + coords_fin)
    
    # Calculate realistic temperature distribution
    # Bottom surface (z=0) is hot (100°C), top surfaces are cooler due to convection
    # Temperature decreases with distance from heat source and height
    
    temperatures = []
    for coord in coords:
        x, y, z = coord
        
        # Base temperature calculation
        if z <= base_dz:  # In base region
            # Temperature decreases with distance from center
            center_dist = np.sqrt((x - base_dx/2)**2 + (y - base_dy/2)**2)
            base_temp = 100.0 - 20.0 * center_dist / (base_dx/2)  # 100°C at center, ~80°C at edges
            # Add some cooling with height
            height_factor = 1.0 - 0.3 * (z / base_dz)
            temp = base_temp * height_factor
            
        else:  # In fin region
            # Temperature decreases with height due to convection
            height_above_base = z - base_dz
            max_height = fin_dz
            
            # Base temperature at fin bottom
            fin_base_temp = 85.0  # Slightly cooler than base center
            
            # Convective cooling with height
            convective_cooling = 40.0 * (height_above_base / max_height)
            temp = fin_base_temp - convective_cooling
            
            # Add some lateral cooling
            lateral_dist = min(abs(x - 0.25), abs(x - 0.25))  # Distance from fin center
            lateral_cooling = 10.0 * lateral_dist / (fin_dx/2)
            temp -= lateral_cooling
        
        # Ensure minimum temperature
        temp = max(temp, 25.0)  # Ambient temperature
        temperatures.append(temp)
    
    temperatures = np.array(temperatures)
    
    # Save the data
    data = np.hstack([coords, temperatures.reshape(-1, 1)])
    np.save("data/fenics_ground_truth.npy", data)
    print(f"Mock heat sink data saved to data/fenics_ground_truth.npy")
    print(f"Shape: {data.shape}")
    print(f"Temperature range: {temperatures.min():.1f}°C to {temperatures.max():.1f}°C")
    
    # Create visualization
    fig = plt.figure(figsize=(12, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(coords[:,0], coords[:,1], coords[:,2], c=temperatures, cmap='hot', s=20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Mock Heat Sink Temperature Distribution')
    plt.colorbar(sc, ax=ax1, label='Temperature (°C)', shrink=0.8)
    
    # 2D slice through center (y=0.25)
    ax2 = fig.add_subplot(122)
    center_y = 0.25
    center_mask = np.abs(coords[:,1] - center_y) < 0.05
    center_coords = coords[center_mask]
    center_temps = temperatures[center_mask]
    
    scatter = ax2.scatter(center_coords[:,0], center_coords[:,2], c=center_temps, cmap='hot', s=30)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title(f'Temperature Slice (Y={center_y})')
    plt.colorbar(scatter, ax=ax2, label='Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig('plots/fenics_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to plots/fenics_visualization.png")
    
    return coords, temperatures

if __name__ == "__main__":
    create_mock_heatsink_data()
