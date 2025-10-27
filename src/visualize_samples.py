import numpy as np
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_boundary_samples(mesh_file="heatsink.msh", samples_file="results/sampled_boundary_points.npz"):
    """
    Loads a mesh and sampled boundary points, then visualizes them in 3D.
    """
    print(f"Loading mesh from {mesh_file}...")
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file not found at '{mesh_file}'. Ensure it's in the project root.")
        return
        
    mesh = meshio.read(mesh_file)
    points = mesh.points  # Vertex coordinates
    
    # We only need triangle cells for surface visualization
    triangles = None
    if 'triangle' in mesh.cells_dict:
        triangles = mesh.cells_dict['triangle']
    else:
        print("Warning: No triangle cells found in the mesh for surface plotting.")

    print(f"Loading sampled points from {samples_file}...")
    if not os.path.exists(samples_file):
        print(f"Error: Sample points file not found at '{samples_file}'.")
        print("Please run train_phase2.py first to generate the data.")
        return
        
    data = np.load(samples_file)
    conv_x, conv_y, conv_z = data['conv_x'], data['conv_y'], data['conv_z']
    conv_normals = data['conv_normals']
    dirichlet_x, dirichlet_y, dirichlet_z = data['dirichlet_x'], data['dirichlet_y'], data['dirichlet_z']

    print("Generating 3D visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh surface (optional, can be slow for large meshes)
    if triangles is not None:
        print("Plotting mesh surface...")
        # Plotting the mesh surface itself can be resource intensive. 
        # We can plot the vertices instead for a quicker overview.
        # ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, 
        #                 color='grey', alpha=0.1, edgecolor='none')
        # Alternatively, plot just the vertices:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='lightgrey', alpha=0.2, label='Mesh Vertices')

    # Plot the sampled points
    print("Plotting sampled boundary points...")
    ax.scatter(conv_x, conv_y, conv_z, s=15, color='blue', label='Convective BC Points', alpha=0.8)
    ax.scatter(dirichlet_x, dirichlet_y, dirichlet_z, s=15, color='red', label='Dirichlet BC Points (Bottom)', alpha=0.8)

    # Optional: Plot normal vectors (can be very cluttered, use a subset)
    scale = 0.05  # Adjust scale factor for arrow length
    subset = np.random.choice(len(conv_x), min(100, len(conv_x)), replace=False)  # Plot only 100 normals
    ax.quiver(conv_x[subset], conv_y[subset], conv_z[subset], 
              conv_normals[subset, 0] * scale, 
              conv_normals[subset, 1] * scale, 
              conv_normals[subset, 2] * scale, 
              color='cyan', length=scale*1.5, normalize=True, label='Normals (subset)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of Sampled Boundary Points on Mesh')
    ax.legend()
    
    # Adjust view angle if needed
    ax.view_init(elev=20., azim=-60) 
    
    plt.savefig("plots/boundary_sample_visualization.png")
    print("Visualization saved to plots/boundary_sample_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize_boundary_samples()
