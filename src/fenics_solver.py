# src/fenics_solver.py
import meshio
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

def solve_heatsink():
    # --- 1. Load and Convert the Mesh ---
    # FEniCS needs the mesh in a specific XML format. We use meshio to convert it.
    msh = meshio.read("heatsink.msh")
    
    # Create separate mesh files for the volume and the surfaces
    meshio.write("mesh.xml", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]}))
    meshio.write("mf.xml", meshio.Mesh(points=msh.points, cells={"triangle": msh.cells["triangle"]},
                                      cell_data={"triangle": {"name_to_read": msh.cell_data["triangle"]["gmsh:physical"]}}))

    mesh = Mesh("mesh.xml")
    mf = MeshFunction("size_t", mesh, "mf.xml")

    # --- 2. Define the Physics Problem ---
    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary conditions
    # Bottom surface (Tag 1) has a fixed high temperature (our heat source)
    bc_bottom = DirichletBC(V, Constant(100.0), mf, 1) 
    bcs = [bc_bottom]

    # Define variational problem for conduction and convection
    T = TrialFunction(V)
    v = TestFunction(V)
    
    # Material and environmental properties
    k = Constant(200.0)  # Thermal conductivity of aluminum
    h = Constant(25.0)   # Convection coefficient
    T_air = Constant(25.0) # Ambient air temperature

    # FEniCS weak form of the PDE
    # This equation represents both conduction (left) and convection (right)
    a = dot(k*grad(T), grad(v))*dx + h*T*v*ds(2)
    L = h*T_air*v*ds(2)

    # --- 3. Solve for Temperature ---
    T_solution = Function(V)
    solve(a == L, T_solution, bcs)

    # --- 4. Export and Visualize the Results ---
    # Get coordinates and temperature values
    coords = V.tabulate_dof_coordinates()
    temps = T_solution.vector().get_local()
    
    # Save for our PINN
    np.save("data/fenics_ground_truth.npy", np.hstack([coords, temps.reshape(-1, 1)]))
    print("FEniCS ground truth saved to data/fenics_ground_truth.npy")

    # Visualize a slice
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=temps, cmap='hot')
    plt.colorbar(sc, label='Temperature (°C)')
    ax.set_title('FEniCS Ground Truth Temperature')
    plt.savefig('plots/fenics_visualization.png')
    print("Visualization saved to plots/fenics_visualization.png")

if __name__ == "__main__":
    solve_heatsink()
