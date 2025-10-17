# src/create_mesh.py
import gmsh
import sys

def generate_heatsink_mesh():
    gmsh.initialize()
    gmsh.model.add("heatsink")

    # --- Define Geometry Parameters ---
    base_dx, base_dy, base_dz = 0.5, 0.5, 0.1  # Base dimensions
    fin_dx, fin_dy, fin_dz = 0.1, 0.5, 0.4   # Fin dimensions
    mesh_size = 0.05  # Coarseness of the mesh

    # --- Create the Base Block ---
    base = gmsh.model.occ.addBox(0, 0, 0, base_dx, base_dy, base_dz)

    # --- Create the Fin on top of the base ---
    # Position the fin in the center of the base
    fin_x_start = (base_dx - fin_dx) / 2
    fin = gmsh.model.occ.addBox(fin_x_start, 0, base_dz, fin_dx, fin_dy, fin_dz)

    # --- Fuse the two parts into a single object ---
    # This is crucial for creating a single, continuous mesh
    fused_object = gmsh.model.occ.fuse([(3, base)], [(3, fin)])[0]
    gmsh.model.occ.synchronize()

    # --- Define Physical Groups for Boundary Conditions ---
    # This lets us identify surfaces later in FEniCS
    # Tag 1: Bottom surface (where we'll apply heat)
    # Tag 2: All other surfaces (for convective cooling)
    
    bottom_surface = []
    convection_surfaces = []
    
    surfaces = gmsh.model.occ.getEntities(dim=2)
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        # Identify bottom face by its Z-coordinate
        if abs(com[2] - 0.0) < 1e-6:
            bottom_surface.append(surface[1])
        else:
            convection_surfaces.append(surface[1])

    gmsh.model.addPhysicalGroup(2, bottom_surface, tag=1, name="bottom_surface")
    gmsh.model.addPhysicalGroup(2, convection_surfaces, tag=2, name="convection_surfaces")

    # --- Generate the 3D Mesh ---
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.mesh.generate(3)

    # --- Save the Mesh ---
    mesh_filename = "heatsink.msh"
    gmsh.write(mesh_filename)
    print(f"Mesh saved to {mesh_filename}")

    # Optional: Launch the GUI to view the mesh
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == "__main__":
    generate_heatsink_mesh()
