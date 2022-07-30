import pyvista
# using pythreejs Jupyter Backend library

# set the global theme to use pythreejs
pyvista.global_theme.jupyter_backend = 'pythreejs'

pl = pyvista.Plotter()

# load mesh
filename = "data/Glass.obj"
filename.split("/")[-1]  # omit the path
reader = pyvista.get_reader(filename)
mesh = reader.read()

# mesh.plot(cpos='xy', show_scalar_bar=False)
# source: https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html
pl.add_mesh(mesh, style='surface', lighting=True, opacity=0.001, smooth_shading = True, ambient=0, diffuse=1, specular=5, use_transparency=True, roughness=0.0)
# roughness = 0 for glossy
# 
pl.show()


"""
# lower left, using physically based rendering
pl.add_mesh(pyvista.Sphere(center=(-1, 0, -1)),
            show_edges=False, pbr=True, color='white', roughness=0.2,
            metallic=0.5)

# upper right, matches default pyvista plotting
pl.add_mesh(pyvista.Sphere(center=(1, 0, 1)))

# Upper left, mesh displayed as points
pl.add_mesh(pyvista.Sphere(center=(-1, 0, 1)),
            color='k', style='points', point_size=10)

# mesh in lower right with flat shading
pl.add_mesh(pyvista.Sphere(center=(1, 0, -1)), lighting=False,
            show_edges=True)

# show mesh in the center with a red wireframe
pl.add_mesh(pyvista.Sphere(), lighting=True, show_edges=False,
            color='red', line_width=0.5, style='wireframe',
            opacity=0.99)

pl.camera_position = 'xz'
pl.show()
"""