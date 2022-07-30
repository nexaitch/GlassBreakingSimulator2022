import pyvista
# using pythreejs Jupyter Backend library

# set the global theme to use pythreejs
pyvista.global_theme.jupyter_backend = 'pythreejs'

pl = pyvista.Plotter()

"""
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
# Try to add background image - source: https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_background_image.html
# BACKGROUND IMAGE
filename = "data/Glass.obj"
filename.split("/")[-1]  # omit the path
reader = pyvista.get_reader(filename)
mesh = reader.read()
# enable anti aliasing
pl.enable_anti_aliasing()

# use depth_peeling for better translucent material - https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.enable_depth_peeling.html
#pl.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0)

actor = pl.add_mesh(mesh, style='surface', lighting=True, opacity=0.001, smooth_shading = True, ambient=0, diffuse=1, specular=5, use_transparency=True, roughness=0.0)
pl.add_background_image("data/snow.jpg")


# ADD Camera Orientation Widget - https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_camera_orientation_widget.html
pl.add_camera_orientation_widget()

# ADD TEXT
actor = pl.add_text('Glass Cup', position='lower_left', color='blue',
                    shadow=True, font_size=26)
pl.show()

"""
# TEST SHADOW
filename = "data/Glass.obj"
filename.split("/")[-1]  # omit the path
reader = pyvista.get_reader(filename)
mesh = reader.read()
pl = pyvista.Plotter(lighting='none', window_size=(1000, 1000))
light = pyvista.Light()
light.set_direction_angle(20, -20)
pl.add_light(light)
_ = pl.add_mesh(mesh, style='surface', lighting=True, opacity=1, smooth_shading = True, ambient=0, diffuse=1, specular=5, use_transparency=False, roughness=0.0)
pl.enable_shadows() #the results is not that great?
pl.show()
"""

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