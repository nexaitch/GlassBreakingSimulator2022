import pyvista as pv
from pyvista import _vtk
import numpy as np
from collections import defaultdict

import time

# Playing Sounds !!! make sure to pip install pygame and copy the libmpg123.dll from pygame folder to Windows/System32
##################
import pygame
from pygame import mixer

#Instantiate mixer
mixer.init()

#Load audio file
mixer.music.load('danceofpales.mp3') # BGM
#Set preferred volume
mixer.music.set_volume(0.2)
mixer.music.play(loops=-1) #set loops to -1 to loop indefinitely, start at 0.0

def clip_multiple_planes(mesh, planes, tolerance=1e-6, inplace=False):
    """Very hackish way to clip with multiple planes inplace"""
    # specify planes as pairs of (normal, origin)
    
    if mesh.n_open_edges > 0:
        raise ValueError("This surface appears to be non-manifold.")

    # create the plane for clipping
    collection = _vtk.vtkPlaneCollection()
    for normal, origin in planes:
        plane = pv.generate_plane(normal, origin)
        collection.AddItem(plane)

    alg = _vtk.vtkClipClosedSurface()
    alg.SetGenerateFaces(True)
    alg.SetInputDataObject(mesh)
    alg.SetTolerance(tolerance)
    alg.SetClippingPlanes(collection)
    pv.core.filters._update_alg(alg, False, 'Clipping Closed Surface')
    result = pv.core.filters._get_output(alg)

    if inplace:
        mesh.overwrite(result)
        return mesh
    else:
        return result


def lines_to_adjacency_list(lines: np.ndarray):
    adjacency_list = defaultdict(lambda: [])
    i = 0
    line_len = len(lines)
    while i < line_len:
        n = lines[i]
        spline = lines[i + 1: i + 1 + n]
        for u, v in zip(spline, spline[1:]):
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)
        i += n + 1
    return adjacency_list


def split_voronoi(mesh: pv.PolyData, point_cloud: pv.PolyData):
    """Splits a mesh based on the voronoi tiling of the input point cloud."""
    tetrahedralisation = point_cloud.delaunay_3d()
    edges = tetrahedralisation.extract_all_edges()
    adj_list = lines_to_adjacency_list(edges.lines)
    sections = []
    pts = edges.points
    for i, pt in enumerate(pts):
        planes = []
        for j in adj_list[i]:
            pt2 = pts[j]
            mid = (pt + pt2) / 2
            norm = pt - pt2
            planes.append((norm, mid))
        # p.add_arrows(np.array([p[1] for p in planes]), np.array([p[0] for p in planes]))
        section = clip_multiple_planes(mesh, planes, inplace=False)
        if section.n_points > 0:
            sections.append(section)
    return sections

# Test Callback Functions
def multimove():
    #Play the glass breaking sound effect
    mixer.Channel(0).play(mixer.Sound("break.mp3"), maxtime=1200) # use channels to play SFX on top of background music, use maxtime to stop after certain miliseconds
    time.sleep(0.4) #delay 0.4 seconds to make sound match the (current) breaking animation
    # https://www.pygame.org/docs/ref/music.html
    for i in range(1,100,1): # How long the glass breaking animation lasts
        move()
        #rotate()
        p.update() #update the plot

def move():
    special_point = np.array((0,0,0))
    for s in ss:
        if s.n_points > 0:
            s_vector = np.array(s.center) # center of the bounding box
            d_vector = s_vector - special_point
            # s = s.translate((np.random.random(),np.random.random(),np.random.random()) , inplace=True)
            s = s.translate((d_vector[0]/10,d_vector[1]/10,d_vector[2]/10) , inplace=True) #vector components to move the glass, Divide by a larger number to make it slower, multiply to make faster
    p.update()

def rotate():
    for s in ss:
        if s.n_points > 0:
            s = s.rotate_vector((np.random.random(), np.random.random(), np.random.random()), np.random.randint(low=0, high=90, size=None, dtype=int), inplace=True)
            # inplace = True to update mesh
    p.update()

# ADD function to Regenerate/respawn Glass model after it has been destroyed - Buggy, when regenerating the texture is completely white also does not break anymore
def regen_glass():
    p.clear() # clear plot and remove all actors and properties    
    ss = split_voronoi(test_mesh, test_point_cloud)
    p.add_points(test_point_cloud)

    # PolyData, how to get center of the bounding box!
    print("Center of the very first fragment:", ss[0].center)
    
    for s in ss:
        if s.n_points > 0:
            c = (random.random(), random.random(), random.random())
            p.add_mesh(s, style='surface', lighting=True, opacity=0.1, smooth_shading = True, ambient=0, diffuse=1, specular=5, specular_power=128, use_transparency=True, metallic=1, roughness=0.0)

    p.update()

# Added function to play background music BGM

# IDEAS: Add function to SWITCH what kind of object/3D MODEL to BREAK !!!
# The idea is like a left/right button to scroll through list of objects
# May need a list of all objects to load and object models as well
def switch_object_left():

    return True

def switch_object_right():

    return True

# Add function to load


def generate_points(size: int, origin: np.ndarray = None, spread: float = 1, df: int = 3):
    # https://mathworld.wolfram.com/SpherePointPicking.html

    if origin is None:
        origin = np.zeros(3)

    assert origin.shape[0] == 3 and len(origin.shape) == 1

    origins = np.resize(origin, (size, 3))

    thetas = 2 * np.pi * np.random.rand(size)
    phis = np.arccos(2*np.random.rand(size) - 1)

    # we divide by df, so we can change it without changing the mean distance
    distances = np.random.chisquare(df, size) * spread / df

    xs = distances * np.cos(thetas) * np.sin(phis)
    ys = distances * np.sin(thetas) * np.sin(phis)
    zs = distances * np.cos(phis)
    return np.vstack((xs, ys, zs)).transpose() + origins

# Splitting/Cracking Callback function, crack the glass or object more and more upon button click - Need to Edit n fix
def more_cracks():
    for s in ss:
        if s.n_points > 0:
            p = p
            # inplace = True to update mesh
    p.update()


if __name__ == "__main__":
    import random
    """
    test_mesh = pv.Sphere(2)
    points = np.random.random([20, 3])
    test_point_cloud = pv.PolyData(points)
    ss = split_voronoi(test_mesh, test_point_cloud)
    p = pv.Plotter()
    for s in ss:
        if s.n_points > 0:
            c = (random.random(), random.random(), random.random())
            p.add_mesh(s, color=c, opacity=0.5)
    p.show()
    """
    # With Glass Model
    filename = "data/GlassCup.stl"
    filename.split("/")[-1]  # omit the path
    reader = pv.get_reader(filename)
    test_mesh = reader.read()

    points = generate_points(50, df=20)
    test_point_cloud = pv.PolyData(points)
    
    ss = split_voronoi(test_mesh, test_point_cloud)
    p = pv.Plotter()
    p.add_points(test_point_cloud)

    # PolyData, how to get center of the bounding box!
    print("Center of the very first fragment:", ss[0].center)

    # TESTING PBR (Physically Based Rendering)
    from pyvista import examples
    # Download skybox
    # cubemap = examples.download_sky_box_cube_map()
    # Space cubemap
    # cubemap = examples.download_cubemap_space_4k()

    # May be useful to convert panorama image to a cubemap - https://jaxry.github.io/panorama-to-cubemap/

    # Creating a cubemap - https://docs.pyvista.org/api/utilities/_autosummary/pyvista.cubemap_from_filenames.html?highlight=cube+map#pyvista.cubemap_from_filenames - TEST
    # specify 6 images as the cube faces
    """image_paths = [
    'ballroom.jpg',
    'ballroom.jpg',
    'ballroom.jpg',
    'ballroom.jpg',
    'ballroom.jpg',
    'ballroom.jpg',
    ]"""
    # cubemap = pv.cubemap_from_filenames(image_paths=image_paths)  

    # https://docs.pyvista.org/api/utilities/_autosummary/pyvista.cubemap.html
    # p.add_actor(cubemap.to_skybox())
    cubemap = pv.cubemap('cubemap_test', 'spacehigh', '.jpg') #create a cubemap from the 6 images, arguments are the Directory folder, followed by the prefix for the images, followed by the image extension
    p.add_actor(cubemap.to_skybox()) # convert cubemap to skybox  
    p.set_environment_texture(cubemap)  # For reflecting the environment off the mesh
    
    for s in ss:
        if s.n_points > 0:
            c = (random.random(), random.random(), random.random())
            # Try using PBR mode for more realism, PBR only works with PolyData
            p.add_mesh(s, color='white', pbr=True, metallic=1, roughness=0.1, diffuse=1, opacity=0.1, smooth_shading = True, use_transparency=True, specular=5)
    # More functions

    # p.add_background_image("ballroom.jpg")


    # ADD Camera Orientation Widget - https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_camera_orientation_widget.html
    # p.add_camera_orientation_widget()

    # ADD TEXT
    actor = p.add_text('Glass Cup', position='lower_left', color='blue',
                        shadow=True, font_size=26)

    # enable anti aliasing
    p.enable_anti_aliasing()

    # use depth_peeling for better translucent material - https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.enable_depth_peeling.html ??? - test

    # key press events - CASE SENSTITIVE
    p.add_key_event("a", move)

    p.add_key_event("b", multimove)

    p.add_key_event("c", rotate)

    p.add_key_event("d", regen_glass)


    # p.enable_eye_dome_lighting()
    p.show()

    """
    # Splitting volumes TEST - extract each split volume
    # Source: https://docs.pyvista.org/examples/01-filter/compute-volume.html#split-vol-ref
    for i, body in enumerate(ss):
        pl = pv.Plotter()
        pl.add_mesh(body, style='surface', lighting=True, opacity=0.1, smooth_shading = True, ambient=0, diffuse=1, specular=5, specular_power=128, use_transparency=True, metallic=1, roughness=0.0)
        print(f"Body {i} volume: {body.volume:.3f}")
        pl.show()
        pl.clear()
    """

    """
    # Translate each separate volume
    p2 = pv.Plotter()
    for s in ss:
        if s.n_points > 0:
            trans = s.translate((np.random.random(),np.random.random(),np.random.random()) , inplace=True)
            p2.add_mesh(trans, style='surface', lighting=True, opacity=0.1, smooth_shading = True, ambient=0, diffuse=1, specular=5, specular_power=128, use_transparency=True, metallic=1, roughness=0.0)
    p2.show()
    """
