import time
from collections import defaultdict
from typing import Callable, Dict, Any

import numpy as np
import pyvista as pv
from pygame import mixer
from pyvista import _vtk


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


def make_physics_function(
        origin: np.ndarray = np.array((0, 0, 0)),
        gravity: np.ndarray = np.array((0, 0, 0)),
        forward_impact: np.ndarray = np.array((0, -1, 0)),
        explodiness: float = 2,
        spin_amount: float = 8,
        damping: float = 0.01,
        angular_damping: float = 0.01,
        damping_velocity: float = 2,
        dt: float | Callable[[int], float] = 0.001,
        randomness: float = 0,
) -> Callable:
    rs = [section.center_of_mass() - origin for section in section_meshes]

    fl = np.linalg.norm(forward_impact)
    # forward direction
    fd = np.zeros(3) if fl < 1e-6 else forward_impact / fl

    r_perpendicular_components = [r - np.dot(r, fd) * fd for r in rs]
    # inverse square law
    velocities = [(explodiness * r * np.power(np.dot(r, r), -3 / 2)
                  + forward_impact * np.power(np.dot(r_perpendicular, r_perpendicular), -1 / 2)
                  + np.random.randn(3) * randomness) / section.volume
                  for r, r_perpendicular, section in zip(rs, r_perpendicular_components, section_meshes)]
    # kinda hackish way to give everything a random rotation but oh well
    angular_axes = [np.random.random(3) - 0.5 for _ in section_meshes]
    angular_velocities = [spin_amount / section.volume for section in section_meshes]
    t = 0

    def do_the_thing():
        nonlocal t
        if isinstance(dt, float):
            delta_time = dt
        else:
            t += 1
            delta_time = dt(t)

        for i, (v, section, axis, omega) in enumerate(
                zip(velocities, section_meshes, angular_axes, angular_velocities)):
            section.translate(v * delta_time, inplace=True)
            section.rotate_vector(axis, omega * delta_time, point=section.center_of_mass(), inplace=True)
            v += gravity * delta_time
            # simulate some kind of turbulence idk
            if np.linalg.norm(v) > damping_velocity:
                v *= (1 - damping * delta_time)
            angular_velocities[i] *= (1 - angular_damping - delta_time)

        p.update()

    return do_the_thing


def setup_scene():
    # With Glass Model
    filename = "data/GlassCup.stl"
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
    # Download skybox
    # cubemap = examples.download_sky_box_cube_map()
    # Space cubemap
    # cubemap = examples.download_cubemap_space_4k()

    # May be useful to convert panorama image to a cubemap - https://jaxry.github.io/panorama-to-cubemap/

    # Creating a cubemap
    # https://docs.pyvista.org/api/utilities/_autosummary/pyvista.cubemap_from_filenames.html?highlight=cube+map#pyvista.cubemap_from_filenames
    #  - TEST
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
    cubemap = pv.cubemap('cubemap_test', 'spacehigh', '.jpg')
    # create a cubemap from the 6 images, arguments are the Directory folder, followed by the prefix for the images, followed by the image extension
    p.add_actor(cubemap.to_skybox())  # convert cubemap to skybox
    p.set_environment_texture(cubemap)  # For reflecting the environment off the mesh
    # More functions

    # p.add_background_image("ballroom.jpg")

    # ADD Camera Orientation Widget
    # https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_camera_orientation_widget.html
    # p.add_camera_orientation_widget()


# ADD function to Regenerate/respawn Glass model after it has been destroyed - Buggy, when regenerating the texture is completely white also does not break anymore
# Added function to play background music BGM

# IDEAS: Add function to SWITCH what kind of object/3D MODEL to BREAK !!!
# The idea is like a left/right button to scroll through list of objects
# May need a list of all objects to load and object models as well
def switch_object_right(): # change model to NEXT in the list
    global model_index , list_of_models, test_mesh, text_actor #global variable for model index and list of all models, and text actor
    model_index = (model_index + 1) % 11
    filename = list_of_models[model_index]
    reader = pv.get_reader(filename)
    test_mesh = reader.read()
    p.remove_actor(text_actor) #remove text actor
    text_actor = p.add_text(list_of_names[model_index], position='lower_left', color='blue',
                       shadow=True, font_size=26) # update the text based on the model index and add it as an actor
    reset()


def switch_object_left(): # change model to PREV in the list
    global model_index , list_of_models, test_mesh, text_actor #global variable for model index and list of all models, and text actor
    model_index = (model_index - 1) % 11
    filename = list_of_models[model_index]
    reader = pv.get_reader(filename)
    test_mesh = reader.read()
    p.remove_actor(text_actor)
    text_actor = p.add_text(list_of_names[model_index], position='lower_left', color='blue',
                       shadow=True, font_size=26)
    reset()

# Add function to load

def reset():
    global main_mesh_actor, section_actors
    p.remove_actor(main_mesh_actor)
    for sa in section_actors:
        p.remove_actor(sa)
    main_mesh_actor = p.add_mesh(test_mesh, **glass_texture)
    section_actors = []
    p.update()


def generate_points(size: int, origin: np.ndarray = None, spread: float = 1, df: int = 3):
    # https://mathworld.wolfram.com/SpherePointPicking.html

    if origin is None:
        origin = np.zeros(3)

    assert origin.shape[0] == 3 and len(origin.shape) == 1

    origins = np.resize(origin, (size, 3))

    thetas = 2 * np.pi * np.random.rand(size)
    phis = np.arccos(2 * np.random.rand(size) - 1)

    # we divide by df, so we can change it without changing the mean distance
    distances = np.random.chisquare(df, size) * spread / df

    xs = distances * np.cos(thetas) * np.sin(phis)
    ys = distances * np.sin(thetas) * np.sin(phis)
    zs = distances * np.cos(phis)
    return np.vstack((xs, ys, zs)).transpose() + origins


def smooth_ramp(initial, final, timescale=100, midpoint=100):
    def do_the_thing(t):
        t -= midpoint
        t /= timescale
        frac = 1/(1 + np.exp(-t))
        return initial * (1 - frac) + final * frac

    return do_the_thing


def play_music():
    # Playing Sounds !!!
    # make sure to pip install pygame and copy the libmpg123.dll from pygame folder to Windows/System32
    ##################

    # Instantiate mixer
    mixer.init()

    # Load audio file
    mixer.music.load('danceofpales.mp3')  # BGM
    # Set preferred volume
    mixer.music.set_volume(0.01)
    mixer.music.play(loops=-1)  # set loops to -1 to loop indefinitely, start at 0.0


def explode(point=np.array((0, 0, 0))):
    global main_mesh_actor, section_actors, section_meshes

    if main_mesh_actor is None or len(section_actors) > 0:
        print("can't explode")
        return

    mixer.Channel(0).play(mixer.Sound("break.mp3"), maxtime=1200)
    time.sleep(0.2)
    points = generate_points(params["fragment count"], origin=point, df=params["df"], spread=params["scale"])
    point_cloud = pv.PolyData(points)

    section_meshes = split_voronoi(test_mesh, point_cloud)

    section_actors = []

    for s in section_meshes:
        # Try using PBR mode for more realism, PBR only works with PolyData
        p.add_mesh(s, **glass_texture)

    p.remove_actor(main_mesh_actor)
    p.update()

    for s in section_meshes:
        # Try using PBR mode for more realism, PBR only works with PolyData
        ac = p.add_mesh(s, **glass_texture)
        section_actors.append(ac)

    instant = 2
    smooth = 0.2
    do_physics = make_physics_function(
        explodiness=params["explodiness"],
        forward_impact=np.array((0, -1, 0)) * params["impact"],
        randomness=params["randomness"],
        dt=smooth_ramp(1/60/params["slow motion"], 1/60, smooth * params["slow motion"], instant * params["slow motion"])
    )

    # start_time = time.perf_counter()
    iterations = 600
    for i in range(iterations):  # How long the glass breaking animation lasts
        do_physics()
    # end_time = time.perf_counter()
    # print(f"FPS: {iterations/(end_time-start_time)}")


def update_property(param_name: str, params_dict: Dict[str, Any], preprocessing=lambda x: x):
    def do_the_thing(value):
        params_dict[param_name] = preprocessing(value)
    return do_the_thing


glass_texture = dict(color='white', pbr=True, metallic=0.8, roughness=0.1, diffuse=1, opacity=0.1,
                     smooth_shading=True,
                     use_transparency=True)

if __name__ == "__main__":
    model_index = 0 #to determine which model is currently selected, there are a total of 11 models
    list_of_models = ["data/Ashtray.stl", "data/BottleCap.stl", "data/BottleVer2.stl", "data/GlassCup.stl", "data/JewelV2.stl", "data/Orb.stl", "data/GlassPane.stl", "data/Plate.stl", "data/PotionHigh.stl", "data/Prism.stl", "data/WineLowPoly.stl"] #list of all available 3D models
    list_of_names = ["Ashtray","Bottle Cap", "Bottle", "Cup", "Jewel", "Orb", "Pane", "Plate", "Potion", "Prism", "Wine"] # list of texts corresponding to object models
    main_mesh_actor = None
    section_actors = []
    section_meshes = []
    p = pv.Plotter()

    import random

    play_music()

    # With Glass Model
    filename = "data/Ashtray.stl"
    reader = pv.get_reader(filename)
    test_mesh = reader.read()

    main_mesh_actor = p.add_mesh(test_mesh, **glass_texture)

    # TESTING PBR (Physically Based Rendering)
    # Download skybox
    # cubemap = examples.download_sky_box_cube_map()
    # Space cubemap
    # cubemap = examples.download_cubemap_space_4k()

    # May be useful to convert panorama image to a cubemap - https://jaxry.github.io/panorama-to-cubemap/

    # Creating a cubemap
    # https://docs.pyvista.org/api/utilities/_autosummary/pyvista.cubemap_from_filenames.html?highlight=cube+map#pyvista.cubemap_from_filenames
    #  - TEST
    # specify 6 images as the cube faces
    # cubemap = pv.cubemap_from_filenames(image_paths=image_paths)  

    # https://docs.pyvista.org/api/utilities/_autosummary/pyvista.cubemap.html
    # p.add_actor(cubemap.to_skybox())
    cubemap = pv.cubemap('cubemap_test', 'spacehigh', '.jpg')
    # create a cubemap from the 6 images, arguments are the Directory folder, followed by the prefix for the images, followed by the image extension
    p.add_actor(cubemap.to_skybox())  # convert cubemap to skybox
    p.set_environment_texture(cubemap)  # For reflecting the environment off the mesh
    # More functions

    # p.add_background_image("ballroom.jpg")

    # ADD Camera Orientation Widget
    # https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_camera_orientation_widget.html
    # p.add_camera_orientation_widget()

    # ADD TEXT actor
    text_actor = p.add_text('Ashtray', position='lower_left', color='blue',
                       shadow=True, font_size=26)

    # enable anti aliasing
    p.enable_anti_aliasing()

    # use depth_peeling for better translucent material
    # https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.enable_depth_peeling.html
    # ??? - test

    p.enable_surface_picking(callback=explode, left_clicking=True, show_message="Click to explode, R to reset")

    params = {
        "explodiness": 0.5,
        "df": 3,
        "scale": 1,
        "fragment count": 50,
        "impact": 1,
        "randomness": 0,
        "slow motion": 30,
    }

    p.add_slider_widget(update_property("explodiness", params), (0, 3), params["explodiness"], "explodiness",
                        pointa=(0.6, 0.9), pointb=(0.9, 0.9), style="modern")
    p.add_slider_widget(update_property("df", params, int), (2, 20), params["df"], "df", fmt="%.0f",
                        pointa=(0.6, 0.8), pointb=(0.9, 0.8), style="modern")
    p.add_slider_widget(update_property("scale", params), (0.5, 5), params["scale"], "scale",
                        pointa=(0.6, 0.7), pointb=(0.9, 0.7), style="modern")
    p.add_slider_widget(update_property("fragment count", params, int), (10, 100), params["fragment count"], "fragment count", fmt="%.0f",
                        pointa=(0.6, 0.6), pointb=(0.9, 0.6), style="modern")
    p.add_slider_widget(update_property("impact", params), (0, 2), params["impact"], "impact",
                        pointa=(0.6, 0.5), pointb=(0.9, 0.5), style="modern")
    p.add_slider_widget(update_property("randomness", params, lambda x: x*x), (0, 1), params["randomness"], "randomness",
                        pointa=(0.6, 0.4), pointb=(0.9, 0.4), style="modern")
    p.add_slider_widget(update_property("slow motion", params), (1, 100), params["slow motion"], "slow motion",
                        pointa=(0.6, 0.3), pointb=(0.9, 0.3), style="modern")

    p.add_key_event("r", reset)

    p.show_axes()

    # Switch object to next object in list
    p.add_key_event("x", switch_object_right)
    p.add_key_event("X", switch_object_right)
    # Switch object to prev object in list
    p.add_key_event("z", switch_object_left)
    p.add_key_event("Z", switch_object_left)

    # p.enable_eye_dome_lighting()
    p.show()
