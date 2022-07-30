import pyvista as pv
from pyvista import _vtk
import numpy as np
from collections import defaultdict


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
    for i in range(1,10,1):
        move()
        rotate()
        p.update() #update the plot

def move():
    for s in ss:
        if s.n_points > 0:
            s = s.translate((np.random.random(),np.random.random(),np.random.random()) , inplace=True)
    p.update()

def rotate():
    for s in ss:
        if s.n_points > 0:
            s = s.rotate_vector((np.random.random(), np.random.random(), np.random.random()), np.random.randint(low=0, high=90, size=None, dtype=int), inplace=True)
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

    points = np.random.random([20, 3])
    test_point_cloud = pv.PolyData(points)
    
    ss = split_voronoi(test_mesh, test_point_cloud)
    p = pv.Plotter()
    
    for s in ss:
        if s.n_points > 0:
            c = (random.random(), random.random(), random.random())
            p.add_mesh(s, style='surface', lighting=True, opacity=0.1, smooth_shading = True, ambient=0, diffuse=1, specular=5, specular_power=128, use_transparency=True, metallic=1, roughness=0.0)
    # More functions

    p.add_background_image("data/snow.jpg")


    # ADD Camera Orientation Widget - https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_camera_orientation_widget.html
    p.add_camera_orientation_widget()

    # ADD TEXT
    actor = p.add_text('Glass Cup', position='lower_left', color='blue',
                        shadow=True, font_size=26)

    # enable anti aliasing
    p.enable_anti_aliasing()

    # use depth_peeling for better translucent material - https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.enable_depth_peeling.html ??? - test

    # key press events
    p.add_key_event("a", move)

    p.add_key_event("b", multimove)

    p.add_key_event("c", rotate)

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
