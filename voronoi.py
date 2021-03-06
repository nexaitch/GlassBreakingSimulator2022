import logging

import pyvista as pv
from pyvista import _vtk
import numpy as np
from collections import defaultdict


def clip_multiple_planes(mesh, planes, tolerance=1e-6, inplace=False):
    """
    Very hackish way to clip with multiple planes inplace.
    :param mesh The mesh to clip
    :param planes a list of planes, specified as a pair of (normal, origin) vectors
    """
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
    """Converts a lines array to adjacency list."""
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
        # section.clean(inplace=True)
        if section.n_open_edges > 0:
            logging.warning(f"{section.n_open_edges} open edges detected")
        if section.n_points > 0:
            sections.append(section)
            # print(section.volume)
    return sections


if __name__ == "__main__":
    import random
    # test_mesh = pv.Sphere(2)
    test_mesh = pv.get_reader("data/GlassCup.stl").read()
    logging.info(f"{test_mesh.n_open_edges} open edges found")
    test_mesh.translate(-test_mesh.center_of_mass(), inplace=True)
    points = (np.random.random([40, 3])-.5)*1.5 + 1
    test_point_cloud = pv.PolyData(points)
    ss = split_voronoi(test_mesh, test_point_cloud)
    p = pv.Plotter()
    for s in ss:
        if s.n_points > 0:
            c = (random.random(), random.random(), random.random())
            p.add_mesh(s, color=c, opacity=0.5)
    p.add_points(test_point_cloud.points, opacity=1)
    p.show()

