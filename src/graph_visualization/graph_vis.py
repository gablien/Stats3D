"""
This module contains the functions to plot a vascular graph with the pyvista library
"""

import os
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, f'/home/{os.getlogin()}/programs/ClearMap3')
import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.Analysis.Graphs.GraphRendering as gr


def plot_pyvista_plot(g, edge_colors, scale=np.array([1,1,1])):
    """
    Plot a graphtool graph using the pyvista library  

    Parameters
    ----------
    g: Graph
        The graph to plot
    edge_colors: np.array
        Array containing the color to give to each vertex, 
        should be the same length than the number of edges 
    scale: np.array
        x,y,z scale

    Returns
    -------
    A mesh plot
    """
    if g.n_edges != len(edge_colors):
        raise ValueError(f"graph has {g.n_edges} edges, but {len(edge_colors)} colors were provided.")
    interpolation = gr.interpolate_edge_geometry(g, smooth=5, order=2, points_per_pixel=0.2, verbose=False)

    coordinates, faces, colors = gr.mesh_tube_from_coordinates_and_radii(*interpolation,
                                        n_tube_points=5, edge_colors=edge_colors,
                                        processes=None, verbose=False)
    
    coordinates = np.array(coordinates)
    coordinates *= scale

    n_faces = faces.shape[0]
    pv_faces = np.insert(faces, 0, 3, axis=1).flatten()

    mesh = pv.PolyData(coordinates, pv_faces, n_faces=n_faces)
    mesh.point_data['colors'] = colors

    return mesh.plot(smooth_shading=True, scalars='colors', rgba=True, return_viewer=True)
    

def vertex_filter_to_edge_filter(graph, vertex_filter, operator=np.logical_and):
    """
    Converts a vertex filter to an edge filter

    Parameters
    ----------
    graph : Graph
        The graph to convert the filter for
    vertex_filter : np.array
        The vertex filter to convert

    Returns
    -------
    The edge filter (as a numpy array)
    """
    connectivity = graph.edge_connectivity()
    start_vertex_follows_filter = vertex_filter[connectivity[:, 0]]
    end_vertex_follows_filter = vertex_filter[connectivity[:, 1]]
    edge_filter = operator(start_vertex_follows_filter, end_vertex_follows_filter)
    return edge_filter