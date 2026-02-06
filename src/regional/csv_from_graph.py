"""
This module contains the functions to generate a csv from the vascular graph with relevant info on vascular structure 
"""

import os
import sys

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from natsort import natsorted
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mode

try:
    import ClearMap.Settings as settings
except:
    sys.path.append(f"/home/{os.getlogin()}/programs/ClearMap3")
    import ClearMap.Settings as settings
import ClearMap.Analysis.graphs.graph_gt as ggt
from ClearMap.Analysis.vasculature.vasc_graph_utils import vertex_filter_to_edge_filter
from ClearMap.Analysis.vasculature.vasc_graph_utils import vertex_to_edge_property
from ClearMap.Analysis.vasculature.vasc_graph_utils import parallel_get_vessels_lengths


MIN_CHUNK_SIZE = 5
MAX_PROCS = 5
MIN_DISTANCE_TO_SURFACE = 2 # minimal distance to surface for self loops
MIN_LENGTH = 5 # minimal length for self loops


def compute_csv_from_graph(graph_paths, stats_file_path, df_sample, add_tortuosity = False, only_arteries=False):
    """
    Parameters
    ----------
    graph_paths: dictionnary
        keys = sample_ids, values = paths to graphs
    stats_file_path: posixpath object
        path of the csv file containing the results.
        If it does not exist, will be created.
    df_sample: posixpath object
        path to the csv matching sample id and experimental group
    add_tortuosity: boolean:
        If True, the tortuosity will be computed
    """

    for sample_id, graph_path in graph_paths.items():
        sample_results = []
        print(f"Computing graph stats from sample {sample_id} : {graph_path}")
        print(f"Loading graph from sample {sample_id}")
        initial_graph = ggt.load(str(graph_path))
        print("Graph loaded!")

        if only_arteries:
            artery_filter = initial_graph.edge_property('artery')==True
            graph = initial_graph.sub_graph(edge_filter=artery_filter)
        else: 
            graph = initial_graph

        # Compute edge annotation
        print("Adding edges annotation")
        annotation_data = np.array(graph.edge_geometry_property("annotation"))
        indices = np.array(graph.edge_property("edge_geometry_indices"))
        annot_from_eg = np.empty(len(indices), dtype=annotation_data.dtype)
        for i, (start, end) in enumerate(indices):
            segment = annotation_data[start:end]
            annot_from_eg[i] = mode(segment, keepdims=False)[0]
        graph.add_edge_property("annotation", annot_from_eg)
        print("Edges annotation computed!")

        print("Adding edge length")
        edge_lengths = parallel_get_vessels_lengths(graph, edge_coordinates_name='coordinates_atlas', clip_below_unity=True,
                                 min_chunk_size=MIN_CHUNK_SIZE, n_processes=MAX_PROCS)
        graph.add_edge_property(f"length_coordinates_atlas", edge_lengths)
        print("Edges length computed!")

        # Pre-compute all properties 
        vertex_annotation = graph.vertex_property("annotation")
        edge_geometry_annotation = graph.edge_geometry_property("annotation")
        edge_annotation = graph.edge_property("annotation")
        vertex_degrees = graph.vertex_degrees()
        edge_connectivity = graph.edge_connectivity()
        vertex_coordinates_atlas = graph.vertex_property("coordinates_atlas")
        # edge_geometry_radii = graph.graph_property("edge_geometry_radii")
        edge_radii = graph.edge_property("radii_atlas")
        edge_length_coordinates_atlas = graph.edge_property("length_coordinates_atlas")

        # Compute tortuosity 
        if add_tortuosity:
            print("Computing tortuosity")
            edge_geometry_length_coords = graph.graph_property("edge_geometry_length_coordinates")
            lengths_um = np.array([
                edge_geometry_length_coords[index[0]:index[1]].sum() for index in graph.edge_property("edge_geometry_indices")
            ])
            distances_px = np.linalg.norm(
                vertex_coordinates_atlas[edge_connectivity[:, 0]] - 
                vertex_coordinates_atlas[edge_connectivity[:, 1]], axis=1
            )
            tortuosity = (lengths_um + 1) / (distances_px * 25 + 1)
            print("Tortuosity computed!")

        # Identify self-loops
        print("Computing self-loops")
        is_self_loop = edge_connectivity[:, 0] == edge_connectivity[:, 1]
        print("Self loops computed!")

        # Pre-compute hemisphere properties if they exist
        has_hemisphere = ('hemisphere' in graph.vertex_properties)
        print('has_hemisphere: ', has_hemisphere)
        if has_hemisphere and not ('hemisphere' in graph.edge_properties):
            print("Adding edge hemisphere")
            hemisphere_data = np.array(graph.edge_geometry_property("hemisphere"))
            indices = np.array(graph.edge_property("edge_geometry_indices"))
            hemis_from_eg = np.empty(len(indices), dtype=hemisphere_data.dtype)
            for i, (start, end) in enumerate(indices):
                hemis_from_eg[i] = hemisphere_data[start:end].max()
            graph.add_edge_property("hemisphere", hemis_from_eg)
            print("Edges hemispehre computed!")

        # Check for tip_cell_prediction property 
        has_tip_prediction = 'tip_cell_prediction' in graph.vertex_properties
        if has_tip_prediction:
            print('Vertex property tip_cell_prediction does exist and will be used for csv construction')
            tip_cell_prediction = graph.vertex_property('tip_cell_prediction')

        # Get unique regions and hemispheres
        region_ids = np.unique(vertex_annotation)
        region_ids = region_ids[region_ids != 0]
        hemispheres = np.unique(graph.vertex_property('hemisphere')) if has_hemisphere else [None]

        # Main loop
        for region_id in region_ids:
            for current_hemisphere in hemispheres:
                # Create regional filters 
                if has_hemisphere:
                    vertex_filter = (vertex_annotation == region_id) & (graph.vertex_property('hemisphere') == current_hemisphere)
                    edge_filter = (edge_annotation == region_id) & (graph.edge_property('hemisphere') == current_hemisphere)
                    edge_geometry_filter = (edge_geometry_annotation == region_id) & (graph.edge_geometry_property('hemisphere') == current_hemisphere)
                else:
                    vertex_filter = vertex_annotation == region_id
                    edge_filter = edge_annotation == region_id
                    edge_geometry_filter = edge_geometry_annotation == region_id

                # General edge/vertex statistics 
                n_vertices = np.sum(vertex_filter)
                n_edges = np.sum(edge_filter)
                
                # Degree statistics
                n_deg1 = np.sum(vertex_filter & (vertex_degrees == 1))
                n_deg3 = np.sum(vertex_filter & (vertex_degrees == 3))
                
                # True degree 1 calculation
                if has_tip_prediction:
                    n_true_degree_1 = np.sum((tip_cell_prediction == 1) & vertex_filter)
                else:
                    n_true_degree_1 = 'nan'

                # Radius statistics
                # mean_radius = edge_geometry_radii[edge_geometry_filter].mean() if np.any(edge_geometry_filter) else 0
                mean_radius = edge_radii[edge_filter].mean() if np.any(edge_filter) else 0

                # Tortuosity statistics
                if add_tortuosity:
                    tortuosity_mask = edge_filter & ~is_self_loop
                    mean_tortuosity = tortuosity[tortuosity_mask].mean() if np.any(tortuosity_mask) else 0

                # Length calculations
                length_results = edge_length_coordinates_atlas[edge_filter].sum() if np.any(edge_filter) else 0

                # Build result record
                result_record = {
                    'sample_id': sample_id,
                    'hemisphere': current_hemisphere if has_hemisphere else 'all',
                    'region_id': int(region_id),
                    'n_vertices': n_vertices,
                    'n_degree_1': n_deg1,
                    'n_true_degree_1': n_true_degree_1,
                    'n_degree_3': n_deg3,
                    'n_edges': n_edges,
                    'mean_radius_um': mean_radius * 25, # multiply by 25 to go from voxels (atlas space) to micrometers 
                    'sum_length_atlas_um': length_results * 25
                }
                if add_tortuosity:
                    result_record['mean_tortuosity'] = mean_tortuosity

                sample_results.append(result_record)

        print(f"Done computing graph stats from sample {sample_id} : {graph_path}")

        # Convert all results to DataFrame and merge with sample data
        df = pd.DataFrame(sample_results)
        # complete_df = df.merge(df_sample, on="sample_id")
        
        # Write to CSV
        if not stats_file_path.exists():
            df.to_csv(stats_file_path, index=False)
        else:
            df.to_csv(stats_file_path, index=False, mode='a', header=False)
        print(f"Done writing graph stats in {stats_file_path}")
