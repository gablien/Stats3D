"""
This module contains useful functions for graph analysis
"""

from pathlib import Path


def find_files_by_pattern(directory, pattern):
    """
    Find all files in a directory that match a given pattern using pathlib.
    
    Args:
        directory (Path): The directory path to search in
        pattern (str): The pattern to match
    
    Returns:
        file_paths (List[str]): List of full file paths that match the pattern
    """    
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory")
    
    matching_files = directory.glob(f"**/*{pattern}*")
    file_paths = [path for path in matching_files if path.is_file()]
    
    return sorted(file_paths)


def extract_children_coordinates(region_id, ontology_df):
    """
    Extract the coordinates of the  regions corresponding to the given id and its children 

    Parameters
    ----------
    region_id : int
        The atlas id to consider 

    Returns
    ----------
    region_ids : list
        list of ids of the region + its children 

    """

    region_ids = ontology_df.set_index('id').at[region_id,'all_children_structures_ids'] # we add the id of all children from the region 
    region_ids.append(region_id) # we add the region id
    return region_ids 