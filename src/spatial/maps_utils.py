import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tifffile

from ClearMap import Settings as settings
import ClearMap.IO.IO as clearmap_io


def create_masked_effect_size_map(pval_path, control_path, test_path):
    """
    pval_path (Path object): path to p-value map
    control_path (Path object): path to control density map
    test_path (Path object): path to pregnant density map
    atlas_path (Path object): path to atlas
    """
    pval_map = tifffile.imread(pval_path)
    virgin = tifffile.imread(control_path)
    pregnant = tifffile.imread(test_path)

    # create effect size map, masked by significative p-values
    binary_mask = np.where(np.any(pval_map != 0, axis=-1), 1, 0).astype(np.uint8)
    diff = pregnant - virgin
    diff_masked = diff * binary_mask

    # save
    save_path = pval_path.with_stem(f"{pval_path.stem}_effect_size")
    tifffile.imsave(save_path, diff_masked)    


def create_effect_size_map(control_path, test_path, save_path):
    """
    pval_path (Path object): path to p-value map
    control_path (Path object): path to control density map
    test_path (Path object): path to pregnant density map
    atlas_path (Path object): path to atlas
    """
    virgin = tifffile.imread(control_path)
    pregnant = tifffile.imread(test_path)

    # create effect size map, masked by significative p-values
    diff = pregnant - virgin

    # save
    tifffile.imsave(save_path, diff)    


def create_ratio_map(control_path, test_path, save_path):
    """
    pval_path (Path object): path to p-value map
    control_path (Path object): path to control density map
    test_path (Path object): path to pregnant density map
    atlas_path (Path object): path to atlas
    """
    virgin = tifffile.imread(control_path)
    pregnant = tifffile.imread(test_path)

    # create effect size map, masked by significative p-values
    ratio = pregnant / virgin

    # save
    tifffile.imsave(save_path, ratio)  


def create_zscore_map(control_directory, test_directory, save_path):
    control_files = list(Path(control_directory).glob('*.tif'))
    test_files = list(Path(test_directory).glob('*.tif'))
    control_maps = np.stack([tifffile.imread(f) for f in control_files], axis=0)
    test_maps = np.stack([tifffile.imread(f) for f in test_files], axis=0)
    print(f"Control shape: {control_maps.shape}, Test shape: {test_maps.shape}")

    # Calculate voxel-wise mean and std for virgin group
    control_mean = np.mean(control_maps, axis=0)
    control_std = np.std(control_maps, axis=0, ddof=1)  # ddof=1 for sample std

    # Avoid division by zero - set std to nan where it's 0
    control_std[control_std == 0] = np.nan

    # Calculate z-score for each test map
    z_scores = (test_maps - control_mean) / control_std

    # Mean z-score map across test group
    mean_z_score_map = np.nanmean(z_scores, axis=0)
    tifffile.imsave(save_path, mean_z_score_map.astype(np.float32)) 


def extract_subregion_from_map(work_dir, region, ontology_df, custom_name=None):
    """
    work_dir (Path): the directory containing all the maps to subregion + an atlas of same size and orientation
    region: id (int) of the parent region to extract, or list of ids (list)
    """

    if type(region) == int:
        target_regions = ontology_df.set_index('id').at[region, 'all_children_structures_ids']
        target_regions.append(region)
    elif type(region) == list:
        target_regions = region
    else:
        raise('Type of region is not recognized, please select an int or a list')
    target_regions_set = set(target_regions)

    if custom_name:
        region_name = custom_name
    else:
        region_name = ontology_df[ontology_df['id'] == region]['name'].iloc[0]

    # Get all tif files and put the atlas in first position
    tif_files = [f for f in work_dir.glob("*.tif")]
    sorted_files = sorted(tif_files, key=lambda x: (0 if ("ABA" in x.name or "atlas" in x.name) else 1, x.name))

    structure_mask = None 

    for filepath in sorted_files:
        print("processing ", filepath.name)
        
        # process atlas first to create the mask 
        if "ABA" in filepath.name or "atlas" in filepath.name:
            annotation_file = tifffile.imread(filepath)

            structure_mask = np.isin(annotation_file, list(target_regions_set))
            # structure_mask = np.zeros(annotation_file.shape, dtype='bool')
            # for child in target_regions_set:
            #     structure_mask = np.logical_or(structure_mask, annotation_file == child)
            annotation_file[np.invert(structure_mask)] = 0
            
            output_path = work_dir / f"{filepath.stem}_{region_name}{filepath.suffix}"
            tifffile.imwrite(output_path, annotation_file)
            
        # process density maps and p-values maps
        else:
            if structure_mask is None:
                raise ValueError("Atlas must be processed first!")
            
            heatmap = tifffile.imread(filepath)

            if len(heatmap.shape) == 4:
                for channel in range(heatmap.shape[3]):
                    heatmap[~structure_mask, channel] = 0
            elif len(heatmap.shape) == 3:
                heatmap[np.invert(structure_mask)] = 0
            
            output_path = work_dir / f"{filepath.stem}_{region_name}{filepath.suffix}"
            tifffile.imwrite(output_path, heatmap)


def compute_atlas_boundaries(atlas_path):
    """
    atlas_path (str): path to atlas
    """
    atlas = tifffile.imread(atlas_path).astype(np.uint8)

    boundaries = np.zeros_like(atlas, dtype=np.uint8)
    for z in range(atlas.shape[0]):
        slice_img = atlas[z]
        # Compute gradient: pixels that have neighbors with a different label
        gx = cv2.Sobel(slice_img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(slice_img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Any gradient > 0 indicates boundary between different labels

        boundary = (grad_mag > 0).astype(np.uint8)
        # boundary = binary_dilation(boundary)
        # boundary = skeletonize(boundary)
        boundaries[z] = boundary

    return boundaries


def superpose_atlas_boundaries(atlas_path, input_path, intensity=5):
    """
    atlas_path(Path object): path to atlas
    input_path(Path object): path to density map
    """
    boundaries = compute_atlas_boundaries(atlas_path)
    map = tifffile.imread(input_path)
    
    overlay = map.copy()
    overlay[boundaries > 0] = intensity

    # save
    save_path = input_path.with_stem(f"{input_path.stem}_with_atlas")
    tifffile.imsave(save_path, overlay)


def signed_max_projection(volume, axis):
    abs_max_indices = np.argmax(np.abs(volume), axis=axis)
    return np.take_along_axis(volume, np.expand_dims(abs_max_indices, axis=axis), axis=axis).squeeze(axis=axis)


def max_project(volume, orientation):
    axis = {'coronal': 0, 'horizontal': 1, 'sagittal': 2}.get(orientation)
    if axis is None:
        raise ValueError(f"Unknown orientation: {orientation}")
    return signed_max_projection(volume, axis)


def save_projection(image, output_path, colormap, contrast):
    plt.imsave(
        output_path,
        image,
        cmap=colormap, 
        vmin=contrast[0],
        vmax= contrast[1]
    )


def create_projected_map(folder, pattern, region, colormap, contrast):
    for filename in os.listdir(folder):
        if filename.endswith(".tif") and pattern in filename:
            if region is None or region in filename:
                filepath = folder / filename
                print(f"Processing {filepath}")
                volume = tifffile.imread(filepath)

                orientations = ['coronal', 'horizontal', 'sagittal']
                for orientation in orientations:
                    projection = max_project(volume, orientation)
                    out_filename = f"{os.path.splitext(filename)[0]}_{orientation}.png"
                    if region:
                        output_folder = folder / f'projection_{region}'
                    else: 
                        output_folder = folder / 'projection'
                    os.makedirs(output_folder, exist_ok=True)
                    out_path = os.path.join(output_folder, out_filename)
                    save_projection(projection, out_path, colormap, contrast)
                    print(f"Saved projection: {out_path}")



def create_thick_projection(folder, pattern, orientation, start_plane, end_plane, colormap, contrast):
    '''
    folder: str, the folder containing the tif maps
    pattern: str, the pattern for selection of files to project. 
    orientation: str, either 'coronal', 'sagittal' or 'horizontal' (works only if your tif image is in coronal orientation)
    start_plane: int, first plane for the projection 
    end_place: int, last plane to be considered
    colormap: str, matplotlib colormap. We recommend PiYG for diverging maps.
    contrast: list of int, contains the min and max of contrast range   
    '''
    for filename in os.listdir(folder):
        if filename.endswith(".tif") and pattern in filename:
            filepath = folder / filename
            
            print(f"Processing {filepath}")
            volume = tifffile.imread(filepath)

            if start_plane < 0 or end_plane >= volume.shape[0]:
                raise ValueError(f"Plane indices must be between 0 and {volume.shape[0]-1}")
            
            sub_volume = volume[start_plane:end_plane+1, :, :]
            projection = np.max(sub_volume, axis=0)

            out_filename = f"{os.path.splitext(filename)[0]}_{orientation}_{start_plane}-to-{end_plane}.png"

            output_folder = folder / 'projection'
            os.makedirs(output_folder, exist_ok=True)
            out_path = os.path.join(output_folder, out_filename)
            save_projection(projection, out_path, colormap, contrast)
            print(f"Saved projection: {out_path}")