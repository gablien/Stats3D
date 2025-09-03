"""
This module contains useful functions for volcano and bar plot 
"""

import os.path
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

try:
    import ClearMap.Settings as settings
except:
    sys.path.append(f"/home/{os.getlogin()}/programs/ClearMap/ClearMap2")
    import ClearMap.Settings as settings

ontology_df = pd.read_json(Path(settings.atlas_folder) / 'ABA_annotation_last.jsonl', lines=True)


def filter_by_samples(result_df, sample_list):
    df = result_df[result_df['sample_id'].isin(sample_list)]
    return df


def filter_by_hemisphere(result_df, hemisphere):
    if 'Hemisphere' in result_df.columns:
        df = result_df[result_df['Hemisphere'].isin(hemisphere)]
    elif 'hemisphere' in result_df.columns:
        df = result_df[result_df['hemisphere'].isin(hemisphere)]
    else:
        print('No hemisphere data found, the hemisphere filter could not be applied')
        df = result_df
    return df


def filter_by_region(result_df, region):
    df = result_df[result_df['region_id'].isin(region)]
    return df


def create_volcano_data(result_df, control_name, test_name, param):
    if 'Structure ID' in result_df.columns:
        structure_id = 'Structure ID'
    elif 'region_id' in result_df.columns:
        structure_id = 'region_id'

    list_id = np.unique(result_df[structure_id])
    fold_list = np.zeros(len(list_id))
    p_list = np.zeros(len(list_id))

    for idx, id in enumerate(list_id):
        group_test = result_df[(result_df['group'] == test_name) & (result_df[structure_id] == id)]
        group_ctl = result_df[(result_df['group'] == control_name) & (result_df[structure_id] == id)]
        if len(group_test) == 0 or len(group_ctl) == 0:
            print(f"Experimental groups are empty! for id {id}")
        stat, pval = ttest_ind(group_test[param], group_ctl[param])
        p_list[idx] = -np.log10(pval)
        if np.mean(group_ctl[param]) != 0:
            fold_list[idx] = np.log2(np.mean(group_test[param]) / np.mean(group_ctl[param]))

    df_volcano = pd.DataFrame()
    df_volcano['region_id'] = list_id
    df_volcano['log2(fold_change)'] = fold_list
    df_volcano['-log10(p-val)'] = p_list

    return df_volcano