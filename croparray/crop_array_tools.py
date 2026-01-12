# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Python code to create, manipulate, and analyze an array of crops from TIF images or videos.
Created: Summer of 2020 (updated Jan 2026 by Tim Stasevich)
Authors: Tim Stasevich & Luis Aguilera.
'''

# Conventions.
# module_name, package_name, ClassName, method_name,
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME,
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

import_libraries = 1
if import_libraries == 1:
    import numpy as np 
    import pandas as pd
    import xarray as xr
    from scipy.ndimage import gaussian_filter1d
    import trackpy as tp
    import matplotlib.pyplot as plt 
    from matplotlib import gridspec
#    import napari
#    import seaborn as sns
    import math
    # from napari.utils import nbscreenshot

from .io import open_croparray, open_croparray_zarr
from .build import _create_crop_array_dataset, create_crop_array
from .measure import best_z_proj, measure_signal, measure_signal_raw
from .plot import montage
from .napari_view import view_montage
from .dataframe import variables_to_df
from .tracking import (
    perform_tracking_with_exclusions,
    tracking_spots,
    detecting_spots,
    to_track_array,
)
from .trackarray.build import track_array
from .trackarray.dataframe import create_tracks_df, track_signals_to_df
from .trackarray.napari_view import display_cell_and_tracks




##### Modules
def print_banner():
    print(" \n"
        "░█████╗░██████╗░░█████╗░██████╗░░█████╗░██████╗░██████╗░░█████╗░██╗░░░██╗\n"
        "██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚██╗░██╔╝\n"
        "██║░░╚═╝██████╔╝██║░░██║██████╔╝███████║██████╔╝██████╔╝███████║░╚████╔╝░\n"
        "██║░░██╗██╔══██╗██║░░██║██╔═══╝░██╔══██║██╔══██╗██╔══██╗██╔══██║░░╚██╔╝░░\n"
        "╚█████╔╝██║░░██║╚█████╔╝██║░░░░░██║░░██║██║░░██║██║░░██║██║░░██║░░░██║░░░\n"
        "░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░\n"
        "                                     by : Luis Aguilera and Tim Stasevich \n\n"        )



def plot_normalized_best_z_raw(ds, figsize=(20, 15)):
    """
    Plot the normalized sum of `best_z_raw` for each track over time.
    
    Parameters:
    - ds (xarray.Dataset): The dataset containing `best_z_raw` and other variables.
    - figsize (tuple): The size of the figure (width, height).
    """
    
    # Sum `best_z_raw` over `y` and `x` dimensions for each track and time
    best_z_raw_sum = ds.best_z.sum(dim=['y', 'x'])
    
    # Normalize to the first time point for each track
    def normalize_to_first_time_point(track_data):
        first_time_point_value = track_data.isel(t=0)  # Value at the first time point
        return track_data / first_time_point_value

    best_z_raw_sum_normalized = best_z_raw_sum.groupby('track_id').map(normalize_to_first_time_point)
    
    # Get track IDs and number of tracks
    track_ids = ds.track_id.values
    num_tracks = len(track_ids)

    # Determine number of rows and columns for the grid
    n_cols = 5
    n_rows = (num_tracks + n_cols - 1) // n_cols  # This ensures enough rows to fit all tracks

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    for idx, track_id in enumerate(track_ids):
        ax = axes[idx]
        # Extract data for the current track
        data = best_z_raw_sum_normalized.sel(track_id=track_id)
        
        # Check if the data has values
        if data.size == 0:
            ax.axis('off')  # If no data, hide the axis
            continue

        time_points = data['t'].values  # Extract time points for the current track
        data_values = data.values.squeeze()  # Get the normalized values

        ax.plot(time_points, data_values, label=f'Track {track_id}')
        ax.set_title(f'Track {track_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Sum of best_z_raw')
        ax.legend()

    # Hide unused subplots
    for ax in axes[num_tracks:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def spot_detect_and_qc(img, minmass=6000, size=5):
    """
    Locates features in an image using trackpy locate function and creates a new image with a pixel at the location of the feature closest to the center. This pixel has the signal value of the feature.

    This function is used to identify and highlight the location of features in an image. This is useful for visualizing the features and their signal values from an croparray dataset with padding. This function can be used to identify or quality control the features in the image such as mRNA and translation spots.

    Parameters:
    img (numpy.ndarray): The input image.
    minmass (int, optional): The minimum integrated brightness. Defaults to 6000.
    size (int, optional): The size of the features in pixels. Defaults to 5.

    Returns:
    numpy.ndarray: A new image of the same size as the input image, with a pixel at the location of the feature closest to the center. The pixel value is the signal value of the feature.

    """
    features = tp.locate(img, size, minmass)
    new_img = np.zeros_like(img)
    if len(features) > 0:
        # Calculate the center of the image
        center_x = img.shape[1] / 2
        center_y = img.shape[0] / 2

        if len(features) > 1:
            # Calculate the Euclidean distance from each feature to the center
            distances = np.sqrt((features['x'] - center_x)**2 + (features['y'] - center_y)**2)
            # Find the index of the feature with the smallest distance
            closest_index = np.argmin(distances)
            x_value = features['x'].values[closest_index]
            y_value = features['y'].values[closest_index]
            signal_value = features['signal'].values[closest_index]
        else:
            x_value = features['x'].values[0]
            y_value = features['y'].values[0]
            signal_value = features['signal'].values[0]

        # Set the pixel at (x_value, y_value) to the maximum pixel value
        new_img[int(y_value), int(x_value)] = signal_value
    return new_img


