import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .dataframe import variables_to_df



# Tracking with trackpy
def perform_tracking_with_exclusions(df, search_range=10, memory=1):
    """
    Perform particle tracking on DataFrame with option to exclude certain data points 
    and assign them a default track ID.

    Parameters:
    ----------
    df: DataFrame
        DataFrame containing particle coordinates and potentially other data.
    search_range: int
        Maximum distance particles can move between frames.
    memory: int
        Maximum number of frames during which a particle can vanish, then reappear, and still be considered the same particle.

    Returns:
    -------
    DataFrame with tracked particles, including excluded ones assigned a default track ID.
    """
    import trackpy as tp
    # Step 1: Filter out rows with x=0 and y=0
    valid_data = df[(df['x'] != 0) | (df['y'] != 0)]
    excluded_data = df[(df['x'] == 0) & (df['y'] == 0)].copy()
    
    # Step 2: Perform tracking on the filtered data
    tracked_data = tp.link_df(valid_data, search_range=search_range, memory=memory)
    
    # Step 3: Add the filtered-out rows back with a specific track_id
    excluded_data['particle'] = -1  # Assign default track_id to excluded rows
    
    # Combine tracked data with excluded data and sort by original index to maintain order
    combined_data = pd.concat([tracked_data, excluded_data]).sort_index()
    combined_data['track_id'] = combined_data['particle'].fillna(-1).astype(int)+1

    return combined_data

# Create a track array from positions in a crop array
def to_track_array(
    ca,
    channel_to_track: int = 0,
    min_track_length: int = 5,
    search_range: int = 10,
    memory: int = 1,
):
    """
    Track particles in a given croparray dataset and update the croparray dataset with new track IDs,
    filtering out short tracks.

    Parameters:
    - ca: The dataset array
    - channel_to_track: Channel index used for tracking particles
    - min_track_length: Minimum length required for a track to be kept
    - search_range: Search range for linking particles to form tracks
    - memory: Number of frames a track can skip
    """
    from .trackarray.build import track_array

    # Step 1: Prepare the DataFrame for tracking
    dft = variables_to_df(ca, ['xc', 'yc'])
    dft['frame'] = (dft['t'] / ca.dt.values).astype(int)
    dft['x'] = dft['xc']
    dft['y'] = dft['yc']
    
    # Step 2: Perform tracking
    dft_filtered = perform_tracking_with_exclusions(dft[dft['ch'] == channel_to_track], search_range=search_range, memory=memory)
    
    # Step 3: Filter out short tracks
    track_lengths = dft_filtered.groupby('track_id').size()
    short_tracks = track_lengths[track_lengths < min_track_length].index
    dft_filtered.loc[dft_filtered['track_id'].isin(short_tracks), 'track_id'] = 0
    
    # Step 4: Update the original dataset with new track IDs
    dft_filtered.set_index(['fov', 'n', 't'], inplace=True)
    track_id_array = dft_filtered['track_id'].to_xarray()

    # Preserve original IDs (spot-level) once, if present and not already preserved
    if "id" in ca and "spot_id" not in ca:
        ca["spot_id"] = ca["id"]

    # Overwrite id with track IDs (post-tracking semantics)
    ca["id"] = track_id_array

    return track_array(ca, as_object=True)



