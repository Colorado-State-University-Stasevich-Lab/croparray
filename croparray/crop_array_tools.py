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

def create_tracks_df(my_ta):
    # Initialize empty lists to store data
    track_dfs = []

    # Iterate over each track_id
    for track_id in my_ta.track_id.values:
        # Filter the xarray dataset by the specified track_id
        track_data = my_ta.sel(track_id=track_id)

        # Filter out rows where xc or yc are both [0, 0]
        valid_indices_xc = np.where(track_data.xc != 0)[0]
        valid_indices_yc = np.where(track_data.yc != 0)[0]
        valid_indices = np.intersect1d(valid_indices_xc, valid_indices_yc)

        # Filter the data based on the valid indices
        filtered_data = track_data.isel(t=valid_indices)

        # Create DataFrame
        track_df = pd.DataFrame({
            'track_id': [track_id] * len(filtered_data.t),
            't': filtered_data.t.values.tolist(),
            'y': filtered_data.yc[:, 0].values.tolist(),
            'x': filtered_data.xc[:, 0].values.tolist()
        })

        # Append the DataFrame to the list
        track_dfs.append(track_df)
    # track_dfs['y'] -= len(my_ta.y.values)
    # track_dfs['x'] -= len(my_ta.x.values)

    # Concatenate all DataFrames in the list
    result_df = pd.concat(track_dfs, ignore_index=True)

    result_df['y'] -= ((len(my_ta.y.values)-1)/2)
    result_df['x'] -= ((len(my_ta.x.values)-1)/2)
    return result_df

def display_cell_and_tracks(img_croparray, tracks_df):
    """
    Display the maximum intensity projection of the images and the tracks in Napari.

    Parameters:
    img_croparray (numpy.ndarray): Array containing image data with dimensions (fov, t, z, x, y, ch).
    tracks_df (pandas.DataFrame): DataFrame containing track information.

    Returns:
    napari.Viewer: The viewer instance with the images and tracks added.
    """
    # Compute the maximum projection along the specified axis
    img_max = np.max(img_croparray[0, :, :, :, :], axis=1)
    
    # Get the number of channels
    num_channels = img_max.shape[-1]
    
    # Initialize the Napari viewer based on the number of channels
    if num_channels == 1:
        viewer_tracks = napari.view_image(img_max[:, :, :, 0], colormap='green',name = 'Ch 1', blending='additive')
    elif num_channels == 2:
        viewer_tracks = napari.view_image(img_max[:, :, :, 0], colormap='magenta',name = 'Ch 1', blending='additive')
        viewer_tracks.add_image(img_max[:, :, :, 1], colormap='green',name = 'Ch 2', blending='additive')
    elif num_channels == 3:
        viewer_tracks = napari.view_image(img_max[:, :, :, 0], colormap='red',name = 'Ch 1', blending='additive')
        viewer_tracks.add_image(img_max[:, :, :, 1], colormap='green',name = 'Ch 2', blending='additive')
        viewer_tracks.add_image(img_max[:, :, :, 2], colormap='blue',name = 'Ch 3', blending='additive')
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    # Add tracks and points to the viewer
    viewer_tracks.add_tracks(tracks_df, name='track tails')
    viewer_tracks.add_points(tracks_df[['t', 'y', 'x']].values, 
                              size=12, 
                              face_color='transparent', 
                              edge_color='yellow', 
                              symbol='disc', 
                              name='track_points')
    
    return viewer_tracks


# Pull out variables in a crop array to a dataframe
def variables_to_df(ca, var_names):
    """
    Creates a pandas dataframe from the specified variables of a crop array.  

    Parameters:
    ----------
    ca: crop array (x-array dataset)
        A crop array.
    var_names: list of str
        Names of the variables in the xarray dataset to convert to a dataframe.
    
    Returns:
    -------
    A concatenated pandas dataframe with the specified variables such that each column of the dataframe corresponds to one coordinate dimension in the crop array. Basically the output corresponds to xr.to_dataframe(), but with multiindex flattened.
    """
        # Check if variables exist in the dataset
    for var in var_names:
        if var not in ca:
            raise ValueError(f"'{var}' not found in the provided xarray dataset.")

    # Check if the variables have the same dimensions
    dims = ca[var_names[0]].dims
    for var in var_names[1:]:
        if ca[var].dims != dims:
            raise ValueError(f"Variables do not have matching dimensions. {var_names[0]} has dimensions {dims} while {var} has dimensions {ca[var].dims}")

    # Convert each variable to a dataframe and concatenate them
    dfs = [ca[var].to_dataframe().reset_index(level=list(range(len(dims)))) for var in var_names]
    
    final_df = pd.concat(dfs, axis=1)
    # Drop duplicate columns if any arise due to the reset index operation
    final_df = final_df.loc[:,~final_df.columns.duplicated()]
    
    return final_df

def track_signals_to_df(my_ta):
    """
    Combine signal and signal_raw data from each channel into a single DataFrame.

    Parameters:
    - my_ta: xarray.Dataset containing 'signal' and 'signal_raw' data for each channel.

    Returns:
    - DataFrame: Combined DataFrame with columns for each signal and signal_raw.
    """
    # Initialize an empty list to hold individual DataFrames for each channel
    combined_data = []

    # Get the number of channels
    num_channels = my_ta.ch.size

    # Loop through each channel to extract and combine the data
    for ch in range(num_channels):
        # Convert signal_raw to DataFrame
        df_signal_raw = my_ta.sel(ch=ch)['signal_raw'].to_dataframe().reset_index()
        
        # Convert signal to DataFrame
        df_signal = my_ta.sel(ch=ch)['signal'].to_dataframe().reset_index()

        # Merge both DataFrames on track_id, t, and ch
        df_combined = pd.merge(df_signal_raw, df_signal, on=['track_id', 't', 'ch'], suffixes=('', '_signal'))

        # Rename columns to distinguish between signal and signal_raw
        df_combined.rename(columns={'signal': f'signal_ch_{ch}', 'signal_raw': f'signal_raw_ch_{ch}'}, inplace=True)

        # Append the combined DataFrame to the list
        combined_data.append(df_combined)

    # Concatenate all DataFrames along the columns
    final_df = pd.concat(combined_data, axis=1)

    # Drop duplicate columns if necessary (like 'ch' appearing multiple times)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Remove unwanted columns
    columns_to_remove = ['ch', 'fov', 'fov_signal']
    final_df = final_df.drop(columns=columns_to_remove, errors='ignore')

    return final_df


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

def plot_trackarray_crops(
    data,
    fov=0,
    track_id=1,
    t=(0, 10, 3),
    rolling=1,
    quantile_range=(0.02, 0.99),
    rgb_channels=(0, 1, 2),
    ch=None,
    suppress_labels=True,
    show_suptitle=True,
):
    """
    Plot track-centered image crops across time and channels using xarray.plot.imshow.

    For each track:
      - Optionally shows an RGB composite row (if >=3 channels and ch is None)
      - Shows grayscale rows for each channel (or only `ch` if provided)

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing a 'best_z' DataArray with dims like:
        (track_id, fov, t, y, x, ch) or (track_id, t, y, x, ch).
    fov : int, default 0
        Field of view to select, if applicable.
    track_id : int or sequence[int], default 1
        Track(s) to plot. One grid per track.
    t : tuple[int, int, int], default (0, 10, 3)
        (start, stop, step) along time axis.
    rolling : int, default 1
        Rolling mean window over time. If 1, no smoothing.
    quantile_range : tuple[float, float], default (0.02, 0.99)
        Quantiles used for per-channel normalization (positive pixels only).
    rgb_channels : tuple, default (0, 1, 2)
        Channels to use for RGB composite when available.
        Duplicates are allowed (e.g. (1, 1, 2)).
    ch : int or None, default None
        If provided, plot only this channel in grayscale.
    suppress_labels : bool, default True
        If True, removes per-crop titles, axis labels, and ticks.
    show_suptitle : bool, default True
        If True, adds a readable figure-level title per track.

    Returns
    -------
    dict[int, xr.DataArray]
        Mapping from track_id -> normalized DataArray used for plotting
        with dims (t, y, x, ch). If `ch` is provided, ch size = 1.
    """
    import numpy as np
    import xarray as xr

    if "best_z" not in data:
        raise KeyError("Dataset must contain a 'best_z' DataArray.")

    def _decorate_facetgrid(g, suptitle=None):
        """Optionally suppress per-panel labels and add a figure-level title."""
        if suppress_labels:
            # Remove facet titles (e.g. t=0, t=3, ...)
            try:
                g.set_titles("")
            except Exception:
                try:
                    g.set_titles(template="")
                except Exception:
                    pass

            # Remove axis labels
            try:
                g.set_xlabels("")
                g.set_ylabels("")
            except Exception:
                pass

        if show_suptitle and suptitle and hasattr(g, "fig") and g.fig is not None:
            try:
                g.fig.suptitle(suptitle, y=1.02)
            except Exception:
                pass

        return g

    track_ids = (
        list(track_id)
        if isinstance(track_id, (list, tuple, np.ndarray))
        else [track_id]
    )

    eps = 1e-6
    results = {}

    for tid in track_ids:
        bz = data["best_z"].sel(track_id=tid)

        # --- FOV selection ---
        if "fov" in bz.dims:
            bz = bz.sel(fov=fov)
        elif "fov" in data.coords:
            try:
                bz = bz.where(data["fov"] == fov, drop=True)
            except Exception:
                pass

        # --- Time slicing ---
        start, stop, step = t
        bz = bz.isel(t=slice(start, stop, step))

        # --- Rolling average ---
        if rolling and rolling > 1:
            bz = bz.rolling(t=rolling, center=True, min_periods=1).mean()

        # --- Single-channel mode ---
        if ch is not None:
            try:
                bz1 = bz.sel(ch=ch)
            except Exception:
                bz1 = bz.isel(ch=int(ch))

            bz1 = bz1.expand_dims("ch").assign_coords(ch=[ch])

            da_pos = bz1.isel(ch=0).where(lambda x: x > 0)
            q0 = da_pos.quantile(quantile_range[0])
            q1 = da_pos.quantile(quantile_range[1])

            normed = (
                ((bz1.isel(ch=0) - q0) / (q1 - q0 + eps))
                .clip(0, 1)
                .expand_dims(ch=[ch])
            )

            g = normed.isel(ch=0).plot.imshow(
                col="t",
                cmap="gray",
                xticks=[] if suppress_labels else None,
                yticks=[] if suppress_labels else None,
                aspect=1,
                size=5,
                vmin=0,
                vmax=1,
                robust=True,
                add_labels=not suppress_labels,
                add_colorbar=False,
            )
            _decorate_facetgrid(g, suptitle=f"track_id={tid} | ch={ch}")

            results[int(tid)] = normed
            continue

        # --- Per-channel normalization ---
        ch_normed = []
        for ch_val in bz["ch"].values:
            da_pos = bz.sel(ch=ch_val).where(lambda x: x > 0)
            q0 = da_pos.quantile(quantile_range[0])
            q1 = da_pos.quantile(quantile_range[1])
            ch_normed.append(
                ((bz.sel(ch=ch_val) - q0) / (q1 - q0 + eps)).clip(0, 1)
            )

        normed = xr.concat(ch_normed, dim="ch").assign_coords(ch=bz["ch"].values)

        # --- RGB composite ---
        if normed.sizes.get("ch", 0) >= 3:
            try:
                rgb_da = normed.sel(ch=list(rgb_channels))
            except Exception:
                rgb_da = normed.isel(ch=slice(0, 3))

            if rgb_da.sizes.get("ch", 0) == 3:
                g = rgb_da.plot.imshow(
                    col="t",
                    rgb="ch",
                    xticks=[] if suppress_labels else None,
                    yticks=[] if suppress_labels else None,
                    aspect=1,
                    size=5,
                    vmin=0,
                    vmax=1,
                    add_labels=not suppress_labels,
                    add_colorbar=False,
                )
                _decorate_facetgrid(g, suptitle=f"track_id={tid} (RGB)")

        # --- Grayscale rows ---
        for ch_val in normed["ch"].values:
            g = normed.sel(ch=ch_val).plot.imshow(
                col="t",
                cmap="gray",
                xticks=[] if suppress_labels else None,
                yticks=[] if suppress_labels else None,
                aspect=1,
                size=5,
                vmin=0,
                vmax=1,
                robust=True,
                add_labels=not suppress_labels,
                add_colorbar=False,
            )
            _decorate_facetgrid(g, suptitle=f"track_id={tid}, ch={ch_val}")

        results[int(tid)] = normed

    return results



def plot_track_signal_traces(
    ta_dataset,
    track_ids,
    rgb=(1, 1, 1),
    colors=("#00f670", "#f67000", "#7000f6"),
    markers=("o", "s", "D"),      # marker per channel
    marker_size=6,                # <-- NEW: line marker size
    scatter_size=25,              # <-- NEW: scatter marker area (points^2)
    markevery=5,                  # <-- NEW: show every Nth marker on lines
    figsize=(7, 2.8),
    ylim=None,
    xlim=None,
    col_wrap=3,
    y2=None,                      # channel index for right axis
    y2lim=None,                   # right-axis limits
    y2_label=None,                # right-axis label
    legend_loc="upper right",     # or "outside"
    show_legend=True
):
    """
    Plot signal traces for a list of track_ids in a subplot grid layout.
    Optionally place one channel on a secondary (right) y-axis.

    Parameters:
    - ta_dataset: xarray Dataset with 'signal' variable
    - track_ids (list[int]): track IDs to plot
    - rgb (tuple[int,int,int]): e.g., (1, 0, 1) = plot ch 0 and 2 on left axis unless one is y2
    - colors (list[str]): color for each channel index
    - figsize (tuple): size of each individual subplot
    - ylim (tuple or None): y-axis limits for left axis
    - xlim (tuple or None): x-axis limits for both axes
    - col_wrap (int): number of subplots per row
    - y2 (int or None): channel index to draw on right axis
    - y2lim (tuple or None): y-axis limits for right axis
    - y2_label (str or None): label for right axis
    - Each channel can have its own color and marker.
    - Right y-axis colored to match its channel.
    - Legend placement: 'upper right', 'lower left', 'outside', etc.
    - marker_size controls line markers; scatter_size controls mean-point scatter.
    """

    sig_df = variables_to_df(ta_dataset, ['signal'])

    sns.set_style("whitegrid")
    sns.set(font_scale=1.1)

    n = len(track_ids)
    ncols = col_wrap
    nrows = math.ceil(n / col_wrap)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        squeeze=False
    )

    # Ensure markers covers all channels
    if len(markers) < len(colors):
        markers = list(markers) + ["o"] * (len(colors) - len(markers))

    for idx, track_id in enumerate(track_ids):
        row, col = divmod(idx, col_wrap)
        ax = axes[row][col]
        ax2 = ax.twinx() if y2 is not None else None

        # Plot each channel
        for ch in range(len(colors)):
            if not ((rgb[ch] == 1) or (y2 == ch)):
                continue

            color = colors[ch]
            marker = markers[ch]
            subset = sig_df[(sig_df['track_id'] == track_id) & (sig_df['ch'] == ch)]
            if subset.empty:
                continue

            target_ax = ax2 if (y2 is not None and ch == y2) else ax

            # Line with markers (disable seaborn's auto-legend)
            sns.lineplot(
                data=subset, x="t", y="signal", ax=target_ax,
                color=color, label=f"ch {ch}",
                lw=2, dashes=False, legend=False,
                marker=marker, markersize=marker_size, markevery=markevery
            )

            # Mean points (also not in legend)
            mean_df = subset.groupby('t')['signal'].mean().reset_index()
            sns.scatterplot(
                data=mean_df, x="t", y="signal", ax=target_ax,
                color=color, s=scatter_size, legend=False
            )

        # Axis labels and limits
        ax.set_title(f"Track {int(track_id)}")
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("intensity (a.u.)")
        if ylim: ax.set_ylim(ylim)
        if xlim: ax.set_xlim(xlim)

        # Right axis styling
        if ax2 is not None:
            right_color = colors[y2 % len(colors)]
            ax2.set_ylabel(y2_label or f"intensity (a.u.) [ch {y2}]", color=right_color)
            if y2lim: ax2.set_ylim(y2lim)
            if xlim:  ax2.set_xlim(xlim)
            ax2.tick_params(axis='y', colors=right_color)
            ax2.spines['right'].set_color(right_color)

        # One combined legend per subplot
        if show_legend:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
            handles, labels = h1 + h2, l1 + l2

            if legend_loc == "outside":
                ax.legend(handles, labels, loc='upper left',
                          bbox_to_anchor=(1.15, 1.0),
                          borderaxespad=0., frameon=True)
            else:
                ax.legend(handles, labels, loc=legend_loc, frameon=True)
        else:
            if ax.get_legend(): ax.get_legend().remove()

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, col_wrap)
        fig.delaxes(axes[row][col])

    # More room for outside legends
    if legend_loc == "outside":
        fig.subplots_adjust(right=0.82)

    plt.tight_layout()
    plt.show()

import numpy as np

def eccentricity_crop(
    img: np.ndarray,
    *,
    window_radius: int | None = 7,
    threshold_percentile: float = 99.0,
    min_area: int = 3,
    subtract_bg: str | None = "median",
    fallback_to_moments: bool = True,
    eps: float = 1e-12,
) -> float:
    """
    Compute eccentricity for a single 2D crop (y,x).

    Method:
      - find peak (brightest pixel) in a center window
      - threshold a window around that peak (percentile)
      - take connected component containing the peak
      - return skimage.regionprops eccentricity
      - fall back to intensity-moment eccentricity if segmentation fails

    Returns float in [0,1] or np.nan.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("eccentricity_crop expects a 2D array (y,x).")

    H, W = img.shape
    r = int(window_radius) if window_radius is not None else None

    # Peak selection window around crop center
    cy, cx = H // 2, W // 2
    if r is None:
        y1, y2, x1, x2 = 0, H, 0, W
    else:
        y1, y2 = max(0, cy - r), min(H, cy + r + 1)
        x1, x2 = max(0, cx - r), min(W, cx + r + 1)

    sub0 = img[y1:y2, x1:x2].astype(np.float64, copy=False)
    if subtract_bg == "median":
        sub0 = sub0 - np.median(sub0)
    elif subtract_bg == "min":
        sub0 = sub0 - np.min(sub0)
    elif subtract_bg is None:
        pass
    else:
        raise ValueError("subtract_bg must be 'median', 'min', or None.")

    iy, ix = np.unravel_index(np.argmax(sub0), sub0.shape)
    ypk, xpk = y1 + int(iy), x1 + int(ix)

    # Segmentation window around peak
    if r is None:
        yy1, yy2, xx1, xx2 = 0, H, 0, W
    else:
        yy1, yy2 = max(0, ypk - r), min(H, ypk + r + 1)
        xx1, xx2 = max(0, xpk - r), min(W, xpk + r + 1)

    sub = img[yy1:yy2, xx1:xx2].astype(np.float64, copy=False)
    if subtract_bg == "median":
        sub = sub - np.median(sub)
    elif subtract_bg == "min":
        sub = sub - np.min(sub)
    elif subtract_bg is None:
        pass

    sub_pos = np.clip(sub, 0, None)
    thr = np.percentile(sub_pos, float(threshold_percentile))
    mask = sub_pos > thr

    if mask.sum() < min_area:
        return _ecc_fallback_moments(img, xpk, ypk, r, subtract_bg, eps) if fallback_to_moments else np.nan

    from skimage.measure import label, regionprops
    lab = label(mask)

    py, px = ypk - yy1, xpk - xx1
    if not (0 <= py < lab.shape[0] and 0 <= px < lab.shape[1]):
        return np.nan

    peak_label = int(lab[py, px])
    if peak_label == 0:
        return _ecc_fallback_moments(img, xpk, ypk, r, subtract_bg, eps) if fallback_to_moments else np.nan

    prop = next((p for p in regionprops(lab) if p.label == peak_label), None)
    if prop is None or prop.area < min_area:
        return _ecc_fallback_moments(img, xpk, ypk, r, subtract_bg, eps) if fallback_to_moments else np.nan

    return float(prop.eccentricity)


def _ecc_fallback_moments(img, xpk, ypk, r, subtract_bg, eps):
    """Intensity-weighted second moments in a window around (xpk, ypk)."""
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape

    if r is None:
        y1, y2, x1, x2 = 0, H, 0, W
    else:
        y1, y2 = max(0, ypk - r), min(H, ypk + r + 1)
        x1, x2 = max(0, xpk - r), min(W, xpk + r + 1)

    sub = img[y1:y2, x1:x2]
    if subtract_bg == "median":
        sub = sub - np.median(sub)
    elif subtract_bg == "min":
        sub = sub - np.min(sub)

    w = np.clip(sub, 0, None)
    wsum = w.sum()
    if wsum <= eps:
        return np.nan

    yy, xx = np.indices(w.shape)
    xx = xx + x1
    yy = yy + y1

    mx = (w * xx).sum() / wsum
    my = (w * yy).sum() / wsum
    dx = xx - mx
    dy = yy - my

    cxx = (w * dx * dx).sum() / wsum
    cyy = (w * dy * dy).sum() / wsum
    cxy = (w * dx * dy).sum() / wsum

    evals = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]])
    a2, b2 = float(evals[1]), float(evals[0])
    if a2 <= eps:
        return np.nan

    e = np.sqrt(max(0.0, 1.0 - (b2 / (a2 + eps))))
    return float(np.clip(e, 0.0, 1.0))


import xarray as xr
import numpy as np

def add_eccentricity_all_channels(
    ca: xr.Dataset,
    *,
    in_var: str = "best_z",
    signal_var: str = "signal",
    out_prefix: str = "ecc",
    signal_cutoff: float | None = None,
    window_radius: int | None = None,  # None => uses ca.xy_pad if present
    threshold_percentile: float = 99.0,
    min_area: int = 3,
    subtract_bg: str | None = "median",
    fallback_to_moments: bool = True,
) -> xr.Dataset:
    """
    Compute eccentricity per crop for all channels in a croparray.

    Adds:
      - f"{out_prefix}_ch{ch}" for each channel, dims (fov,n,t)

    If signal_cutoff is set, ecc is NaN where ca[signal_var] < cutoff (per crop).
    """
    if in_var not in ca:
        raise KeyError(f"Dataset missing {in_var!r}.")
    if signal_cutoff is not None and signal_var not in ca:
        raise KeyError(f"signal_cutoff was provided but dataset missing {signal_var!r}.")

    if window_radius is None:
        window_radius = int(np.asarray(ca["xy_pad"])) if "xy_pad" in ca else None

    for ch in ca[in_var].coords["ch"].values:
        ch_int = int(ch)
        img_da = ca[in_var].sel(ch=ch)  # dims (fov,n,t,y,x)

        ecc = xr.apply_ufunc(
            eccentricity_crop,
            img_da,
            input_core_dims=[["y", "x"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs=dict(
                window_radius=window_radius,
                threshold_percentile=threshold_percentile,
                min_area=min_area,
                subtract_bg=subtract_bg,
                fallback_to_moments=fallback_to_moments,
            ),
        )

        if signal_cutoff is not None:
            sig = ca[signal_var].sel(ch=ch)  # dims (fov,n,t)
            ecc = ecc.where(sig >= signal_cutoff)

        ca[f"{out_prefix}_ch{ch_int}"] = ecc

    return ca


def view_eccentricity_napari(
    ca: xr.Dataset,
    *,
    ecc_var: str,                      # e.g. "ecc_ch2"
    row: str = "n",                    # croparray default
    col: str = "t",
    fov: int = 0,
    ch_images: tuple[int, ...] = (1, 2),
    overlay_opacity: float = 0.35,
    blending: str = "additive",
    show_spots_var: str | None = None,  # e.g. "ch2_spots" if present in ca before montage
    spots_colormap: str = "yellow",
):
    """
    Montage-based QC viewer:
      - montages ca using existing montage(ca, row=..., col=...)
      - shows best_z for selected channels
      - overlays a Labels layer where each tile is colored by ecc_var

    Works on croparray (row='n') and trackarray if you call row='track_id'.
    """
    import napari
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if "best_z" not in ca:
        raise KeyError("Dataset must contain 'best_z'.")
    if ecc_var not in ca:
        raise KeyError(f"Dataset missing {ecc_var!r}.")
    if "xy_pad" not in ca:
        raise KeyError("Expected ca.xy_pad to infer tile size.")

    xy_pad = int(np.asarray(ca["xy_pad"]))
    tile_y = 2 * xy_pad + 1
    tile_x = 2 * xy_pad + 1

    # Build montage (your existing function)
    m = montage(ca, row=row, col=col)

    # Determine grid shape in the montage metadata
    n_rows = m.sizes.get(row, None)
    n_cols = m.sizes.get(col, None)
    if n_rows is None or n_cols is None:
        raise ValueError(f"Montage output missing dims row={row!r} and/or col={col!r}.")

    # Labels array in the montage plane (r,c)
    R = int(m.sizes["r"])
    C = int(m.sizes["c"])
    labels = np.zeros((R, C), dtype=np.int32)

    for rr in range(n_rows):
        r0, r1 = rr * tile_y, (rr + 1) * tile_y
        for cc in range(n_cols):
            c0, c1 = cc * tile_x, (cc + 1) * tile_x
            labels[r0:r1, c0:c1] = 1 + rr * n_cols + cc

    # Eccentricity values per tile from original dataset
    ecc_tile = ca[ecc_var]
    if "fov" in ecc_tile.dims:
        ecc_tile = ecc_tile.isel(fov=fov)

    if row not in ecc_tile.dims or col not in ecc_tile.dims:
        raise ValueError(f"{ecc_var!r} must have dims including row={row!r} and col={col!r}. Got {ecc_tile.dims}.")

    ecc_tile = ecc_tile.transpose(row, col).values

    valid = np.isfinite(ecc_tile)
    vmin = float(np.nanpercentile(ecc_tile, 2)) if valid.any() else 0.0
    vmax = float(np.nanpercentile(ecc_tile, 98)) if valid.any() else 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap()

    label_color = {}
    for rr in range(n_rows):
        for cc in range(n_cols):
            lid = 1 + rr * n_cols + cc
            val = ecc_tile[rr, cc] if (rr < ecc_tile.shape[0] and cc < ecc_tile.shape[1]) else np.nan
            label_color[lid] = cmap(norm(float(val))) if np.isfinite(val) else (0.0, 0.0, 0.0, 0.0)

    # Napari viewer
    viewer = napari.Viewer()

    # Show best_z channels (montage has dims r,c,ch,...)
    for ch in ch_images:
        im = m.best_z.sel(ch=ch)
        vmax_im = float(im.fillna(0).data.max())
        viewer.add_image(
            im,
            name=f"best_z_ch{int(ch)}",
            blending=blending,
            contrast_limits=[0, vmax_im],
        )

    # Optional: add a spots layer if it exists in the montage
    if show_spots_var is not None:
        if show_spots_var not in m:
            raise KeyError(f"Montage output missing {show_spots_var!r}. (Did you add it to ca before montage?)")
        sp = m[show_spots_var]
        viewer.add_image(
            sp,
            name=show_spots_var,
            colormap=spots_colormap,
            blending=blending,
            contrast_limits=[0, float(sp.data.max())],
        )

    viewer.add_labels(
        labels,
        name=f"{ecc_var}_overlay",
        color=label_color,
        opacity=overlay_opacity,
    )

    return viewer
