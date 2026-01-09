from __future__ import annotations

from typing import Optional

import numpy as np



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



