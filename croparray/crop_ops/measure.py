"""
crop_ops.measure

Functions in crop_ops operate on a *single crop*:

- Input:  2D numpy array (y, x)
- Output: numpy array or scalar

Rules:
- Pure functions: no dataset mutation
- No reliance on global state
- Heavy dependencies (e.g., trackpy) should be imported inside functions
"""

from __future__ import annotations

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
    import trackpy as tp
    import numpy as np
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No maxima survived mass- and size-based filtering.*",
            category=UserWarning,
            module=r"trackpy\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message="All local maxima were in the margins.*",
            category=UserWarning,
            module=r"trackpy\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message="Image is completely black.*",
            category=UserWarning,
            module=r"trackpy\..*",
        )
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
