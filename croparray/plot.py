"""
Plotting utilities for croparray.

Conventions:
- All functions here are pure (no CropArray mutation).
- Dataset-aware functions accept `ds` explicitly.
- Generic helpers accept arrays/images and are auto-exposed via CropArrayPlot.
"""


import numpy as np
import xarray as xr


# Make various montages of crop array for easy viewing in napari
def montage(ca, **kwargs):
    '''
    Returns a montage of a crop array for easier visualizaton.

    Parameters
    ----------
    ca: crop arrray (x-array dataset)
        A crop array.
    col: string (optional) 
        String specifying crop array coordinate to arrange in columns, either 'fov', 'n', or 't' (default). 
    row: string (optional) 
        String specifying crop array coordinate to arrange in rows, either 'fov', 'n' (default), or 't'. 

    Returns
    -------
    A reshaped crop array in which individual crops are arranged in a two-dimensional array of dimensions row x col. If kwarg coordinates for row and col are the same, a two dimensional array of dimensions sqrt(row) x sqrt(row) is returned.   
    '''
    # Get the optional key word arguments (kwargs):
    col = kwargs.get('col', 't')
    row = kwargs.get('row', 'n') 

    if row != col: # arrange crops in rows and columns in the xy plane 
        output = ca.stack(r=(row,'y'),c=(col,'x')).transpose('cell','rep','exp','tracks','fov','n','t','z','r','c','ch', missing_dims='ignore') 

    if row == col:  # arrange crops in square col x col montage in the xy-plane
        col_length = len(ca.coords.get(col))
        my_sqrt = np.sqrt(col_length) # How big a square do we need to make?
        remainder = my_sqrt % 1
        if remainder == 0: 
            my_size = int(my_sqrt) # montage square will be (my_size x my_size)
        else:
            my_size = int(np.floor(my_sqrt) + 1) # montage square will be (my_size + 1) x (my_size + 1)
        # pad w/ 0 so there are enough crops to fill a perfect my_size x my_size square
        pad_amount = my_size*my_size - col_length
        # Reshape dataset so 'col' coordinates is rearranged into a perfect square.
        # See https://stackoverflow.com/questions/59504320/how-do-i-subdivide-refine-a-dimension-in-an-xarray-dataset/59685729#59685729 for details
        output = ca.pad(pad_width={col:(0,pad_amount)}, mode='constant', constant_values = 0  # !!! Careful, ds.pad may change in future x-array
        ).assign_coords(montage_row = np.arange(my_size), montage_col = np.arange(my_size)
        ).stack(montage = ('montage_row', 'montage_col') # make montage_row x montage_col stacked coordinates
        ).reset_index(col, drop=True  # remove 'col' coordinate, but keep in individual x-arrays in dataset
        ).rename({col:'montage'}  # rename 'col' dimension in x-arrays to 'montage'
        ).unstack(                # unstack the montage_row and montage_col coordinates    
        ).stack(r=('montage_row','y'), c=('montage_col','x')  # Arrange crops in c x r square in xy-plane
        ).transpose('cell','rep','exp','tracks','fov','n','t','z','r','c','ch',missing_dims='ignore') # Ensure standard crop array ordering

    return output




def rescale_rgb_0_255(arr):
    """
    Rescale an image array to uint8 [0, 255] using global min/max.
    Works for (Y,X,3) or any array with last dim = channels.
    """
    import numpy as np
    arr = np.asarray(arr, dtype=float)

    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)

    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out


def show_rgb_large(img8, *, scale=1.0, title=None):
    """
    Display an RGB image at an appropriate physical size in matplotlib.

    Parameters
    ----------
    img8 : ndarray
        (Y, X, 3) uint8 image
    scale : float
        Multiplicative scale factor for display size (1.0 ≈ 1 pixel = 1/100 inch)
    """
    import matplotlib.pyplot as plt
    h, w = img8.shape[:2]
    dpi = 100

    fig = plt.figure(figsize=(w / dpi * scale, h / dpi * scale), dpi=dpi)
    plt.imshow(img8)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
