
def view_montage(my_ca_montage):
    """
    Display a montage of images in Napari based on the number of channels.

    Parameters:
    my_ca_montage (xarray.DataArray or xarray.Dataset): The input dataset with channel information.
    """
    import napari
    num_channels = len(my_ca_montage.ch.values)
    
    # Initialize the Napari viewer
    viewer = napari.Viewer()

    if num_channels == 3:
        viewer.add_image(
            my_ca_montage.sel(ch=0).best_z, 
            colormap='red', 
            name='Channel 1', 
            blending='additive', 
            contrast_limits=[0, my_ca_montage.sel(ch=0).best_z.quantile(0.9995).values]
        )
        viewer.add_image(
            my_ca_montage.sel(ch=1).best_z, 
            colormap='green', 
            name='Channel 2', 
            blending='additive', 
            contrast_limits=[0, my_ca_montage.sel(ch=1).best_z.quantile(0.9995).values]
        )
        viewer.add_image(
            my_ca_montage.sel(ch=2).best_z, 
            colormap='blue', 
            name='Channel 3', 
            blending='additive', 
            contrast_limits=[0, my_ca_montage.sel(ch=2).best_z.quantile(0.9995).values]
        )
    
    elif num_channels == 2:
        viewer.add_image(
            my_ca_montage.sel(ch=0).best_z, 
            colormap='magenta', 
            name='Channel 1', 
            blending='additive', 
            contrast_limits=[0, my_ca_montage.sel(ch=0).best_z.quantile(0.9995).values]
        )
        viewer.add_image(
            my_ca_montage.sel(ch=1).best_z, 
            colormap='green', 
            name='Channel 2', 
            blending='additive', 
            contrast_limits=[0, my_ca_montage.sel(ch=1).best_z.quantile(0.9995).values]
        )

    elif num_channels == 1:
        viewer.add_image(
            my_ca_montage.sel(ch=0).best_z, 
            colormap='green', 
            name='Channel 1', 
            blending='additive', 
            contrast_limits=[0, my_ca_montage.sel(ch=0).best_z.quantile(0.9995).values]
        )
    
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    return viewer
