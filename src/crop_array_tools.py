# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Python code to create, manipulate, and analyze an array of crops from TIF images or videos.
Created: Summer of 2020
Authors: Tim Stasevich.
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

##### Modules

def create_crop_array(video, df, **kwargs): 
    """
    Creates a crop x-array from a tif video and a dataframe containing the ids and coordinates of spots of interest. Cropping is only performed in the lateral xy-plane (so each crop has all z-slices in the video). Padding in the xy-plane by zeros is added to create crops for spots that are too close to the edge of the video. 

    Parameters
    ----------
    video: numpy array
        A 7D numpy array with intensity information from a tif video. The dimensions of the numpy array must be ordered (fov, f, z, y, x, ch), where fov = field of view, f = frame, z = axial z-coordinate, y = lateral y-coordinate, x = lateral x-coordinate, and ch = channels. Note any dimension can have length one (eg. single fov videos would have an fov dimension of length one or a single channel video would have a ch dimension of length one).  
    df: pandas dataframe
        A dataframe with the ids and coordinates of selected spots for making crops from video. Minimally, the dataframe must have 5 columns (1) 'fov': the fov number for each spot; can also be a filename for each fov. (2) 'id': the integer id of each spot. (3) 'f': integer frame number of each spot. (4) 'yc': the lateral y-coordinate of the spot for centering the crop in y, (5) 'xc': the lateral x-coodinate of the spot for centering the crop in x. Any additional columns must be numeric and will be automatically converted to individual x-arrays in the crop array dataset that have the column header as a name.
    xy_pad: int, optional
        The amount of pixels to pad the centered pixel for each crop in the lateral x and y directions. Note the centered pixel is defined as the pixel containing the coordinates (xc, yc, zc) for each crop. As an example, if xy_pad = 5, then each crop in the crop array will have x and y dimensions of 11 = 2*xy_pad + 1.
    dz: int, optional
        The size of pixels in the x-direction.
    dy: int, optional 
        The size of pixels in the y-direction.   
    dz: int, optional 
        The size of pixels in the z-direction.
    dt: int, optional
        The time between sequential frames in the video.   
    video_filename: str, optional
        The name of the tif video file.
    video_date: str, optional
        The date the video was acquired, in the form 'yyyy-mm-dd'.
    homography: numpy array, optional
        A 3x3 transformation matrix that corrects for the misalignment of channel 0 to the other channels in the video.  

    Returns
    ---------
    A crop x-array dataset ca (i.e. crop array) containing 9 default x-arrays (+ optional x-arrays based on optional inputted df columns).
    Coordinates of x-array dataset: fov, n, t, z, y, x, ch
        fov = [0, 1, ... n_fov]
        n = [0, 1, ... n_crops]
        t = [0, 1, ... n_frames] dt
        z = [0, 1, ... z_slices] dz
        y = [-xy_pad, xy_pad + 1, ... xy_pad] dy
        x = [-xy_pad, xy_pad + 1, ... xy_pad] dx
        ch = [0, 1, ... n_channels]
    Attributes of dataset: filename, date
    X-arrays in dataset:
    1. ca.int -- coords: (fov, n, t, z, y, x ch); attributes: 'units'; uint16
        An X-array containing the intensities of all crops in the crop array.
    2. ca.id -- coords: (fov, n, t); attributes: 'units'; uint16
        An x-array containing the ids of the crops in the video.
    3. ca.yc -- coords: (fov, n, t, ch); attributes: 'units'; uint16
        An x-array containing the yc coordinates of the crops in the video.
    4. ca.xc -- coords: (fov, n, t, ch); attributes: 'units'; uint16
        An x-array containing the zc coordinates of the crops in the video.
    5. ca.xy_pad -- coords: (fov, n, t, ch); attributes: 'units'; uint16
        A 1D array containing xy-pad. 
    6. ca.dt -- coords: (fov, n, t, ch); attributes: 'units'; float
        A 1D arary containing dt.
    7. ca.dz -- coords: (fov, n, t, ch), float
        A 1D arary containing dz.
    9. ca.dy -- coords: (fov, n, t, ch), float
        A 1D arary containing dy.
    9. ca.dx -- coords: (fov, n, t, ch), float
        A 1D arary containing dx.

    """
    # Get the optional key word arguments (kwargs):
    xy_pad = kwargs.get('xy_pad', 5) # default padding of 5 pixels
    my_dx = kwargs.get('dx', 1)
    my_dy = kwargs.get('dy', 1)
    my_dz = kwargs.get('dz', 1)
    my_dt = kwargs.get('dt', 1)
    units = kwargs.get('units',['space','time'])
    name = kwargs.get('name', 'video_filename')
    date = kwargs.get('date', 'video_date') 
    # Get homography matrix; default is a 3D identity matrix for transformating x, y, and z
    homography = kwargs.get('homography', np.eye(3))

    # Get dimensions of video
    n_fov, n_frames, z_slices, height_y, width_x, n_channels = list(video.shape)
    print('Original video dimensions: ', video.shape)

    # Pad video in xy-lateral direction by xy-pad so crops can be made for all spots
    npad = ((0,0),(0,0), (0,0), (xy_pad+1,xy_pad+1), (xy_pad+1,xy_pad+1), (0,0))
    video = np.pad(video, pad_width=npad, mode='constant', constant_values=0)
    print('Padded video dimensions: ', video.shape)

    # Create the 'n' spot/crop counter column for indexing df by 'fov', 'n', and 'f':
    my_frames = np.arange(n_frames) # A list of the frames
    my_crops = df.groupby(['fov','f']) # Group spots by frame and fov
    df['n'] = my_crops.cumcount() # Create a new column 'n' as a cumulative counter of spots per frame per fov
    n_spots_max = df['n'].max() + 1 # Get max number of spots to create numpy fov x n x f array to hold all crops (note add 1 since start from zero)
    print('Max # of spots per frame: ', n_spots_max)
    
    # Create empty array to hold all crop array crops with coordinate dimensions fov, n, f, z, y, x, ch:
    my_crops_all = np.zeros((n_fov, n_spots_max, n_frames, z_slices, 2*xy_pad+1, 2*xy_pad+1, n_channels))
    print('Shape of numpy array to hold all crop intensity data: ', my_crops_all.shape)
    
    # Create arrays for xc and yc with coordinate dimensions (fov, n, f, ch)
    # Note these arrays do depend on channel ch, and we'll use the homography matrix to correct channel (xc, yc) coordinates 
    my_xc_all = np.zeros((n_fov, n_spots_max, n_frames, n_channels))
    my_yc_all = np.zeros((n_fov, n_spots_max, n_frames, n_channels))
    print('Shape of xc and yc numpy arrays: ', my_xc_all.shape)

    # Create arrays to hold crop 'id' and any other extra numeric columns in df
    # Note these will create x-arrays with coordinate dimensions (fov, n, f)
    my_columns=list(df.columns)   # List of columns for making x-arrays
    my_columns.remove('fov')        # but need to remove the common layer coordinates 'fov', 'f', 'yc', and 'xc'
    my_columns.remove('f') 
    my_columns.remove('yc')
    my_columns.remove('xc')
    my_columns.remove('n')  # also need to remove this newly made column
    my_layers = np.zeros((len(my_columns), n_fov, n_spots_max, n_frames)) 
    print('Shape of extra my_layers numpy array: ', my_layers.shape)
    
    # Assign crops to empty arrays defined above
    my_fov_ind = 0  # index counter for fov (in case fovs are counted from non-zero start or filenames)
    for my_fov in df['fov'].unique():
        my_f_ind = 0        # index counter for time (in case times start from non-zero start or filenames)
        for my_f in df['f'].unique():            
            # collect all crops at my_fov and my_f 
            my_spots = df[(df['f'] == my_f) & (df['fov'] == my_fov)]
            # get list of all crop counter ns:
            my_ns = my_spots['n'].values.astype(int) # this preserves order in df         
            # fill my_layers numpy arrays
            col_counter = 0
            for col in my_columns:   # these are the other columns besides 'n', 'f, and 'fov' in the 
                my_vals = my_spots[col].round().values.astype(int) # this preserves order in df, so same as my_ns above
                my_layers[col_counter, my_fov_ind, :len(my_vals), my_f_ind] = my_vals 
                col_counter = col_counter + 1
            # create temp arrays to hold (x, y) coords in all channels for all spots at my_fov and my_f
            my_x = np.zeros((n_channels, len(my_ns)))
            my_y = np.zeros((n_channels, len(my_ns)))
            # correct (x,y) coordinates of all crops at my_f and my_fov using homography matrix
            # use the homography to correct channels 1 and 2 (assumed channel 0 is red channel)        
            for ch in np.arange(n_channels):
                if ch == 0:  # don't correct channel 0
                    my_x[ch] = (my_spots['xc'] + xy_pad + 1).round(0).values.astype(int)
                    my_y[ch] = (my_spots['yc'] + xy_pad + 1).round(0).values.astype(int)
                else:   # correct other channels using same homography (since green/blue are image on same camera)
                    temp = [list(np.dot(homography,np.array([pos[0],pos[1],1]))[0:2]) 
                            for pos in my_spots[['xc','yc']].values]
                    my_x[ch], my_y[ch] = np.array(temp).T
                    my_x[ch] = (my_x[ch] + xy_pad + 1).round(0).astype(int)
                    my_y[ch] = (my_y[ch] + xy_pad + 1).round(0).astype(int) 
            for i in my_ns:  # note my_ns is alreayd an index
                for ch in np.arange(n_channels):
                    # create all 3D crops in crop array using corrected x and y values:
                    my_crops_all[my_fov_ind,i,my_f_ind,:,:,:,ch] = video[my_fov_ind, my_f_ind,:,
                            my_y[ch,i].astype(int)-xy_pad:my_y[ch,i].astype(int)+xy_pad+1,
                            my_x[ch,i].astype(int)-xy_pad:my_x[ch,i].astype(int)+xy_pad+1,ch]
                    # create xc array
                    my_xc_all[my_fov_ind, i, my_f_ind, ch] = my_x[ch,i]
                    my_yc_all[my_fov_ind, i, my_f_ind, ch] = my_y[ch,i] 
            my_f_ind = my_f_ind + 1  
        my_fov_ind = my_fov_ind + 1

    # Create X-arrays from the data arrays to go into the X-array dataset: 
    #Create coordinates
    n = np.arange(n_spots_max)
    t = np.arange(n_frames)*my_dt
    z = np.arange(z_slices)*my_dz
    y = np.arange(-xy_pad,xy_pad+1)*my_dy
    x = np.arange(-xy_pad,xy_pad+1)*my_dx
    ch = np.arange(n_channels)
    fov = np.arange(n_fov)
    dx = xr.DataArray(my_dx, coords=[], dims=[], attrs={'units':'nm'}) 
    dy = xr.DataArray(my_dy, coords=[], dims=[], attrs={'units':'nm'}) 
    dz = xr.DataArray(my_dz, coords=[], dims=[], attrs={'units':'nm'}) 
    dt = xr.DataArray(my_dt, coords=[], dims=[], attrs={'units':'sec'}) 
    intensity = xr.DataArray(my_crops_all.astype(int), coords=[fov, n, t, z, y, x, ch], dims=['fov', 'n', 't', 'z', 'y', 'x', 'ch'], attrs = {'units':'intensity (a.u.)'})
    xc = xr.DataArray(my_xc_all.astype(int), coords= [fov, n, t, ch], dims=['fov', 'n', 't', 'ch'], attrs = {'units':units[0]})
    yc = xr.DataArray(my_yc_all.astype(int), coords= [fov, n, t, ch], dims=['fov', 'n', 't', 'ch'], attrs = {'units':units[0]})
    optional_layers = [xr.DataArray(my_layers[col], coords = [fov, n, t], dims=['fov', 'n', 't']) for col in np.arange(len(my_columns))] 
    
    #Set up dictionary of x-arrays for making a dataset
    dict1 = dict(zip(my_columns, optional_layers))    
    dict2 = {
    'int': intensity,
    'xc': xc,
    'yc': yc,
    'dx': dx,
    'dy': dy,
    'dz': dz,
    'dt': dt
    }
    dict2.update(dict1)

    # Create the X-array dataset
    ds = xr.Dataset(
    dict2, 
    attrs = {'units': units, 'name': name, 'date': date}
    )

    return ds
