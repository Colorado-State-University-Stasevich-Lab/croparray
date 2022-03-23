# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Python code to create, manipulate, and analyze an array of crops from TIF images or videos.
Created: Summer of 2020
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



def create_crop_array(video, df, **kwargs): 
    """
    Creates a crop x-array from a tif video and a dataframe containing the ids and coordinates of spots of interest. Cropping is only performed in the lateral xy-plane (so each crop has all z-slices in the video). Padding in the xy-plane by zeros is added to create crops for spots that are too close to the edge of the video. 
    Parameters
    ----------
    video: numpy array
        A 6D numpy array with intensity information from a tif video. The dimensions of the numpy array must be ordered (fov, f, z, y, x, ch), where fov = field of view, f = frame, z = axial z-coordinate, y = lateral y-coordinate, x = lateral x-coordinate, and ch = channels. Note any dimension can have length one (eg. single fov videos would have an fov dimension of length one or a single channel video would have a ch dimension of length one).  
    df: pandas dataframe
        A dataframe with the ids and coordinates of selected spots for making crops from video. Minimally, the dataframe must have 5 columns (1) 'fov': the fov number for each spot; can also be a filename for each fov. (2) 'id': the integer id of each spot. (3) 'f': integer frame number of each spot starting from zero. (4) 'yc': the lateral y-coordinate of the spot for centering the crop in y, (5) 'xc': the lateral x-coodinate of the spot for centering the crop in x. Any additional columns must be numeric and will be automatically converted to individual x-arrays in the crop array dataset that have the column header as a name.
    xy_pad: int, optional
        The amount of pixels to pad the centered pixel for each crop in the lateral x and y directions. Note the centered pixel is defined as the pixel containing the coordinates (xc, yc) for each crop. As an example, if xy_pad = 5, then each crop in the crop array will have x and y dimensions of 11 = 2*xy_pad + 1.
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
    1. ca.int -- dims: (fov, n, t, z, y, x ch); attributes: 'units'; uint16
        An X-array containing the intensities of all crops in the crop array.
    2. ca.id -- dims: (fov, n, t); attributes: 'units'; uint16
        An x-array containing the ids of the crops in the video.
    3. ca.yc -- dims: (fov, n, t, ch); attributes: 'units'; uint16
        An x-array containing the yc coordinates of the crops in the video.
    4. ca.xc -- dims: (fov, n, t, ch); attributes: 'units'; uint16
        An x-array containing the xc coordinates of the crops in the video.
    5. ca.xy_pad -- dims: (); attributes: 'units'; uint16
        A 1D array containing xy-pad. 
    6. ca.dt -- dims: (); attributes: 'units'; float
        A 1D arary containing dt.
    7. ca.dz -- dims: (); attributes: 'units'; float
        A 1D arary containing dz.
    9. ca.dy -- dims: (); attributes: 'units'; float
        A 1D arary containing dy.
    9. ca.dx -- dims: (); attributes: 'units'; float
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

    # Pad video in xy-lateral direction by xy_pad so crops can be made for all spots
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
    my_fov_ind = 0  # index counter for fov (in case fovs are listed as filenames)
    for my_fov in df['fov'].unique():
        for my_f in np.sort(df['f'].unique()): # frames MUST be integers counting from 0       
            # collect all crops at my_fov and my_f 
            my_spots = df[(df['f'] == my_f) & (df['fov'] == my_fov)]
            # get list of all crop counter ns:
            my_ns = my_spots['n'].values.astype(int) # this preserves order in df         
            # fill my_layers numpy arrays
            col_counter = 0
            for col in my_columns:   # these are the other columns besides 'n', 'f, and 'fov' in the 
                my_vals = my_spots[col].round().values.astype(int) # this preserves order in df, so same as my_ns above
                my_layers[col_counter, my_fov_ind, :len(my_vals), my_f] = my_vals 
                col_counter = col_counter + 1
            # create temp arrays to hold (x, y) coords in all channels for all spots at my_fov and my_f
            my_x = np.zeros((n_channels, len(my_ns)))
            my_y = np.zeros((n_channels, len(my_ns)))
            # correct (x,y) coordinates of all crops at my_f and my_fov using homography matrix
            # use the homography to correct channels 1 and 2 (assumed channel 0 is red channel)        
            for ch in np.arange(n_channels):
                if ch == 0 & len(my_spots)>0:  # don't correct channel 0
                    my_x[ch] = (my_spots['xc'] + xy_pad + 1).round(0).values.astype(np.int16)
                    my_y[ch] = (my_spots['yc'] + xy_pad + 1).round(0).values.astype(int)
                elif len(my_spots)>0:   # correct other channels using same homography (since green/blue are image on same camera)
                    temp = [list(np.dot(homography,np.array([pos[0],pos[1],1]))[0:2]) 
                            for pos in my_spots[['xc','yc']].values]
                    my_x[ch], my_y[ch] = np.array(temp).T
                    my_x[ch] = (my_x[ch] + xy_pad + 1).round(0).astype(int)
                    my_y[ch] = (my_y[ch] + xy_pad + 1).round(0).astype(int) 
            for i in my_ns:  # note my_ns is alreayd an index
                for ch in np.arange(n_channels):
                    # create all 3D crops in crop array using corrected x and y values:
                    my_crops_all[my_fov_ind,i,my_f,:,:,:,ch] = video[my_fov_ind, my_f,:,
                            my_y[ch,i].astype(int)-xy_pad:my_y[ch,i].astype(int)+xy_pad+1,
                            my_x[ch,i].astype(int)-xy_pad:my_x[ch,i].astype(int)+xy_pad+1,ch]
                    # create xc array
                    my_xc_all[my_fov_ind, i, my_f, ch] = my_x[ch,i]
                    my_yc_all[my_fov_ind, i, my_f, ch] = my_y[ch,i]  
        my_fov_ind = my_fov_ind + 1

    # Create X-arrays from the data arrays to go into the X-array dataset: 
    #Create coordinates
    n = xr.DataArray(np.arange(n_spots_max).astype(np.int16), attrs={'long_name':'crop count'})
    t =  xr.DataArray(np.arange(n_frames)*my_dt, attrs={'units':units[1],'long_name':'time'})
    z = xr.DataArray(np.arange(z_slices)*my_dz, attrs={'units':units[0],'long_name':'axial z-distance'})
    y = xr.DataArray(np.arange(-xy_pad,xy_pad+1)*my_dy, attrs={'units':units[0],'long_name':'radial y-distance'})
    x = xr.DataArray(np.arange(-xy_pad,xy_pad+1)*my_dx, attrs={'units':units[0],'long_name':'radial x-distance'})
    ch = np.arange(n_channels)
    fov = np.arange(n_fov)

    #Create x-array variables/layers
    dx = xr.DataArray(my_dx, coords=[], dims=[], attrs={'units':units[0],'long_name':'x-resolution'}) 
    dy = xr.DataArray(my_dy, coords=[], dims=[], attrs={'units':units[0],'long_name':'y-resolution'}) 
    dz = xr.DataArray(my_dz, coords=[], dims=[], attrs={'units':units[0],'long_name':'z-resolution'}) 
    dt = xr.DataArray(my_dt, coords=[], dims=[], attrs={'units':units[1], 'long_name':'temporal resolution'}) 
    xy_pad = xr.DataArray(xy_pad, coords=[], dims=[], attrs={'units':'pixels'}) 
    intensity = xr.DataArray(my_crops_all.astype(int), coords=[fov, n, t, z, y, x, ch], dims=['fov', 'n', 't', 'z', 'y', 'x', 'ch'], attrs = {'units':'intensity (a.u.)','long_name':'intensity'})
    xc = xr.DataArray(my_xc_all.astype(int), coords= [fov, n, t, ch], dims=['fov', 'n', 't', 'ch'], attrs = {'units':'pixels','long_name':'crop center x'})
    yc = xr.DataArray(my_yc_all.astype(int), coords= [fov, n, t, ch], dims=['fov', 'n', 't', 'ch'], attrs = {'units':'pixels','long_name':'crop center y'})
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
    'dt': dt,
    'xy_pad': xy_pad
    }
    dict2.update(dict1)

    # Create the X-array dataset
    ds = xr.Dataset(
    dict2, 
    attrs = {'name': name, 'date': date}
    )
    return ds


def best_z_proj(ca, **kwargs):
    '''
    Returns an x-array that holds the best-z projection of intensities of spots in a reference channel and augments ca to include a 'ca.zc' layer that holds the best-z values.

    Parameters
    ----------
    ca: crop array (x-array dataset)
        A crop array.
    ref_ch: int, optional 
        A reference intensity channel for finding the best-z projection. Default: ref_ch = 0
    disk_r: int, optional
        The radius of the disk (in pixels) to make measurements to determine the best-z slice. Default: disk_r = 1
    roll_num: int, optional
        The number of z-slices to use in the rolling-z max projection for determining the best-z slice. min_periods in the da.rolling() function is set to 1 so there will be no Nans at the z-edges of crops. Default: roll_num = 1.

    Returns
    ------- 
    A 'best-z' x-array with dimensions (fov,n,t,y,x,ch). The best-z x-array contains the intensities of each crop in the 'best' z-slice, where 'best' is defined as the slice having the maximal intensity within a centered xy disk of radius disk_r pixels. A rolling-z maximum projection (over roll_n z-slices) can optionally be performed so best-z represents a max-z projection across multiple z-slices. In addition, the inputted crop array ca is augmented to now contain a 'zc' layer tha contains the best-z position of each crop. 
    
    '''
    # Get the optional key word arguments (kwargs):
    ref_ch = kwargs.get('ref_ch', 0)
    disk_r = kwargs.get('disk_r', 1) 
    roll_n = kwargs.get('roll_n', 1)

    res = ca.dx # resolution for defining disk to make measurements to determine best-z

    # Get z-signals in disk within each z-plane and apply rolling z-average of these signals
    z_sig = ca.sel(ch=ref_ch).int.where(lambda a: a.x**2 + a.y**2 <= (disk_r*res)**2).mean(dim=['x','y']).rolling(z=roll_n, center=True, min_periods=1).max()

    # Choose z-plane in ca.int corresponding to max z-signal for each channel, then concatenate x-arrays with coordinate channels
    output = xr.concat([ca.int.sel(ch=i).rolling(z=roll_n,center=True,min_periods=1).max().isel(z_sig.argmax(dim=['z'])) for i in ca.ch], dim='ch')
    
    # Add/overwrite the 'zc' layer in the inputted crop-array
    ca['zc'] = z_sig.argmax(dim='z')
    ca.zc.attrs['units']='pixels'
    ca.zc.attrs['long_name']='crop center z'

    return output


def measure_signal(ca, **kwargs):
    '''
    A function to measure and visualize the intensity signal of all crops in the crop array ca. 

    Parameters
    ----------
    ca: crop arrray (x-array dataset)
        A crop array.
    ref_ch: int, optional 
        A reference intensity channel for finding the best-z plane for measurements. Default: ref_ch = 0    
    disk_r: int, optional
        The radius (in pixels) within which the intensity signal for each crop is measured. Default: disk_r = 1
    disk_bg: int, optional
        The radius (in pixels) of an outer ring (of width one pixel) within which which the background signal for each crop is measured. Default: disk_bg = ca.xy_pad.
    roll_num: int, optional
        The number of z-slices to use in the rolling-z max projection for determining the best-z slices to perform intensity measurements. min_periods in the da.rolling() function is set to 1 so there will be no Nans at the z-edges of crops. Default: roll_num = 1.

    Returns
    ------- 
    An augmented crop array ca with two additional variables: (1) ca.best_z is an x-array with dimensions (fov,n,t,y,x,ch) that contains the best-z-projection after background subtraction; (2) ca.signal is an x-array with dimensions (fov,n,t,ch) that contains the background-subtracted intensity signal of each crop in ca.best_z. 
    '''
    # Get the optional key word arguments (kwargs):
    my_ref_ch = kwargs.get('ref_ch', 0)
    my_disk_r = kwargs.get('disk_r', 1) 
    my_disk_bg = kwargs.get('disk_bg',ca.xy_pad) 
    my_roll_n = kwargs.get('roll_n', 1)

    # Create best-z projection (if not already)
    best_z = best_z_proj(ca, ref_ch=my_ref_ch, disk_r=my_disk_r, roll_n=my_roll_n)
    
    # Make mask for measuring within inner ring (the disk):
    disk_sig = best_z.where(lambda a: a.x**2 + a.y**2 <= (my_disk_r*ca.dx)**2).mean(dim=['x','y'])
    
    # Make mask for measuring background within outer ring (the donut):
    donut_sig = best_z.where(lambda a: (a.x**2 + a.y**2 >= (my_disk_bg*ca.dx)**2) & (a.x**2 + a.y**2 < ((my_disk_bg+1)*ca.dx)**2)).mean(dim=['x','y'])

    # Measure signal as disk - donut: 
    signal = disk_sig - donut_sig 

    # Add best_z variable to ca
    ca['best_z'] = best_z - disk_sig
    ca['best_z'].attrs['units'] = 'intensity (a.u.)'
    ca['best_z'].attrs['long_name'] = 'max intensity projection into best-z plane(s)'

    # Add best_z_signal variable to ca:
    ca['signal'] = signal
    ca['signal'].attrs['units'] = 'intensity (a.u.)'
    ca['signal'].attrs['long_name'] = 'crop signal'

    pass

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
        output = ca.stack(r=(row,'y'),c=(col,'x')).transpose('tracks','fov','n','t','z','r','c','ch', missing_dims='ignore') 

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
        ).transpose('tracks','fov','n','t','z','r','c','ch',missing_dims='ignore') # Ensure standard crop array ordering

    return output


# Detecting particles for each frame
def tracking_spots(img,particle_diameter=5,max_distance_movement=5,min_trajectory_length=5, num_iterations = 100,show_plots=True):
    """
    Creates a 

    Parameters
    ----------
    img: NumPy array
        A 3D NumPy array with intensity information from a tif video. The dimensions of the NumPy array must be ordered (f, y, x), where f = frame, y = lateral y-coordinate, and x = lateral x-coordinate.  
    particle_diameter: int
        Particle size in pixels. This has to be an odd integer.
    max_distance_movement: int, optional
        The maximum distance that a particle can move between frames.
    min_trajectory_length: int, optional
        Minimum frames in a trajectory.
    num_iterations: int, optional 
        The number of iterations is used to automatically calculate an intensity threshold.
    show_plots: bool, optional 
        Flag to indicate if the plot with the optimization process is returned.
        
    Returns
    ---------
    A dataset with the following fields 'fov','id','f','zc','yc','xc','MEAN_INTENSITY'

    """    
    num_spots = []
    number_time_points = img.shape[0]
    if number_time_points >20:
        smothing_window = 5
    else:
        smothing_window = 2
    smothing_window = 10
    tested_intensities = np.round(np.linspace(40, np.amax(img), num_iterations ),0)
    for i, int_tested in enumerate (tested_intensities):
        try:
            f = tp.locate(img[0,:,:], diameter=particle_diameter, minmass = int_tested )
            num_spots.append(len(f))
        except:
            num_spots.append(0)
    num_spots =np.array(num_spots)   
    
    # Optimization process for selecting intensity
    vector_detected_spots = num_spots/ np.max(num_spots)
    smooth_vector_detected_spots = gaussian_filter1d(vector_detected_spots, smothing_window)
    second_derivative_vector_detected_spots = np.gradient(np.gradient(smooth_vector_detected_spots))      # Second deriivative
    inflection_points = np.where(np.diff(np.sign(second_derivative_vector_detected_spots)))[0]  # Finding the inflection points
    try:
        selected_minmass = np.round(tested_intensities [inflection_points[0]],0)
        optimization_worked= True
    except:
        #inflection_points=[0]
        #selected_minmass = np.round(tested_intensities [inflection_points[0]],0)
        optimization_worked = False
    
    # Tracking after finding the best threshold
    try:
        f = tp.batch(img, diameter=particle_diameter,minmass=selected_minmass)
        linked = tp.link_df(f, max_distance_movement) # Linking trajectories
        tracking_df = tp.filter_stubs(linked, min_trajectory_length) # Filtering with minimum length
    except:
        if optimization_worked == True:
            # Backup if the select threshold is too low.
            selected_minmass = np.round(tested_intensities [inflection_points[1]],0)
            f = tp.batch(img, diameter=particle_diameter,minmass=selected_minmass)
            linked = tp.link_df(f, max_distance_movement) # Linking trajectories
            tracking_df = tp.filter_stubs(linked, min_trajectory_length) # Filtering with minimum length
        else:
            tracking_df=[]
            optimization_worked = False
        
    if optimization_worked ==True:
        ####### Converting TracPy dataframe to Croparray format #######
        # Renaming columns names
        tracking_df['z']= 0  
        tracking_df.rename(columns={'x': 'xc','y': 'yc', 'z': 'zc', 'frame': 'f','particle':'id','mass':'MEAN_INTENSITY'}, inplace=True, errors='raise')
        # Chaning data type
        spots = tracking_df.astype({'zc': int,'yc': int,'xc': int,'f': int,'id': int,'MEAN_INTENSITY': int})
        spots['fov']= 0  
        # Selecting some columns
        spots=spots[['fov','id','f','zc','yc','xc','MEAN_INTENSITY']]
        # From trackpy ids are not in order nor consecutive. This code replaces these values and make them ordered consecutive numbers.
        unique_spots_id = spots.id.unique() # unique spots ids
        # Replacing spots with id number.
        for i,id_spot in enumerate(unique_spots_id):
            spots.loc[spots.id == id_spot,'id']=- i # To avoid replacing and mixing different numbers. I am making the new id a negative number.
        spots['id'] = spots['id'].abs() # now getting the absolute value.
        print('Detected trajectories: ',np.max(spots.id)+1 )
        # Plotting    
        if show_plots==True:        
            plt.figure(figsize =(5,5))
            plt.plot(smooth_vector_detected_spots/np.max(smooth_vector_detected_spots) , label='norm detected_spots',linewidth=5,color='lime')
            plt.plot(second_derivative_vector_detected_spots / np.max(second_derivative_vector_detected_spots), label=r"$f''(spots)$",color='orangered',linewidth=5)
            for i, infl in enumerate(inflection_points, 1):
                plt.plot(infl,0, 'o',label='Inflection Point '+str(i), markersize=20, markerfacecolor='cyan')
            plt.legend(bbox_to_anchor=(1.55, 1.0))
            plt.ylim(-0.2,1.1)
            plt.xlabel('Threshold index', size=16)
            plt.ylabel('Norm. number of spots', size=16)
            plt.show()
            
            def plotting_spots(img_2D,dataframe):
                n_particles = dataframe['id'].nunique()
                NUM_ROWS = 1
                NUM_COLUMNS = 3
                index_video = 0
                title_str = 'Video'
                individual_figure_size = 7
                gs = gridspec.GridSpec(NUM_ROWS, NUM_COLUMNS)
                gs.update(wspace = 0.01, hspace = 0.1) # set the spacing between axes.
                # Figure with raw video
                fig = plt.figure(figsize = (individual_figure_size*NUM_COLUMNS, individual_figure_size*NUM_ROWS))
                ax = fig.add_subplot(gs[index_video])
                ax.imshow(img_2D,cmap='gray') 
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title = title_str + ' Original')
                # Figure with filtered video
                ax = fig.add_subplot(gs[index_video+1])
                ax.imshow(img_2D,cmap='gray_r') 
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title = title_str + ' Reverse color' )
                # Figure with filtered video and marking the spots
                ax = fig.add_subplot(gs[index_video+2])
                ax.imshow(img_2D,cmap='gray_r') 
                # Plots the detected spots.
                for k in range (0, n_particles):
                    frames_part = dataframe.loc[dataframe['id'] == dataframe['id'].unique()[k]].f.values
                    if selected_time in frames_part: # plotting the circles for each detected particle at a given time point
                        index_val = np.where(frames_part == selected_time)
                        x_pos = int(dataframe.loc[dataframe['id'] == dataframe['id'].unique()[k]].xc.values[index_val])
                        y_pos = int(dataframe.loc[dataframe['id'] == dataframe['id'].unique()[k]].yc.values[index_val])
                        circle = plt.Circle((x_pos, y_pos), particle_diameter//4, color = 'red', fill = False)
                        ax.add_artist(circle)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set(title = title_str + ' + Detected Spots' ) 
                plt.show()
            # Plotting spots and image
            selected_time =0
            plotting_spots(img_2D=img[selected_time,...],dataframe=spots)
    else:
        print('No tracking was possible with the list of given parameters.')
        spots =[]
    return spots 

# Detecting particles for each frame w/o tracking
def detecting_spots(img,particle_diameter=5, num_iterations = 100,show_plots=True):
    """
    Creates a 

    Parameters
    ----------
    img: NumPy array
        A 3D NumPy array with intensity information from a tif video. The dimensions of the NumPy array must be ordered (f, y, x), where f = frame, y = lateral y-coordinate, and x = lateral x-coordinate.  
    particle_diameter: int
        Particle size in pixels. This has to be an odd integer.
    max_distance_movement: int, optional
        The maximum distance that a particle can move between frames.
    min_trajectory_length: int, optional
        Minimum frames in a trajectory.
    num_iterations: int, optional 
        The number of iterations is used to automatically calculate an intensity threshold.
    show_plots: bool, optional 
        Flag to indicate if the plot with the optimization process is returned.
        
    Returns
    ---------
    A dataset with the following fields 'fov','id','f','zc','yc','xc','MEAN_INTENSITY'

    """    
    num_spots = []
    number_time_points = img.shape[0]
    if number_time_points >20:
        smothing_window = 5
    else:
        smothing_window = 2
    smothing_window = 10
    tested_intensities = np.round(np.linspace(200,10000, num_iterations ),0)
    for i, int_tested in enumerate (tested_intensities):
        try:
            f = tp.locate(img[0,:,:], diameter=particle_diameter, minmass = int_tested )
            num_spots.append(len(f))
        except:
            num_spots.append(0)
    num_spots =np.array(num_spots)   
    
    # Optimization process for selecting intensity
    vector_detected_spots = num_spots/ np.max(num_spots)
    smooth_vector_detected_spots = gaussian_filter1d(vector_detected_spots, smothing_window)
    second_derivative_vector_detected_spots = np.gradient(np.gradient(smooth_vector_detected_spots))      # Second deriivative
    inflection_points = np.where(np.diff(np.sign(second_derivative_vector_detected_spots)))[0]  # Finding the inflection points
    try:
        selected_minmass = np.round(tested_intensities [inflection_points[0]],0)
        optimization_worked= True
    except:
        #inflection_points=[0]
        #selected_minmass = np.round(tested_intensities [inflection_points[0]],0)
        optimization_worked = False
    
    # Tracking after finding the best threshold
    try:
        tracking_df = tp.batch(img, diameter=particle_diameter,minmass=selected_minmass)
        #linked = tp.link_df(f, max_distance_movement) # Linking trajectories
        #tracking_df = tp.filter_stubs(linked, min_trajectory_length) # Filtering with minimum length
    except:
        if optimization_worked == True:
            # Backup if the select threshold is too low.
            selected_minmass = np.round(tested_intensities [inflection_points[1]],0)
            tracking_df = tp.batch(img, diameter=particle_diameter,minmass=selected_minmass)
            #linked = tp.link_df(f, max_distance_movement) # Linking trajectories
            #tracking_df = tp.filter_stubs(linked, min_trajectory_length) # Filtering with minimum length

        else:
            tracking_df=[]
            optimization_worked = False
        
    if optimization_worked ==True:
        ####### Converting TracPy dataframe to Croparray format #######
        # Renaming columns names
        tracking_df['z']= 0  
        tracking_df.rename(columns={'x': 'xc','y': 'yc', 'z': 'zc', 'frame': 'f','mass':'MEAN_INTENSITY'}, inplace=True, errors='raise')
        tracking_df['n']=tracking_df.index
        # Chaning data type
        spots = tracking_df.astype({'zc': int,'yc': int,'xc': int,'f': int,'n': int,'MEAN_INTENSITY': int})
        spots['fov']= 0  
        # Selecting some columns
        spots=spots[['fov','n','f','zc','yc','xc','MEAN_INTENSITY']]
        # # From trackpy ids are not in order nor consecutive. This code replaces these values and make them ordered consecutive numbers.
        # unique_spots_id = spots.id.unique() # unique spots ids
        # # Replacing spots with id number.
        # for i,id_spot in enumerate(unique_spots_id):
        #     spots.loc[spots.id == id_spot,'id']=- i # To avoid replacing and mixing different numbers. I am making the new id a negative number.
        # spots['id'] = spots['id'].abs() # now getting the absolute value.
        print('Detected spots: ',len(spots.n))
        # Plotting    
        if show_plots==True:        
            plt.figure(figsize =(5,5))
            plt.plot(smooth_vector_detected_spots/np.max(smooth_vector_detected_spots) , label='norm detected_spots',linewidth=5,color='lime')
            plt.plot(second_derivative_vector_detected_spots / np.max(second_derivative_vector_detected_spots), label=r"$f''(spots)$",color='orangered',linewidth=5)
            for i, infl in enumerate(inflection_points, 1):
                plt.plot(infl,0, 'o',label='Inflection Point '+str(i), markersize=20, markerfacecolor='cyan')
            plt.legend(bbox_to_anchor=(1.55, 1.0))
            plt.ylim(-0.2,1.1)
            plt.xlabel('Threshold index', size=16)
            plt.ylabel('Norm. number of spots', size=16)
            plt.show()
            #plotting_spots(img_2D=img[selected_time,...],dataframe=spots[spots['f']==selected_time])
            plt.figure(figsize =(5,5))
            df_temp = spots[spots['f']==0]
            tp.annotate(df_temp.rename(columns={'xc':'x','yc':'y','f':'frame'}), img[0])
            plt.show()
    else:
        print('No detection was possible with the list of given parameters.')
        spots =[]
    return spots 