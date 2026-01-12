import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

def best_z_proj(ca, **kwargs):
    '''
    Returns an x-array that holds the best-z projection of intensities of spots in a reference channel and augments ca to include a 'ca.zc' layer that holds the best-z values.

    Parameters
    ----------
    ca: crop array (x-array dataset)
        A crop array.
    ref_ch: int, optional 
        A reference intensity channel for finding the best-z projection. Default: ref_ch = 0. If None, best-z is calculated separately for each channel.
    disk_r: int, optional
        The radius of the disk (in pixels) to make measurements to determine the best-z slice. Default: disk_r = 1
    roll_num: int, optional
        The number of z-slices to use in the rolling-z max projection for determining the best-z slice. min_periods in the da.rolling() function is set to 1 so there will be no Nans at the z-edges of crops. Default: roll_num = 1.

    Returns
    ------- 
    A 'best-z' x-array with dimensions (fov,n,t,y,x,ch). The best-z x-array contains the intensities of each crop in the 'best' z-slice, where 'best' is defined as the slice having the maximal intensity within a centered xy disk of radius disk_r pixels. A rolling-z maximum projection (over roll_n z-slices) can optionally be performed so best-z represents a max-z projection across multiple z-slices. In addition, the inputted crop array ca is augmented to now contain a 'zc' layer that contains the best-z position of each crop. 
    '''
    # Get the optional key word arguments (kwargs):
    ref_ch = kwargs.get('ref_ch', 0)
    disk_r = kwargs.get('disk_r', 1) 
    roll_n = kwargs.get('roll_n', 1)

    res = ca.dx  # resolution for defining disk to make measurements to determine best-z

    # Compute best-z separately for each channel using list comprehension
    if ref_ch is None:
        z_sig = [
            ca.sel(ch=ch_index).int.where(lambda a: a.x**2 + a.y**2 <= (disk_r * res)**2)
            .mean(dim=['x', 'y'])
            .rolling(z=roll_n, center=True, min_periods=1)
            .max()
            for ch_index in ca.ch.values
        ]
        # Choose z-plane
        output = xr.concat([
            ca.int.isel(ch=i).rolling(z=roll_n, center=True, min_periods=1).max().isel(z=z_sig[i].argmax(dim='z'))
            for i in range(len(ca.ch.values))
        ], dim='ch')   
        
        # Add/overwrite the 'zc' layer in the inputted crop-array
        ca['zc'] = xr.concat([z_sig[i].argmax(dim='z') for i in range(len(ca.ch.values))], dim='ch')
        ca.zc.attrs['units'] = 'pixels'
        ca.zc.attrs['long_name'] = 'crop center z for each channel'
    else:
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
    ca: crop array (x-array dataset)
        A crop array.
    ref_ch: int, optional 
        A reference intensity channel for finding the best-z plane for measurements. Default: None (uses all channels).    
    disk_r: int, optional
        The radius (in pixels) within which the intensity signal for each crop is measured. Default: disk_r = 1
    disk_bg: int, optional
        The radius (in pixels) of an outer ring (of width one pixel) within which the background signal for each crop is measured. Default: disk_bg = ca.xy_pad.
    roll_num: int, optional
        The number of z-slices to use in the rolling-z max projection for determining the best-z slices to perform intensity measurements. Default: roll_num = 1.

    Returns
    ------- 
    An augmented crop array ca with two additional variables: (1) ca.best_z is an x-array with dimensions (fov,n,t,y,x,ch) that contains the best-z-projection after background subtraction; (2) ca.signal is an x-array with dimensions (fov,n,t,ch) that contains the background-subtracted intensity signal of each crop in ca.best_z. 
    '''
    # Get the optional keyword arguments (kwargs):
    my_ref_ch = kwargs.get('ref_ch', None)
    my_disk_r = kwargs.get('disk_r', 1) 
    my_disk_bg = kwargs.get('disk_bg', ca.xy_pad) 
    my_roll_n = kwargs.get('roll_n', 1)

    # Create best-z projection (if not already)
    best_z = best_z_proj(ca, ref_ch=my_ref_ch, disk_r=my_disk_r, roll_n=my_roll_n)
    
    # Make mask for measuring within inner ring (the disk):
    disk_sig = best_z.where(lambda a: a.x**2 + a.y**2 <= (my_disk_r * ca.dx) ** 2).mean(dim=['x', 'y'])
    
    # Make mask for measuring background within outer ring (the donut):
    donut_sig = best_z.where(lambda a: (a.x**2 + a.y**2 >= (my_disk_bg * ca.dx) ** 2) & (a.x**2 + a.y**2 < ((my_disk_bg + 1) * ca.dx) ** 2)).median(dim=['x', 'y'])

    # Measure signal as disk - donut: 
    signal = disk_sig - donut_sig 

    # Add best_z variable to ca
    ca['best_z'] = best_z - donut_sig
    ca['best_z'].attrs['units'] = 'intensity (a.u.)'
    ca['best_z'].attrs['long_name'] = 'max intensity projection into best-z plane(s)'

    # Add best_z_signal variable to ca:
    ca['signal'] = signal
    ca['signal'].attrs['units'] = 'intensity (a.u.)'
    ca['signal'].attrs['long_name'] = 'crop signal'

    return ca

def measure_signal_raw(ca, **kwargs):
    '''
    A function to measure and visualize the intensity signal of all crops in the crop array ca.

    Parameters
    ----------
    ca: crop array (x-array dataset)
        A crop array.
    ref_ch: int, optional 
        A reference intensity channel for finding the best-z plane for measurements. Default: None (uses all channels).    
    disk_r: int, optional
        The radius (in pixels) within which the intensity signal for each crop is measured. Default: disk_r = 1
    disk_bg: int, optional
        The radius (in pixels) of an outer ring (of width one pixel) within which the background signal for each crop is measured. Default: disk_bg = ca.xy_pad.
    roll_num: int, optional
        The number of z-slices to use in the rolling-z max projection for determining the best-z slices to perform intensity measurements. Default: roll_num = 1.

    Returns
    ------- 
    An augmented crop array ca with two additional variables: (1) ca.best_z_raw is an x-array with dimensions (fov,n,t,y,x,ch) that contains the best-z-projection; (2) ca.signal_raw is an x-array with dimensions (fov,n,t,ch) that contains the intensity signal of each crop in ca.best_z_raw. 
    '''
    # Get the optional keyword arguments (kwargs):
    my_ref_ch = kwargs.get('ref_ch', None)
    my_disk_r = kwargs.get('disk_r', 1) 
    my_roll_n = kwargs.get('roll_n', 1)

    # Create best-z projection (if not already)
    best_z = best_z_proj(ca, ref_ch=my_ref_ch, disk_r=my_disk_r, roll_n=my_roll_n)
    
    # Make mask for measuring within inner ring (the disk):
    disk_sig = best_z.where(lambda a: a.x**2 + a.y**2 <= (my_disk_r) ** 2).sum(dim=['x', 'y'])

    # Add best_z variable to ca
    ca['best_z_raw'] = best_z
    ca['best_z_raw'].attrs['units'] = 'intensity (a.u.)'
    ca['best_z_raw'].attrs['long_name'] = 'max intensity projection into best-z plane(s)'

    # Add best_z_signal variable to ca:
    ca['signal_raw'] = disk_sig
    ca['signal_raw'].attrs['units'] = 'intensity (a.u.)'
    ca['signal_raw'].attrs['long_name'] = 'crop signal'

    return ca