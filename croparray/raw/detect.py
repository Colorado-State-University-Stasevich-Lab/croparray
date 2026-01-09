from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Optional plotting imports can stay local inside the function.
# Trackpy import MUST be local inside the function to keep it optional.
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
    import trackypy as tp
    if show_plots:
        import matplotlib.pyplot as plt
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