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
    import trackypy as tp
    num_spots = []
    number_time_points = img.shape[0]
    if number_time_points >20:
        smothing_window = 5
    else:
        smothing_window = 2
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
