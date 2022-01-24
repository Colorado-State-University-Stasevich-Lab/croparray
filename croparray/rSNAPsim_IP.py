# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:10:24 2020

@author: luisaguilera
"""

# https://stackoverflow.com/questions/18418664/python-inherit-reuse-specific-methods-from-classes
# https://stackoverflow.com/questions/32284568/how-to-create-multiple-class-with-same-function


# THIS IS THE LIST OF LIBRARIES NEEDED TO RUN THE CODE. 
# Please make sure that you have them installed in your computer, in case you need to install some of them:
# open your terminal and type:
# pip install "name_of_library",
# For example, if you want to install bqplot use the following line:  pip install bqplot

#import os

# Plotting
#import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.path as mpltPath

# To manipulate arrays
import numpy as np  
from numpy import unravel_index

# For data frames
import pandas as pd   
#from pandas import DataFrame, Series

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# To read .tiff files
#from skimage import io, color, restoration, img_as_float
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_minimum
#from skimage import feature
#from skimage import measure
from skimage.morphology import binary_closing
from skimage.measure import find_contours
from skimage.draw import polygon2mask
from skimage.draw import polygon
from skimage.util import random_noise

# Matrix transformation and homography calculation
from skimage import transform


try:
    import cv2
except:
    print("Error importing cv2, the manual selection for the mask will not work.")



# Particle tracking
import trackpy as tp 
tp.quiet()  # Turn off progress reports for best performance


# To create interactive elements
import ipywidgets as widgets 
from ipywidgets import interactive, HBox, Layout, GridspecLayout #, interact, fixed, interact_manual, Button, VBox

import bqplot as bq
from bqplot import LinearScale, ColorScale, HeatMap

#import seaborn as sns  # for nicer graphics
import scipy.stats as sps
#from scipy import ndimage
from scipy.signal import find_peaks #, peak_prominences, find_peaks_cwt

#import statistics
from statistics import median_low

import pyfiglet
print( pyfiglet.figlet_format("rSNAPsim IP", font = "slant" ) )

from scipy.spatial import Delaunay
from skimage.draw import polygon_perimeter
import random
import math


### CLASSES

class BeadsAlignment():
    """
      This class allows the user to align the 2 cameras.
        """
    def __init__(self, im_beads):
        self.im_beads = im_beads
    
    def make_beads_alignment(self): 
        # Bandpass filter for the beads function
        lshort = 1 # low pass filter threshold
        llong = 71 # high pass filter threshold
        height, width = self.im_beads.shape[1],self.im_beads.shape[2]
        self.im_beads[0,:height,:width]= tp.bandpass(self.im_beads[0,:height,:width], lshort, llong, threshold=1, truncate=4) # Red channel
        self.im_beads[1,:height,:width]= tp.bandpass(self.im_beads[1,:height,:width], lshort, llong, threshold=1, truncate=4) # Green channel
        # Locating beads in the image using "tp.locate" function from trackpy.
        spot_size = 5
        minIntensity = 400
        f_red = tp.locate(self.im_beads[0,:height,:width],spot_size, minIntensity,maxsize=7,percentile=60) # data frame for the red channel
        f_green = tp.locate(self.im_beads[1,:height,:width],spot_size, minIntensity,maxsize=7,percentile=60)  # data frame for the green channel
        # Converting coordenates to float32 array for the red channel
        x_coord_red = np.array(f_red.x.values, np.float32)
        y_coord_red = np.array(f_red.y.values, np.float32)
        positions_red = np.column_stack((x_coord_red,y_coord_red ))
        # Converting coordenates to float32 array for the green channel
        x_coord_green = np.array(f_green.x.values, np.float32)
        y_coord_green = np.array(f_green.y.values, np.float32)
        positions_green = np.column_stack(( x_coord_green,y_coord_green ))
        # First step to remove of unmatched spots. Comparing Red versus Green channel.
        comparison_red = np.zeros((positions_red.shape[0]))
        comparison_green = np.zeros((positions_green.shape[0]))
        min_distance=4
        for i in range (0, positions_red.shape[0]):
            idx = np.argmin(abs((positions_green[:,0] - positions_red[i,0])))
            comparison_red[i] = (abs(positions_green[idx,0] - positions_red[i,0]) <min_distance) & (abs(positions_green [idx,1] - positions_red[i,1]) <min_distance)
        for i in range (0, positions_green.shape[0]):
            idx = np.argmin(abs((positions_red[:,0] - positions_green[i,0])))
            comparison_green[i] = (abs(positions_red[idx,0] - positions_green[i,0]) <min_distance) & (abs(positions_red [idx,1] - positions_green[i,1]) <min_distance)
        positions_red = np.delete(positions_red, np.where( comparison_red ==0)[0], 0)
        positions_green = np.delete(positions_green, np.where(comparison_green ==0)[0], 0)
        # Second step to remove of unmatched spots. Comparing Green versus Red channel.
        comparison_red = np.zeros((positions_red.shape[0]))
        comparison_green = np.zeros((positions_green.shape[0]))
        min_distance=4
        for i in range (0, positions_green.shape[0]):
            idx = np.argmin(abs((positions_red[:,0] - positions_green[i,0])))
            comparison_green[i] = (abs(positions_red[idx,0] - positions_green[i,0]) <min_distance) & (abs(positions_red [idx,1] - positions_green[i,1]) <min_distance)
        for i in range (0, positions_red.shape[0]):
            idx = np.argmin(abs((positions_green[:,0] - positions_red[i,0])))
            comparison_red[i] = (abs(positions_green[idx,0] - positions_red[i,0]) <min_distance) & (abs(positions_green [idx,1] - positions_red[i,1]) <min_distance)
        positions_red = np.delete(positions_red, np.where( comparison_red ==0)[0], 0)
        positions_green = np.delete(positions_green, np.where(comparison_green ==0)[0], 0)
        print('The number of spots detected for the red channel are:')
        print(positions_red.shape)
        print('The number of spots detected for the green channel are:')
        print(positions_green.shape)
        # Calculating the minimum value of rows for the alignment
        no_spots_for_alignment = min(positions_red.shape[0],positions_green.shape[0])
        # homography, status = cv2.findHomography(srcPoints, dstPoints)
        #srcPoints – Coordinates of the points in the original plane, a matrix of the type CV_32FC2 or vector<Point2f> .
        #dstPoints – Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or a vector<Point2f> .
        # homography, status = cv2.findHomography(positions_green[:no_spots_for_alignment,:2],positions_red[:no_spots_for_alignment,:2],cv2.RANSAC, 5.0)
        
        homography = transform.ProjectiveTransform()
        src = positions_red[:no_spots_for_alignment,:2]
        dst = positions_green[:no_spots_for_alignment,:2]
        homography.estimate(src, dst)
        #image_to_transform = self.im_beads[0,:height,:width]        
        homography_matrix = homography.params
        
        print('')
        print('The homography matrix is:')
        return [homography_matrix,positions_green,positions_red]
    
    

    
class VideoAlignment():
    """
   This class allows the user to align the complete video accoring to the homography calculated with the BeadsAlignment.
   """
    def __init__(self, video, homography):
        self.video = video
        self.homography = homography
        self.vid_rem_ROI = np.zeros_like(video)   
    def make_video_alignment(self): 
        n_channels = self.video.shape[3]
        n_frames =self.video.shape[0]
        height, width = self.video.shape[1],self.video.shape[2]
        # Applying the alignment transformation to the whole video. Matrix multiplication to align the images from the two cameras.
        for k in range(1,n_channels): # green and blue channels
            for i in range(0,n_frames):
                self.vid_rem_ROI[i,:height,:width,k] = transform.warp(self.video[i,:height,:width,k], self.homography, output_shape=(height, width))
        return self.vid_rem_ROI
    


class Bandpass_filter():
    '''
    Class to produce a apply a bandpass filter to the video.
    '''
    def __init__(self,video,low_pass, high_pass):
        self.video = video
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.video_bp_filtered= np.zeros_like(video)            
    def apply_bp_filter(self):
        time_points, height, width, number_channels   = self.video.shape[0],self.video.shape[1], self.video.shape[2], self.video.shape[3]
        for i in range(0,number_channels):
            for k in range(0,time_points):
                self.video_bp_filtered[k,:height,:width,i]= tp.bandpass(self.video[k,:height,:width,i], self.low_pass, self.high_pass, threshold=1, truncate=4)
        return self.video_bp_filtered


class VideoVisualizer():
    '''
    Class to produce a Cell visualization. To start visualization simply move the time slider.
    '''
    def __init__(self,video,mask_array=0,show_contour=0):
        self.video = video
        self.mask_array = mask_array
        self.show_contour = show_contour
    def make_video_app(self):
        def figure_viewer(drop, time):
            plt.figure(1)
            if drop == 'Ch_1':
                channel =0
                plt.imshow(self.video[time,:,:,channel])
            elif drop == 'Ch_2':
                channel = 1
                plt.imshow(self.video[time,:,:,channel])
            elif drop == 'Ch_3':
                channel =2
                plt.imshow(self.video[time,:,:,channel])
            else :
                im = self.video[time,:,:,:].copy()
                imin, imax = np.min(im), np.max(im); im -= imin; 
                imf = np.array(im,'float32'); 
                imf *= 255./(imax-imin); 
                im = np.asarray(np.round(imf), 'uint8')
                plt.imshow(im)
            if (self.show_contour == 1) and (isinstance(self.mask_array, int) == 0):
                contuour_position = find_contours(self.mask_array[time,:,:], 0.8)
                temp = contuour_position[0][:,1]
                temp2 =contuour_position[0][:,0]
                plt.fill(temp,temp2, facecolor='none', edgecolor='yellow')
            plt.show()    
        options = ['Ch_1', 'Ch_2', 'Ch_3', 'All_Channels']
        interactive_plot = interactive(figure_viewer, drop = widgets.Dropdown(options=options,description='Channel'),time = widgets.IntSlider(min=0,max=self.video.shape[0]-1,step=1,value=0,description='time'))
        controls = HBox(interactive_plot.children[:-1], layout = Layout(flex_flow='row wrap'))
        output = interactive_plot.children[-1]
        return controls, output
                

class MaskManual_draw():
    def __init__(self,video,time_point_selected,selected_channel):
        self.im = video[time_point_selected,:,:,selected_channel]
        self.selected_points = []
        self.fig,ax = plt.subplots()
        self.img = ax.imshow(self.im.copy(),cmap='viridis')
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    def poly_img(self,img,pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(255,0,0),2)
        return img
    def onclick(self, event):
        self.selected_points.append([event.xdata,event.ydata])
        if len(self.selected_points)>1:
            self.fig
            self.img.set_data(self.poly_img(self.im.copy(),self.selected_points))

class MaskManual_createMask():
    def __init__(self,video,mask_object,time_point_selected=0,selected_channel=1,show_plot=1):
        self.video = video
        self.mask_object = mask_object
        self.time_point_selected = time_point_selected
        self.selected_channel = selected_channel
        self.number_channels = video.shape
        self.video_removed_mask= np.zeros_like(video) 
        self.show_plot=show_plot
        
    def make_mask(self):
        time_points, height, width, number_channels = self.video.shape[0],self.video.shape[1], self.video.shape[2], self.video.shape[3]
        array_points_coordenates = np.array([self.mask_object.selected_points],'int')
        mask = cv2.fillPoly(np.zeros(self.video[self.time_point_selected,:,:,self.selected_channel].shape,np.uint8),array_points_coordenates,[1,1,1])
        mask_array = np.zeros((time_points, height, width))
        for i in range(0,number_channels):
            for k in range(0,time_points):
                self.video_removed_mask[k,:height,:width,i] = np.multiply(self.video[k,:height,:width,i], mask)
                mask_array [k,:,:] = mask
                
        if self.show_plot ==1:
        # Plotting
            plt.rcParams["figure.figsize"] = (5,5)
            plt.imshow(self.video_removed_mask[self.time_point_selected,:,:,self.selected_channel], cmap=plt.cm.cividis)
            plt.show()        

        return self.video_removed_mask, mask_array


class MaskAutomatic():  
    """
      This class allows the user to automatically deffine a region of interes.
        """
    def __init__(self, video, time_point_selected=0, selected_channel=0,mask_all_timePoints=0,use_bandpass_fileter= 1,bandpass_min=11,bandpass_max=251, show_plot=1):
        self.video = video
        self.time_point = time_point_selected
        self.selected_channel = selected_channel
        self.video_removed_mask= np.zeros_like(video) 
        self.mask_all_timePoints = mask_all_timePoints
        self.show_plot =show_plot
        self.use_bandpass_fileter = use_bandpass_fileter
        self.bandpass_min = bandpass_min
        self.bandpass_max = bandpass_max
        
        
    def make_mask(self):
        time_points, height, width, number_channels = self.video.shape[0],self.video.shape[1], self.video.shape[2], self.video.shape[3]
        def fun_mask(img_short):
            # Applying a bandpass filter to the 
            if self.use_bandpass_fileter ==1:
                converted_img_short = np.zeros_like(img_short) 
                converted_img_short[:,:]= tp.bandpass(img_short, self.bandpass_min, self.bandpass_max)
            else:
                converted_img_short = img_short.copy()
            entropy_img = entropy(converted_img_short, disk(30)) # disk (size Px) moves through the image and applies the filter
            thresh = threshold_minimum(entropy_img)   #Just gives us a threshold value. 
            binary = entropy_img <=thresh  #let us generate a binary image by separating pixels below and above threshold value.
            inv = np.invert(binary) 
            closed_inv = binary_closing(inv)
            # Detecting the contours in the image
            contours = find_contours(closed_inv, 0.8)
            # Selecting the largest area
            im_shape = closed_inv.shape
            area_size = []
            for i in range(0, len(contours)):
                temp_mask = []
                temp_mask = polygon2mask(im_shape, contours[i])
                temp_area = np.sum(temp_mask)
                area_size.append(temp_area)
            selected_area = area_size.index(max(area_size))
            selected_contour = contours[selected_area]
            # Creating the mask
            mask = polygon2mask(im_shape, selected_contour) 
            return mask, entropy_img, img_short, selected_contour
        if (isinstance(self.time_point, int) == 0) and (len(self.time_point) == 0):
            selected_timePoint = 0
            mask, entropy_img, img_short, selected_contour = fun_mask(self.video[selected_timePoint,:,:,self.selected_channel])
        else:
            mask, entropy_img, img_short, selected_contour = fun_mask(self.video[self.time_point,:,:,self.selected_channel])
        if self.show_plot ==1:
        # Plotting
            fig, ax = plt.subplots(1,3, figsize=(16, 5))
            ax[0].imshow(img_short, cmap=plt.cm.cividis)
            ax[1].imshow(entropy_img, cmap=plt.cm.gray)
            ax[2].imshow(mask, cmap=plt.cm.gray)
            temp = selected_contour[:,1]
            temp2 =selected_contour[:,0]
            ax[2].fill(temp,temp2, facecolor='none', edgecolor='yellow')        
            ax[0].set(title='Original')
            ax[1].set(title='Entropy Filter')
            ax[2].set(title='Mask')
            plt.show()
        mask_array = np.zeros((time_points, height, width))
        
        if self.mask_all_timePoints ==1:
            for k in range(0,time_points):
                mask, entropy_img, img_short,selected_contour = fun_mask(self.video[k,:,:,self.selected_channel])
                for i in range(0,number_channels):
                    self.video_removed_mask[k,:height,:width,i] = np.multiply(self.video[k,:height,:width,i], mask)
                mask_array [k,:,:]=mask
        else:
            for i in range(0,number_channels):
                for k in range(0,time_points):
                    self.video_removed_mask[k,:height,:width,i] = np.multiply(self.video[k,:height,:width,i], mask)
                    mask_array [k,:,:] = mask                    
        return self.video_removed_mask, mask_array
    





class trackpy_Profile():
    """
      This class allows the user to observe the intensity profile for the detected particles.
      """
    def __init__(self, video, mask_array=0, particle_size =5, selected_channel=0,time_point_selected=0,min_intensity =100,n_bins=40,show_plot=1):
        self.video = video
        self.selected_channel = selected_channel
        self.time_point = time_point_selected
        self.particle_size = particle_size # according to the documentation must be an even number 3,5,7,9 etc.
        self.minimal_intensity_for_selection = min_intensity # minimal intensity to detect a particle.
        self.n_bins = n_bins
        self.show_plot = show_plot
        self.mask_array = mask_array
    def make_particle_profile(self):
        time_points  = self.video.shape[0]
        def select_time (self,time_points):
            num_spots = []
            for i in range(0,time_points):
                f = tp.locate(self.video[i,:,:,self.selected_channel], self.particle_size, minmass=self.minimal_intensity_for_selection) 
                temp_n_spots = len(f['mass'])
                num_spots.append([temp_n_spots])
            return num_spots
        # "f" is a pandas data freame that contains the infomation about the detected spots
        if (isinstance(self.time_point, int) == 0) and (len(self.time_point) == 0):
            num_spots = select_time (self,time_points)
            selected_timePoint = num_spots.index(max(num_spots))
            max_spots = max(num_spots)
            print('The used time point is: '+ str(selected_timePoint) + ' with ' + str(max_spots[0]) + ' detected spots.')
        else:
            selected_timePoint = self.time_point
        f = tp.locate(self.video[selected_timePoint,:,:,self.selected_channel], self.particle_size, minmass=self.minimal_intensity_for_selection) 
        if self.show_plot == 1:
            plt.rcParams["figure.figsize"] = (5,5)
            plt.hist(f['mass'], bins=self.n_bins, color='dimgray', edgecolor='darkgray',linewidth=1, density= 1)
            plt.xlabel=('Intensity')
            plt.ylabel=('Count')
            plt.title=('Intensity profile for all detected particles')
            min_val = min(f['mass'])
            max_val = max(f['mass'])
            x = np.linspace(min_val, max_val)
            kde = sps.gaussian_kde(f['mass'])
            kde_values = kde.pdf(x)
            plt.plot(x,kde_values, color='green')
            plt.show()
        
    def make_particle_profile_automatic(self):
        time_points  = self.video.shape[0]
        def find_min_values (self, time_points):
            peaks_min_first = []
            for i in range(0,time_points):
                f = tp.locate(self.video[i,:,:,self.selected_channel], self.particle_size, minmass=self.minimal_intensity_for_selection) 
                min_val = min(f['mass'])
                max_val = max(f['mass'])
                x = np.linspace(min_val, max_val)
                kde = sps.gaussian_kde(f['mass'])
                kde_values = kde.pdf(x)
                peaks_min, _ = find_peaks(-kde_values)
                if len(peaks_min) > 0:
                    peaks_min_values = x[peaks_min[0]]
                    peaks_min_first.append(peaks_min_values)
            return peaks_min_first        
        peaks_min_first = find_min_values (self,time_points)
        if (isinstance(self.time_point, int) == 0) and (len(self.time_point) == 0):
            selected_timePoint = 0
        else:
            selected_timePoint = self.time_point
        if  len(peaks_min_first) == 0:   
            f = tp.locate(self.video[selected_timePoint,:,:,self.selected_channel], self.particle_size, minmass=self.minimal_intensity_for_selection) 
            min_val = min(f['mass'])
            max_val = max(f['mass'])
            x = np.linspace(min_val, max_val)
            kde = sps.gaussian_kde(f['mass'])
            kde_values = kde.pdf(x)
            automatic_selection_intensity = 500
        else:
            automatic_selection_intensity = median_low(peaks_min_first)
        f = tp.locate(self.video[selected_timePoint,:,:,self.selected_channel], self.particle_size, minmass=self.minimal_intensity_for_selection) 
        if self.show_plot == 1:
            plt.rcParams["figure.figsize"] = (5,5)
            plt.hist(f['mass'], bins=self.n_bins, color='dimgray', edgecolor='darkgray',linewidth=1, density= 1)
            plt.xlabel=('Intensity')
            plt.ylabel=('Count')
            plt.title=('Intensity profile for all detected particles')
            min_val = min(f['mass'])
            max_val = max(f['mass'])
            x = np.linspace(min_val, max_val)
            kde = sps.gaussian_kde(f['mass'])
            kde_values = kde.pdf(x)
            plt.plot(x,kde_values, color='green')
            plt.plot([automatic_selection_intensity,automatic_selection_intensity],[0, max(kde_values)], color='red' )
            plt.show()
        #print('The inflection point calculated from all time points is: '+ str(automatic_selection_intensity) + '.')
        return automatic_selection_intensity



    def make_particle_profile_automatic_gradient(self):
        f_init = tp.locate(self.video[0,:,:,self.selected_channel], self.particle_size, minmass=0) 
        max_int_in_video= max(f_init.mass)
        if (isinstance(self.time_point, int) == 0) and (len(self.time_point) == 0):
            selected_timePoint = 0
        else:
            selected_timePoint = self.time_point
        #max_int_in_video = np.max(self.video[selected_timePoint,:,:,self.selected_channel])
        max_int_in_video = max_int_in_video*0.8
        if max_int_in_video > 300:
            num_tested_intensity_values = 100
        else:
            num_tested_intensity_values = 20
        num_detected_particles = np.zeros((num_tested_intensity_values),dtype=np.uint32)
        num_detected_particles_inside_mask = np.zeros((num_tested_intensity_values),dtype=np.uint32)
        vector_intensities = np.linspace(1,max_int_in_video,num_tested_intensity_values).astype(np.uint32)
        distance_to_mask_threshold = 5
        for index_p,test_intensity in enumerate(vector_intensities):
            f_test = tp.locate(self.video[selected_timePoint,:,:,self.selected_channel], self.particle_size, minmass=test_intensity, max_iterations=100) 
            num_detected_particles[index_p] = len(f_test.index)
            if num_detected_particles[index_p]  >0:
                counter_selected_particles = 0
                for npart in range(0, num_detected_particles[index_p] ):
                    y_val = f_test.y[f_test.index[npart]].astype(np.int16) # rows
                    x_val = f_test.x[f_test.index[npart]].astype(np.int16) # columns
                    
                    if len(self.mask_array.shape) == 3: 
                        values_in_sel_column =self.mask_array[selected_timePoint,:,x_val]
                        values_in_sel_row =self.mask_array[selected_timePoint,y_val,:]
                    elif len(self.mask_array.shape) == 2:
                        values_in_sel_column =self.mask_array[:,x_val]
                        values_in_sel_row =self.mask_array[y_val,:]
                    position_changes_columns = np.where(values_in_sel_column[:-1] != values_in_sel_column[1:])[0]
                    
                    position_changes_row = np.where(values_in_sel_row[:-1] != values_in_sel_row[1:])[0]
                    top = position_changes_columns[0]
                    bottom = position_changes_columns [1]
                    left = position_changes_row[0]
                    right =  position_changes_row[1]
                    dl, dr, dt, db = abs(x_val-left), abs(x_val-right), abs(y_val-top), abs(y_val-bottom)
                    min_distance_to_mask = min(dl, dr, dt, db)
                    if min_distance_to_mask > distance_to_mask_threshold:
                        counter_selected_particles+=1
                    num_detected_particles_inside_mask[index_p] = counter_selected_particles
            else:
                num_detected_particles_inside_mask[index_p] = 0
        gradeint_num_detected_particles_inside_mask = np.gradient(num_detected_particles_inside_mask)
        selected_intensity_index = np.argmin(gradeint_num_detected_particles_inside_mask) 
        automatic_selection_intensity = vector_intensities[selected_intensity_index]
        
        f = tp.locate(self.video[selected_timePoint,:,:,self.selected_channel], self.particle_size, minmass=self.minimal_intensity_for_selection) 
        if self.show_plot == 1:
            plt.rcParams["figure.figsize"] = (5,5)
            plt.hist(f['mass'], bins=self.n_bins, color='dimgray', edgecolor='darkgray',linewidth=1, density= 1)
            plt.xlabel=('Intensity')
            plt.ylabel=('Count')
            plt.title=('Intensity profile for all detected particles')
            min_val = min(f['mass'])
            max_val = max(f['mass'])
            x = np.linspace(min_val, max_val)
            kde = sps.gaussian_kde(f['mass'])
            kde_values = kde.pdf(x)
            plt.plot(x,kde_values, color='green')
            plt.plot([automatic_selection_intensity,automatic_selection_intensity],[0, max(kde_values)], color='red' )
            plt.show()
        
        
        
        return automatic_selection_intensity
        
        
        
        
        
class VideoVisualizer_tracking():   
    """
      This class allows the user to visualize the detected particles, by manipulating the intensity and particle size.
      """     
    def __init__(self,video,automatic_selection_intensity,mask_array=0,show_contour=0,particle_size=5):
        self.video = video
        self.automatic_selection_intensity = automatic_selection_intensity
        self.mask_array = mask_array
        self.show_contour = show_contour
        self.particle_size =particle_size
    def make_video_app(self):
        def figure_viewer_tr(time,mass_text, drop_size,drop_channel):
            ch = 0
            f = tp.locate(self.video[time,:,:,ch],drop_size, minmass=mass_text,maxsize=self.particle_size,percentile=60) # "f" is a pandas data freame that contains the infomation about the detected spots
            #tp.annotate(f,self.video[time,:,:,ch]);  # tp.anotate is a trackpy function that displays the image with the detected spots  
            #plt.figure()
            #ax = plt.gca()
            #image_timePoint = self.video[time,:,:,ch]
            #plt.imshow(image_timePoint)
            plt.figure(1)
            ax = plt.gca()
            if drop_channel == 'Ch_1':
                channel =0
                plt.imshow(self.video[time,:,:,channel])
            elif drop_channel == 'Ch_2':
                channel = 1
                plt.imshow(self.video[time,:,:,channel])
            elif drop_channel == 'Ch_3':
                channel =2
                plt.imshow(self.video[time,:,:,channel])
            else :
                im = self.video[time,:,:,:].copy()
                imin, imax = np.min(im), np.max(im); im -= imin; 
                imf = np.array(im,'float32'); 
                imf *= 255./(imax-imin); 
                im = np.asarray(np.round(imf), 'uint8')
                plt.imshow(im)
            
            
            
            x_coord_red = np.array(f.x.values, np.float32)
            y_coord_red = np.array(f.y.values, np.float32)
            positions_spots = np.column_stack((x_coord_red,y_coord_red ))
            for i in range(positions_spots.shape[0]):   
                circle = plt.Circle((positions_spots[i,0], positions_spots[i,1]), drop_size, color='red', fill=False)
                ax.add_artist(circle)
            if (self.show_contour == 1) and (isinstance(self.mask_array, int) == 0):                
                contuour_position = find_contours(self.mask_array[time,:,:], 0.8)
                temp = contuour_position[0][:,1]
                temp2 =contuour_position[0][:,0]
                plt.fill(temp,temp2, facecolor='none', edgecolor='yellow')
            plt.show()
        values_size=[3,5,7,9] # Notice value must be an EVEN number.
        options = ['Ch_1', 'Ch_2', 'Ch_3', 'All_Channels']
        interactive_plot_tr = interactive(figure_viewer_tr,mass_text = widgets.IntText(value=self.automatic_selection_intensity ,min=10,description='min Intensity'),drop_size = widgets.Dropdown(options=values_size,value=self.particle_size,description='Particle Size'),time = widgets.IntSlider(min=0,max=self.video.shape[0]-1,step=1,value=0,description='Time'), drop_channel = widgets.Dropdown(options=options,description='Channel'))
        controls = HBox(interactive_plot_tr.children[:-1], layout = Layout(flex_flow='row wrap'))
        output = interactive_plot_tr.children[-1]
        return controls, output

class trackpy():
    """
      This class allows the user to observe the intensity profile for the detected particles.
      """
    def __init__(self,video, tracking_method = 'minimal_distance', particle_size=5, selected_channel=0,selected_intensity=100, minimal_frames=5, min_time_particle_vanishes=1,max_distance_particle_moves=3):
        self.video = video
        self.selected_channel = selected_channel      # selected channel
        self.particle_size = particle_size            # according to the documentation it must be an even number 3,5,7,9 etc.
        self.selected_intensity = selected_intensity  # minimal intensity to detect a particle.
        self.min_time_particle_vanishes = min_time_particle_vanishes          # 
        self.minimal_frames = minimal_frames          #
        self.max_distance_particle_moves = max_distance_particle_moves
        self.tracking_method = tracking_method
    def extract_particleTrajectories(self):
        # "f" is a pandas data freame that contains the infomation about the detected spots.
        # tp.batch is a trackpy function that detects spots for multiple frames in a video.
        f = tp.batch(self.video[:,:,:,self.selected_channel],self.particle_size, minmass=self.selected_intensity,percentile=70, processes='auto')
        # tp.link_df is a trackpy function that links spots detected in multiple frames, this generates the spots trajectories in time.
        
        if self.tracking_method == 'minimal_distance':
            t = tp.link_df(f,self.max_distance_particle_moves, memory=self.min_time_particle_vanishes) # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish). 
        if self.tracking_method == 'nearest_velocity':
            pred = tp.predict.NearestVelocityPredict()
            t = pred.link_df(f,self.max_distance_particle_moves, memory=self.min_time_particle_vanishes) # tp.link_df(data_frame, min_distance_particle_moves, min_time_particle_vanish). 
        
        #trackpy.filtering.filter_stubs(tracks, threshold=100)
        selected_particles_dataframe = tp.filter_stubs(t, self.minimal_frames)  # selecting trajectories that appear in at least 10 frames.
        # Deffining the number of detected spots.
        #print('')
        #print('The number of selected trajectories is:')
        #print(selected_particles_dataframe['particle'].nunique())
        number_detected_particles = selected_particles_dataframe['particle'].nunique()
        return selected_particles_dataframe, number_detected_particles



class Intensity_disk_donut():
    '''This function is inteded to calculate the intensity in each crop by deffining a disk
    '''
    def __init__(self,video, selected_particles_dataframe):
        self.video = video
        self.selected_particles_dataframe = selected_particles_dataframe
        self.crop_size = 5 # size of the half of the crop
        self.disk_size = 3 # size of the half of the disk
        
    def calculate_intensity(self):
        time_points, number_channels  = self.video.shape[0], self.video.shape[3]
        n_particles = self.selected_particles_dataframe['particle'].nunique()
        arr_disk = np.zeros((n_particles,time_points,number_channels))
        # This function is inteded to calculate the intensity in each crop by deffining a disk and subsequently 
        def disk_donut(test_im,disk_size):
            center_coordenates = int(test_im.shape[0]/2)
            # mean intensity in disk
            image_in_disk = test_im[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1]
            mean_intensity_disk = np.mean(image_in_disk)
            # mean intensity in donut.  The center is set to zeros and then the mean is calculated ignoring the zeros.
            recentered_image_donut = test_im.copy()
            recentered_image_donut[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1] = 0 
            mean_intensity_donut = recentered_image_donut[recentered_image_donut!=0].mean() # mean calculation ignoring zeros
            # substracting background minus center intensity
            spot_intensity_disk_donut = mean_intensity_disk - mean_intensity_donut
            spot_intensity_disk_donut = np.nan_to_num(spot_intensity_disk_donut, nan=0) # removing nan values
            return spot_intensity_disk_donut

        def recenter_image_guide_and_additional(image_guide,image_additional,time,x_pos,y_pos,crop_size):
            # function that recenters the spots
            sel_img = image_guide[time,y_pos-(crop_size):y_pos+(crop_size+1):1,x_pos-(crop_size):x_pos+(crop_size+1):1]
            max_coor = unravel_index(sel_img.argmax(), sel_img.shape)
            # adjusting the x-position
            if max_coor[1]> (crop_size):
                x_pos2 = x_pos + abs(max_coor[1]-crop_size)
            if max_coor[1]< (crop_size):
                x_pos2 = x_pos - abs(max_coor[1]-crop_size) 
            if max_coor[1]== (crop_size):
                x_pos2 = x_pos
            # adjusting the y-position    
            if max_coor[0]> (crop_size):
                y_pos2 = y_pos + abs(max_coor[0]-crop_size)
            if max_coor[0]< (crop_size):
                y_pos2 = y_pos - abs(max_coor[0]-crop_size)   
            if max_coor[0]== (crop_size):
                y_pos2 = y_pos    
            recentered_image_additional = image_additional[time,y_pos2-(crop_size):y_pos2+(crop_size+1):1,x_pos2-(crop_size):x_pos2+(crop_size+1):1]
            return recentered_image_additional
        
        for k in range (0,n_particles):
            for j in range(0,time_points):
                for i in range(0,number_channels):
                    x_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[k]].x.values[0])
                    y_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[k]].y.values[0])
                    recentered_image = recenter_image_guide_and_additional(self.video[:,:,:,0],self.video[:,:,:,i],j,x_pos,y_pos,self.crop_size)
                    arr_disk[k,j,i] = disk_donut(recentered_image,self.disk_size)
        
        fig, ax = plt.subplots(3,1, figsize=(16, 4))        
        for id in range (0,n_particles):
            frames_part =self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[id]].frame.values            
            ax[0].plot(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[id]].frame.values, arr_disk[id,frames_part,0],'r')
            ax[1].plot(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[id]].frame.values, arr_disk[id,frames_part,1],'g')
            ax[2].plot(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[id]].frame.values, arr_disk[id,frames_part,2],'b')        
        ax[0].set(title='Intensity vs Time')
        ax[2].set(xlabel='Time (sec)')
        ax[1].set(ylabel='Intensity (a.u.)')
        ax[0].set_xlim([0, time_points-1])
        ax[1].set_xlim([0, time_points-1])
        ax[2].set_xlim([0, time_points-1])
        plt.show()  
        # Initialize a dataframe
        init_dataFrame = {'Cell_No': [],
            'Spot_No': [],
            'Time_sec': [],
            'Red_Int': [],
            'Green_Int': [],
            'Blue_Int': [],
            'x_position': [],
            'y_position': []}
        DataFrame_particles_intensities = pd.DataFrame(init_dataFrame)
        # Iterate for each spot and save time courses in the data frame
        counter = 1
        for id in range (0,n_particles):
            temporal_frames_vector = np.around(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[id]].frame.values)  # time_(sec)
            temporal_red_vector = arr_disk[id, temporal_frames_vector,0] # red
            temporal_green_vector = arr_disk[id,temporal_frames_vector,1] # green
            temporal_blue_vector = arr_disk[id,temporal_frames_vector,2] # blue
            temporal_spot_number_vector = np.around([counter] * len(temporal_frames_vector))
            temporal_cell_number_vector = np.around([1] * len(temporal_frames_vector))
            temporal_x_position_vector = self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[id]].x.values
            temporal_y_position_vector = self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[id]].y.values
            # Section that append the information for each spots
            temp_data_frame = {'Cell_No': temporal_cell_number_vector,
                'Spot_No': temporal_spot_number_vector,
                'Time_sec': temporal_frames_vector,
                'Red_Int': temporal_red_vector,
                'Green_Int': temporal_green_vector,
                'Blue_Int': temporal_blue_vector,
                'x_position': temporal_x_position_vector,
                'y_position': temporal_y_position_vector}
            counter +=1
            temp_DataFrame = pd.DataFrame(temp_data_frame)
            DataFrame_particles_intensities = DataFrame_particles_intensities.append(temp_DataFrame, ignore_index=True)
            DataFrame_particles_intensities = DataFrame_particles_intensities.astype({"Cell_No": int, "Spot_No": int, "Time_sec": int}) # specify data type as integer for some columns    
        return DataFrame_particles_intensities



class Experimental_ac():
    def __init__(self,selected_particles_dataframe, selected_channel, sampling_rate=1, max_lag_output=100,remove_shotnoise=1, show_plot=1):
        self.df = selected_particles_dataframe
        self.sampling_rate = sampling_rate
        self.max_lag_output = max_lag_output
        self.remove_shotnoise = remove_shotnoise
        self.plotting = show_plot
        self.selected_channel = selected_channel
    def calculate_experimental_ac(self):
        def autocorr(x):
            result = np.correlate(x, x, mode='full')
            return result[result.size // 2:]        
        # Calculating the autocovariance
        n_lags = 500
        n_selected_Particles = np.max(self.df.Spot_No)
        ac_array = np.zeros((n_selected_Particles , n_lags ))
        norm_ac_array = np.zeros((n_selected_Particles , n_lags ))
        counter = 0
        # Calculating Autocorrelation.
        for i in range(1,n_selected_Particles+1):
            if self.selected_channel  ==0:
                intensity_signal = self.df.loc[self.df['Spot_No']==i].Red_Int.values
            if self.selected_channel  ==1:
                intensity_signal = self.df.loc[self.df['Spot_No']==i].Green_Int.values
            if self.selected_channel  ==3:
                intensity_signal = self.df.loc[self.df['Spot_No']==i].Blue_Int.values

            temp_ac = autocorr(intensity_signal)
            size_ac_temp  = len(temp_ac)
            ac_array[counter, 0:size_ac_temp] = autocorr(intensity_signal)
            norm_ac_array[counter, 0:size_ac_temp] = autocorr(intensity_signal)/ float(ac_array[counter, :] .max())
            counter += 1
        # Plotting mean autocovariance
        lag_time = [*range(0, self.max_lag_output, 1)]
        lag_time = [element * self.sampling_rate for element in lag_time]
        mean_ac_data = norm_ac_array.mean(0)
        std_ac_data = norm_ac_array.std(0)
        # normalized autocovariance, removing shot noise
        if self.remove_shotnoise ==1:
            plt_lag = lag_time[1:self.max_lag_output]
            plt_mean = mean_ac_data[1:self.max_lag_output]
            plt_std = std_ac_data[1:self.max_lag_output]
            min_x_val = 1
        else:
            plt_lag = lag_time[0:self.max_lag_output]
            plt_mean = mean_ac_data[0:self.max_lag_output]
            plt_std = std_ac_data[0:self.max_lag_output]
            min_x_val=0
        # Plotting
        if self.plotting == 1:
            plt.rcParams["figure.figsize"] = (5,5)
            plt.figure(1)
            plt.plot(plt_lag, plt_mean , 'og');
            plt.fill_between(plt_lag, plt_mean - plt_std, plt_mean + plt_std, color='gray', alpha=0.2)
            plt.xlim([min_x_val, self.max_lag_output]); # Removing the shot noise
            plt.ylim([-0.1, 1]);
            plt.title='Cross-covariance (G)'
            plt.xlabel='Time lag (sec)'
            plt.ylabel='G(I)'
        return plt_lag, plt_mean, plt_std


class Experimental_distribution():
    def __init__(self,selected_particles_dataframe,selected_channel=0, n_bins=30, scaling_factor=1,time_point_selected=0,show_plot=1):
        self.df = selected_particles_dataframe
        self.n_bins = n_bins
        self.scaling_factor = scaling_factor
        self.time_point_selected = time_point_selected
        self.selected_channel = selected_channel
        self.show_plot = show_plot
    def calculate_experimental_distribution(self): 
        list_exp_int_dist = []
        n_selected_Particles = np.max(self.df.Spot_No)
        for i in range(1,n_selected_Particles+1):
            if self.selected_channel  ==0:
                list_exp_int_dist.append(self.df.loc[self.df['Spot_No']==i].Red_Int.values[self.time_point_selected])
            if self.selected_channel  ==1:
                list_exp_int_dist.append(self.df.loc[self.df['Spot_No']==i].Green_Int.values[self.time_point_selected])
            if self.selected_channel  ==3:
                list_exp_int_dist.append(self.df.loc[self.df['Spot_No']==i].Blue_Int.values[self.time_point_selected])
        exp_int_dist = np.asarray(list_exp_int_dist, dtype=np.float32)
        exp_int_dist[exp_int_dist<0] = 0
        exp_int_dist_ump = np.divide(exp_int_dist,self.scaling_factor)
        ind_exp_int_dist_non_zeros = np.argwhere(exp_int_dist)
        ind_exp_int_dist_ump_non_zeros = np.argwhere(exp_int_dist_ump)        
        
        if self.show_plot ==1:
            # Plotting
            fig, ax = plt.subplots(1,2, figsize=(16, 5))
            ax[0].hist(exp_int_dist[ind_exp_int_dist_non_zeros], bins=self.n_bins, color='dimgray', edgecolor='darkgray',linewidth=1) 
            ax[0].set(xlabel='Intensity a.u.')
            ax[0].set(ylabel='Count')
            ax[0].set(title='Intensity in a.u.');
            
            ax[1].hist(exp_int_dist_ump[ind_exp_int_dist_ump_non_zeros], bins=self.n_bins, color='dimgray', edgecolor='darkgray',linewidth=1)
            ax[1].set(xlabel='Intensity u.m.p.')
            ax[1].set(ylabel='Count')
            ax[1].set(title='Intensity in u.m.p.');
        return exp_int_dist[ind_exp_int_dist_non_zeros], exp_int_dist_ump[ind_exp_int_dist_ump_non_zeros]

class Spot_Classification():
    '''This function is inteded to calculate the intensity in each crop by deffining a disk
    '''
    def __init__(self,video, selected_particles_dataframe,automatic_selection_intensity,particle_size):
        self.video = video
        self.selected_particles_dataframe = selected_particles_dataframe
        self.crop_size = particle_size
        self.disk_size = 3
        self.automatic_selection_intensity = automatic_selection_intensity
        self.particle_size = particle_size

    def calculate_classification(self):
        time_points, number_channels  = self.video.shape[0], self.video.shape[3]
        n_particles = self.selected_particles_dataframe['particle'].nunique()
        # This function is inteded to calculate the intensity in each crop by deffining a disk and subsequently 
        def disk_donut(test_im,disk_size):
            center_coordenates = int(test_im.shape[0]/2)
            # mean intensity in disk
            image_in_disk = test_im[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1]
            mean_intensity_disk = np.mean(image_in_disk)
            # mean intensity in donut.  The center is set to zeros and then the mean is calculated ignoring the zeros.
            recentered_image_donut = test_im.copy()
            recentered_image_donut[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1] = 0 
            mean_intensity_donut = recentered_image_donut[recentered_image_donut!=0].mean() # mean calculation ignoring zeros
            # substracting background minus center intensity
            spot_intensity_disk_donut = mean_intensity_disk - mean_intensity_donut
            spot_intensity_disk_donut = np.nan_to_num(spot_intensity_disk_donut, nan=0) # removing nan values
            
            return spot_intensity_disk_donut
        def recenter_image_guide_and_additional(image_guide,image_additional,time,x_pos,y_pos,crop_size):
            # function that recenters the spots
            sel_img = image_guide[time,y_pos-(crop_size):y_pos+(crop_size+1):1,x_pos-(crop_size):x_pos+(crop_size+1):1]
            max_coor = unravel_index(sel_img.argmax(), sel_img.shape)
            # adjusting the x-position
            if max_coor[1]> (crop_size):
                x_pos2 = x_pos + abs(max_coor[1]-crop_size)
            if max_coor[1]< (crop_size):
                x_pos2 = x_pos - abs(max_coor[1]-crop_size) 
            if max_coor[1]== (crop_size):
                x_pos2 = x_pos
            # adjusting the y-position    
            if max_coor[0]> (crop_size):
                y_pos2 = y_pos + abs(max_coor[0]-crop_size)
            if max_coor[0]< (crop_size):
                y_pos2 = y_pos - abs(max_coor[0]-crop_size)   
            if max_coor[0]== (crop_size):
                y_pos2 = y_pos    
            recentered_image_additional = image_additional[time,y_pos2-(crop_size):y_pos2+(crop_size+1):1,x_pos2-(crop_size):x_pos2+(crop_size+1):1]
            return recentered_image_additional
        # Section to calculate the size of the recentered crop
        temp_frames = self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[0]].frame.values
        temp_x_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[0]].x.values[0])
        temp_y_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[0]].y.values[0])
        temp_recentered_image_for_size = recenter_image_guide_and_additional(self.video[:,:,:,0],self.video[:,:,:,0],temp_frames[0],temp_x_pos,temp_y_pos,self.crop_size)
        size_cropped_image= temp_recentered_image_for_size.shape[0]
        # Prealocating memory
        temp_recentered_image1 = np.zeros((time_points,size_cropped_image,size_cropped_image))
        mean_all_particles_all_ch = np.zeros((number_channels,size_cropped_image,size_cropped_image,n_particles+1))
        
        for k in range (0,n_particles):
            frames_part=self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[k]].frame.values
            for i in range(0,number_channels):
                for j in range(0,time_points):
                    if j < len(frames_part):                        
                        x_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[k]].x.values[j])
                        y_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[k]].y.values[j])
                    else:
                        x_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[k]].x.values[0])
                        y_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[k]].y.values[0])
                    temp_recentered_image1[j,:,:] = recenter_image_guide_and_additional(self.video[:,:,:,0],self.video[:,:,:,i],j,x_pos,y_pos,self.crop_size)
                
                mean_all_particles_all_ch [i,:,:,k] = np.mean(temp_recentered_image1[frames_part[0]:frames_part[-1],:,:], axis = 0)

        # This section uses trackpy to detect particles in the average image. Based on detection, it classifies the spot as red, yellow, purple, white
        min_intensity = self.automatic_selection_intensity # adjust this parameter according to your image
        particle_size = self.particle_size  # adjust this parameter according to your image
        particle_classification = np.zeros((n_particles,number_channels)) # Prealocating memory
        for k in range (0,n_particles):
            for i in range(0,number_channels):
                
                f = tp.locate(mean_all_particles_all_ch [i,:,:,k],particle_size, minmass=min_intensity,percentile=90)
                if f.y.values.size ==1 and max(f.ecc.values)<0.3 and 2 <= max(f.y.values) <= (self.crop_size*2)-3 and 3 <= max(f.x.values) <= (self.crop_size*2)-3:   ## add particle location
                    particle_classification [k,i] = 1
        # Automatic classification 
        # Converting intensity =1 if int>threshold, 0 otherwise
        red_spots = particle_classification[:,0] # Where True, yield x, otherwise yield y.
        green_spots = particle_classification[:,1] # Where True, yield x, otherwise yield y.
        blue_spots = particle_classification[:,2] # Where True, yield x, otherwise yield y.
        
        # Spot classification is based on logical operations in each element of the array. For example, a white spot exist if red_spots(=1) *green_spots (=1) *blue_spots (=1) 
        White = red_spots*green_spots*blue_spots 
        Purple = (red_spots*blue_spots) - White
        Yellow = (red_spots*green_spots) - White
        Red_only = red_spots*(1-green_spots)*(1-blue_spots)
        sel_part = mean_all_particles_all_ch.copy()
        x = np.linspace(0, size_cropped_image-1, size_cropped_image)
        y = np.linspace(0, size_cropped_image-1, size_cropped_image)
        x_sc, y_sc, col_sc = LinearScale(), LinearScale(), ColorScale(scheme='Greys') #gist_yarg Greys Greys
        aspect_ratio = 1
        grid = GridspecLayout(n_particles, 4, height='100', width="80%") # , width='200px'
        for k in range (0,n_particles):
            for i in range(0,number_channels):        
                ht_map = HeatMap(x=x, y=y, color=sel_part[i,:,:,k]/np.max(sel_part[i,:,:,k]),scales={'x': x_sc, 'y': y_sc, 'color': col_sc},layout=Layout(width='10px', height='10px',flex_flow='row',display='flex'))
                fig = bq.Figure(marks=[ht_map], padding_y=0.0,padding_x=0.0,min_aspect_ratio=aspect_ratio, max_aspect_ratio=aspect_ratio,fig_margin=dict(top=0, bottom=0, left=0, right=0))
                grid[k,i] = fig
                grid[k,i].layout.height =  'auto'
                value_auto = 'Reject'
            for i in range(3,4): 
                if White[k] ==1:
                    value_auto = 'White(R,G,B)'
                elif Purple[k] ==1:
                    value_auto = 'Purple(R,B)'
                elif Yellow[k] ==1:
                    value_auto = 'Yellow(R,G)'
                elif Red_only[k] ==1:
                    value_auto = 'Red_only(R)'        
                grid[k,i] = widgets.RadioButtons(options=['Red_only(R)', 'Yellow(R,G)', 'Purple(R,B)','White(R,G,B)', 'Reject'],   value = value_auto, layout={'width': 'max-content'}, description='Spot type:',disabled=False)
                grid[k,i].layout.height = 'auto'
        return grid            
    def calculate_classification_bars(self,grid_classification):
        n_particles = self.selected_particles_dataframe['particle'].nunique()
        # Plotting the classification after manually checking the results.
        # Read all check boxes and classify the spots
        chbox_White = np.zeros((n_particles))
        chbox_Purple = np.zeros((n_particles))
        chbox_Yellow = np.zeros((n_particles))
        chbox_Red_only = np.zeros((n_particles))
        chbox_Rejected = np.zeros((n_particles)) # If an spot is ambiguos, you can simply select Reject and the spot won't be considered.
        for id in range (0,n_particles):
            if grid_classification[id,3].value == 'White(R,G,B)':
                chbox_White[id] = 1
            elif grid_classification[id,3].value == 'Purple(R,B)':
                chbox_Purple[id] = 1
            elif grid_classification[id,3].value == 'Yellow(R,G)':
                chbox_Yellow[id] = 1
            elif grid_classification[id,3].value == 'Red_only(R)':
                chbox_Red_only[id] = 1
            elif grid_classification[id,3].value == 'Reject':
                chbox_Rejected[id] = 1
        # Numbers
        hc_num_White =  sum(chbox_White)
        hc_num_Purple = sum(chbox_Purple)
        hc_num_Yellow = sum(chbox_Yellow)
        hc_num_Red_only = sum(chbox_Red_only)
        hc_num_Rejected = sum(chbox_Rejected)
        # total number of spots
        total_no_spots = hc_num_White + hc_num_Purple + hc_num_Yellow + hc_num_Red_only - hc_num_Rejected
        # Fractions
        hc_frac_White =  hc_num_White/total_no_spots
        hc_frac_Purple = hc_num_Purple/total_no_spots
        hc_frac_Yellow = hc_num_Yellow/total_no_spots
        hc_frac_Red_only = hc_num_Red_only/total_no_spots   
        # Plotting manually classified number of spots
        objects = ('Red_only', 'Yellow', 'Purple', 'White')
        x_pos = np.arange(len(objects))
        number_spots = [hc_num_Red_only,hc_num_Yellow,hc_num_Purple,hc_num_White]
        fractions = [hc_frac_Red_only,hc_frac_Yellow,hc_frac_Purple,hc_frac_White]
        # Plotting
        fig, ax = plt.subplots(1,2, figsize=(16, 5))
        ax[0].bar(x_pos, number_spots, align='center', color= (0.7,0.7,0.7),edgecolor='k',linewidth=1)
        ax[0].set_xticks(x_pos)
        ax[0].set_xticklabels(objects)
        ax[0].set(ylabel='Translation numbers')
        ax[0].set(title='RAN translation number (Hand check)');
        ax[1].bar(x_pos, fractions, align='center',color= (0.7,0.7,0.7),edgecolor='k',linewidth=1)
        ax[1].set_xticks(x_pos)
        ax[1].set_xticklabels(objects)
        ax[1].set(ylabel='Translation fraction')
        ax[1].set(title='RAN translation fraction (Hand check)');
        return hc_num_Red_only, hc_num_Yellow, hc_num_Purple, hc_num_White
        





class VideoVisualizer_crops():
    '''This function is inteded to calculate the intensity in each crop by deffining a disk
    '''
    def __init__(self,video, selected_particles_dataframe):
        self.video = video
        self.selected_particles_dataframe = selected_particles_dataframe
        self.crop_size = 5
        self.disk_size = 3
        self.n_particles = self.selected_particles_dataframe['particle'].nunique()
        frames_vector = np.zeros((self.n_particles, 2))            
        for i in range(0,self.n_particles):
            frames_part = self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[i]].frame.values          
            frames_vector[i,:] = frames_part[0],frames_part[-1]
        self.frames_vector = frames_vector
        
    def make_video_crops(self):        
        # This function is inteded to calculate the intensity in each crop by deffining a disk and subsequently 
        def figure_viewer(track,time):            
            def disk_donut(test_im,disk_size):
                center_coordenates = int(test_im.shape[0]/2)
                # mean intensity in disk
                image_in_disk = test_im[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1]
                mean_intensity_disk = np.mean(image_in_disk)
                # mean intensity in donut.  The center is set to zeros and then the mean is calculated ignoring the zeros.
                recentered_image_donut = test_im.copy()
                recentered_image_donut[center_coordenates-disk_size:center_coordenates+disk_size+1,center_coordenates-disk_size:center_coordenates+disk_size+1] = 0 
                mean_intensity_donut = recentered_image_donut[recentered_image_donut!=0].mean() # mean calculation ignoring zeros
                # substracting background minus center intensity
                spot_intensity_disk_donut = mean_intensity_disk - mean_intensity_donut
                return spot_intensity_disk_donut
                
            def recenter_image_guide_and_additional(image_guide,image_additional,time,x_pos,y_pos,crop_size):
                # function that recenters the spots
                sel_img = image_guide[time,y_pos-(crop_size):y_pos+(crop_size+1):1,x_pos-(crop_size):x_pos+(crop_size+1):1]
                max_coor = unravel_index(sel_img.argmax(), sel_img.shape)
                # adjusting the x-position
                if max_coor[1]> (crop_size):
                    x_pos2 = x_pos + abs(max_coor[1]-crop_size)
                if max_coor[1]< (crop_size):
                    x_pos2 = x_pos - abs(max_coor[1]-crop_size) 
                if max_coor[1]== (crop_size):
                    x_pos2 = x_pos
                # adjusting the y-position    
                if max_coor[0]> (crop_size):
                    y_pos2 = y_pos + abs(max_coor[0]-crop_size)
                if max_coor[0]< (crop_size):
                    y_pos2 = y_pos - abs(max_coor[0]-crop_size)   
                if max_coor[0]== (crop_size):
                    y_pos2 = y_pos    
                recentered_image_additional = image_additional[time,y_pos2-(crop_size):y_pos2+(crop_size+1):1,x_pos2-(crop_size):x_pos2+(crop_size+1):1]
                return recentered_image_additional    
            time_points, number_channels  = self.video.shape[0], self.video.shape[3]
            #n_particles = self.selected_particles_dataframe['particle'].nunique()-1
            temp_frames = self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[0]].frame.values
            temp_x_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[0]].x.values[0])
            temp_y_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[0]].y.values[0])
            temp_recentered_image_for_size = recenter_image_guide_and_additional(self.video[:,:,:,0],self.video[:,:,:,0],temp_frames[0],temp_x_pos,temp_y_pos,self.crop_size)
            size_cropped_image= temp_recentered_image_for_size.shape[0]
            frames_part = self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[track]].frame.values
            red_image  = np.zeros((time_points,size_cropped_image,size_cropped_image))
            green_image  = np.zeros((time_points,size_cropped_image,size_cropped_image))
            blue_image  = np.zeros((time_points,size_cropped_image,size_cropped_image))
            for j in frames_part:  #time_points
                if j < len(frames_part):
                    x_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[track]].x.values[j])
                    y_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[track]].y.values[j])
                else:
                    x_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[track]].x.values[0])
                    y_pos=int(self.selected_particles_dataframe.loc[self.selected_particles_dataframe['particle']==self.selected_particles_dataframe['particle'].unique()[track]].y.values[0])
                red_image[j,:,:] = recenter_image_guide_and_additional(self.video[:,:,:,0],self.video[:,:,:,0],j,x_pos,y_pos,self.crop_size)
                green_image[j,:,:] = recenter_image_guide_and_additional(self.video[:,:,:,0],self.video[:,:,:,1],j,x_pos,y_pos,self.crop_size)
                blue_image[j,:,:] = recenter_image_guide_and_additional(self.video[:,:,:,0],self.video[:,:,:,2],j,x_pos,y_pos,self.crop_size)
            fig, ax = plt.subplots(1,number_channels, figsize=(10, 5))
            for i in range(0,number_channels):
                if i ==0:
                    ax[i].imshow(red_image[time,:,:],origin='bottom',cmap=plt.cm.Greys)
                    ax[i].set_axis_off()
                    ax[i].set(title ='Channel_1 (Red)');
                elif i ==1:
                    ax[i].imshow(green_image[time,:,:],origin='bottom',cmap=plt.cm.Greys)
                    ax[i].set_axis_off()
                    ax[i].set(title ='Channel_2 (Green)');
                else:
                    ax[i].imshow(blue_image[time,:,:],origin='bottom',cmap=plt.cm.Greys)
                    ax[i].set_axis_off()
                    ax[i].set(title ='Channel_3 (Blue)');
            print('For track '+ str(track) + ' a spot is detected between time points : '+ str(int(self.frames_vector[track,0])) + ' and ' + str(int(self.frames_vector[track,1])) )
        interactive_plot = interactive(figure_viewer, track = widgets.IntSlider(min=0,max=self.selected_particles_dataframe['particle'].nunique(),step=1,value=0,description='track'),time = widgets.IntSlider(min=0,max=self.video.shape[0]-1,step=1,value=0,description='time'))
        controls = HBox(interactive_plot.children[:-1], layout = Layout(flex_flow='row wrap'))
        output = interactive_plot.children[-1]
        # display(VBox([controls, output]))
        return controls, output          







class Simulated_cell():
    # this section of the code creates a circle in case a polygon is not passed
    
    def __init__(self, number_spots, n_frames, step_size, diffusion_coeffient =0.01,  polygon_array='circle', image_size=[512,512,1], simulated_trajectories_ch1=[0], minimal_background_outside_cell_ch1=10, minimal_background_inside_cell_ch1=20, size_spot_ch1=7, spot_sigma_ch1=2, add_background_noise_ch1=0, amount_background_noise_ch1=0, add_photobleaching_ch1=0, photobleaching_exp_constant_ch1=0, simulated_trajectories_ch2=[0], minimal_background_outside_cell_ch2=10, minimal_background_inside_cell_ch2=20, size_spot_ch2=7, spot_sigma_ch2=2, add_background_noise_ch2=0, amount_background_noise_ch2=0, add_photobleaching_ch2=0, photobleaching_exp_constant_ch2=0, simulated_trajectories_ch3=[0], minimal_background_outside_cell_ch3=10, minimal_background_inside_cell_ch3=20, size_spot_ch3=7, spot_sigma_ch3=2, add_background_noise_ch3=0, amount_background_noise_ch3=0, add_photobleaching_ch3=0, photobleaching_exp_constant_ch3=0, ignore_ch1=0,ignore_ch2=1, ignore_ch3=1,use_triangulation_ch1=0,use_triangulation_ch2=0,use_triangulation_ch3=0):        
        self.number_spots = number_spots
        self.n_frames = n_frames
        self.step_size  = step_size
        self.diffusion_coeffient = diffusion_coeffient 
        self.polygon_array = polygon_array
        self.image_size = image_size
        self.simulated_trajectories_ch1 = simulated_trajectories_ch1
        self.minimal_background_outside_cell_ch1 = minimal_background_outside_cell_ch1
        self.minimal_background_inside_cell_ch1 = minimal_background_inside_cell_ch1
        self.size_spot_ch1 = size_spot_ch1
        self.spot_sigma_ch1 = spot_sigma_ch1
        self.add_background_noise_ch1 = add_background_noise_ch1
        self.amount_background_noise_ch1 = amount_background_noise_ch1
        self.add_photobleaching_ch1 = add_photobleaching_ch1
        self.photobleaching_exp_constant_ch1 = photobleaching_exp_constant_ch1
        self.simulated_trajectories_ch2 = simulated_trajectories_ch2
        self.minimal_background_outside_cell_ch2 = minimal_background_outside_cell_ch2
        self.minimal_background_inside_cell_ch2 = minimal_background_inside_cell_ch2
        self.size_spot_ch2 = size_spot_ch2
        self.spot_sigma_ch2 = spot_sigma_ch2
        self.add_background_noise_ch2 = add_background_noise_ch2
        self.amount_background_noise_ch2 = amount_background_noise_ch2
        self.add_photobleaching_ch2 = add_photobleaching_ch2
        self.photobleaching_exp_constant_ch2 = photobleaching_exp_constant_ch2
        self.simulated_trajectories_ch3 = simulated_trajectories_ch3
        self.minimal_background_outside_cell_ch3 = minimal_background_outside_cell_ch3
        self.minimal_background_inside_cell_ch3 = minimal_background_inside_cell_ch3
        self.size_spot_ch3 = size_spot_ch3
        self.spot_sigma_ch3 = spot_sigma_ch3
        self.add_background_noise_ch3 = add_background_noise_ch3
        self.amount_background_noise_ch3 = amount_background_noise_ch3
        self.add_photobleaching_ch3 = add_photobleaching_ch3 
        self.photobleaching_exp_constant_ch3 = photobleaching_exp_constant_ch3
        self.ignore_ch1 = ignore_ch1
        self.ignore_ch2 = ignore_ch2
        self.ignore_ch3 = ignore_ch3
        self.use_triangulation_ch1 = use_triangulation_ch1
        self.use_triangulation_ch2 = use_triangulation_ch2
        self.use_triangulation_ch3 = use_triangulation_ch3
        self.n_channels = 3
        self.z_slices =1
        if self.polygon_array == 'circle':
            len_poly = 1000
            radius = 200
            center = [250,250]
            self.polygon_array = np.array([[center[0]+ (radius * np.sin(x)) , center[1]+ (radius*np.cos(x)) ] for x in np.linspace(0,2*np.pi,len_poly)[:-1]])

    def make_simulated_cell (self):
        def calculate_intensity_in_figure(tensor_image,n_frames,number_spots, spot_positions_movement, size_spot,step_size):
            tensor_mean_intensity_in_figure = np.zeros((n_frames,number_spots),dtype='float')
            tensor_std_intensity_in_figure = np.zeros((n_frames,number_spots),dtype='float')
            for t_p in np.arange(0, n_frames,step_size):
                center_positions_vector = spot_positions_movement[t_p,:,:]
                temp_tensor_image = tensor_image[t_p,:,:]
                for point_index in range(0,len(center_positions_vector)):
                    center_position = center_positions_vector[point_index]
                    # Defining the current area in the matrix that will be replaced by the spot kernel.
                    current_area = temp_tensor_image[center_position[0]-int(size_spot/2): center_position[0]+int(size_spot/2)+1 ,center_position[1]-int(size_spot/2): center_position[1]+int(size_spot/2)+1 ]                
                    tensor_mean_intensity_in_figure[t_p,point_index] = np.mean(current_area)
                    tensor_std_intensity_in_figure[t_p,point_index] = np.std(current_area)
            return tensor_mean_intensity_in_figure, tensor_std_intensity_in_figure
        
        def make_replacement_pixelated_spots(matrix_background, center_positions_vector, size_spot, spot_sigma, using_ssa, simulated_trajectories_time_point,max_SSA_value, minimal_background):
            '''
                This funciton is intended to replace a kernel gaussian matrix for each spot position. The kernel gaussian matrix is scaled 
                with the values obtained from the SSA o with the values given in a range.
            '''
            # Section that creates the Gaussian Kernel Matrix
            ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(spot_sigma))
            # Copy the matrix_background
            pixelated_image = matrix_background.copy()
            for point_index in range(0,len(center_positions_vector)):
                # creating a position for each spot
                center_position = center_positions_vector[point_index]
                if using_ssa == 1:
                    # Scaling the spot kernel matrix with the intensity value form the SSA.
                    spot_intensity = (simulated_trajectories_time_point[point_index]/max_SSA_value)*255
                    kernel_value_intensity = kernel*spot_intensity
                    kernel_value_intensity[kernel_value_intensity < minimal_background] = minimal_background # making spots with less intensity than the background equal to the background.
                else:
                    # Scaling the spot kernel matrix with a mean value obtained from uniformly random distribution.
                    min_int = 50
                    max_int = 60
                    spot_intensity = int(random.uniform(min_int,max_int))
                    kernel_value_intensity = kernel*spot_intensity
                    kernel_value_intensity[kernel_value_intensity < minimal_background] = minimal_background # making spots with less intensity than the background equal to the background.
                # Defining the current area in the matrix that will be replaced by the spot kernel.
                current_area = pixelated_image[center_position[0]-int(size_spot/2): center_position[0]+int(size_spot/2)+1 ,center_position[1]-int(size_spot/2): center_position[1]+int(size_spot/2)+1 ]
                # Getting the maximum value between the current area and the new kernel intensity. This sections calculates the maximum intensity in case two spots overlap.
                kernel_intensity_maximum_value = np.maximum(kernel_value_intensity, current_area)            
                # Replacing the sub-area in the image with the new spot kernel matrix  
                pixelated_image[center_position[0]-int(size_spot/2): center_position[0]+int(size_spot/2)+1 ,center_position[1]-int(size_spot/2): center_position[1]+int(size_spot/2)+1 ] = kernel_intensity_maximum_value
            return pixelated_image
        def make_spots_movement(polygon_array, number_spots, n_frames, step_size, image_size, diffusion_coeffient):
            # Section of the code that creates the desired number of spots inside a given polygon
            path = mpltPath.Path(polygon_array)
            initial_points_in_polygon = np.zeros((number_spots,2), dtype = 'int')
            counter_number_spots = 0
            conter_security = 0
            max_iterations = 5000
            min_position = 0 # minimal position in pixels
            max_position = image_size[1] # maximal position in pixels
            while (counter_number_spots < number_spots) and (conter_security<max_iterations):
                test_points = (int(random.uniform(min_position,max_position)), int(random.uniform(min_position,max_position)))
                conter_security +=1
                if path.contains_point(test_points) == 1:
                    counter_number_spots+=1 
                    initial_points_in_polygon[counter_number_spots-1,:] = np.asarray(test_points)
                if conter_security>max_iterations:
                    print('error generating spots')
            ## Brownian motion
            # scaling factor for Brownian motion.
            brownian_movement = math.sqrt(2*diffusion_coeffient*step_size) 
            # Prealocating memory
            x_positions = np.array(initial_points_in_polygon[:,0],dtype='int') #  x_position for selected spots inside the polygon
            y_positions = np.array(initial_points_in_polygon[:,1],dtype='int') #  y_position for selected spots inside the polygon
            temp_Position_x = np.zeros_like(x_positions,dtype='int')
            temp_Position_y = np.zeros_like(y_positions,dtype='int')
            newPosition_x = np.zeros_like(x_positions,dtype='int')
            newPosition_y = np.zeros_like(y_positions,dtype='int')
            spot_positions_movement = np.zeros((n_frames,number_spots,2),dtype='int')
            # Main loop that computes the random motion and new spot positions
            for t_p in np.arange(0, n_frames, step_size):
                for i_p in range (0, number_spots):
                    if t_p == 0:
                        temp_Position_x[i_p]= x_positions[i_p]
                        temp_Position_y[i_p]= y_positions[i_p]            
                    else:
                        temp_Position_x[i_p]= newPosition_x[i_p] + int(brownian_movement * np.random.randn(1))
                        temp_Position_y[i_p]= newPosition_y[i_p] + int(brownian_movement * np.random.randn(1))
                    while path.contains_point((temp_Position_x[i_p], temp_Position_y[i_p])) == 0:
                        temp_Position_x[i_p]= newPosition_x[i_p] 
                        temp_Position_y[i_p]= newPosition_y[i_p] 
                    newPosition_x[i_p]= temp_Position_x[i_p]
                    newPosition_y[i_p]= temp_Position_y[i_p]
                spot_positions_movement [t_p,:,:]= np.vstack((newPosition_x, newPosition_y)).T
            return spot_positions_movement
        # Triangulation and background
        def make_matrix_background(polygon_array, use_triangulation, image_size, minimal_background_outside_cell, minimal_background_inside_cell):

            if use_triangulation == 1:
                # Code that creates the triangulation for the cytoskeleton    
                number_spots_triangulation = 2000
                path = mpltPath.Path(polygon_array)
                points_in_polygon = np.zeros((number_spots_triangulation,2), dtype = 'int')
                counter_number_spots = 0
                conter_security = 0
                max_iterations = number_spots_triangulation*10
                min_position = 0 # minimal position in pixels
                max_position = image_size[1] # maximal position in pixels
                while (counter_number_spots < number_spots_triangulation) and (conter_security<max_iterations):
                    test_points = (int(random.uniform(min_position,max_position)), int(random.uniform(min_position,max_position)))
                    conter_security +=1
                    if path.contains_point(test_points) == 1:
                        counter_number_spots+=1 
                        points_in_polygon[counter_number_spots-1,:] = np.asarray(test_points)
                y = points_in_polygon[:,0]
                x = points_in_polygon[:,1]
                points = np.vstack((x,y)).T
                tri = Delaunay(points)
                # creating the canvas
                matrix_background = np.zeros((image_size[0],image_size[1]),dtype='uint8')   
                # making a mask with the values inside the cell equal to one
                mask = np.zeros((image_size[0],image_size[1]),dtype='uint8')   
                rr, cc = polygon(polygon_array[:,0], polygon_array[:,1])
                mask[rr, cc] = 1
                # adding color intensity inside the cell in the canvas
                rr, cc = polygon(polygon_array[:,0], polygon_array[:,1])
                matrix_background[rr, cc] = minimal_background_inside_cell
                # adding the cytoskeleton
                for i in range(0, len(tri.simplices)):
                    plo_x = points[tri.simplices][i][:,0]
                    plo_y = points[tri.simplices][i][:,1]
                    rr, cc = polygon_perimeter(plo_y,plo_x,shape=matrix_background.shape, clip=True)
                    matrix_background[rr, cc] = minimal_background_inside_cell+10
                # removing cytoskeleton outside the cell
                matrix_background = matrix_background*mask
                matrix_background = np.where(matrix_background != 0, matrix_background, minimal_background_outside_cell)
            else:
                # changing the figure background color from black to gray
                matrix_background = np.zeros((image_size[0],image_size[1]),dtype='uint8')    
                matrix_background[:,:] = minimal_background_outside_cell    
                # adding color intensity inside the cell
                rr, cc = polygon(polygon_array[:,0], polygon_array[:,1])
                matrix_background[rr, cc] = minimal_background_inside_cell    
            return matrix_background
        # This function runs all the other functions
        def make_simulation(spot_positions_movement, n_frames, step_size, polygon_array, image_size, minimal_background_outside_cell, minimal_background_inside_cell, size_spot, spot_sigma, simulated_trajectories, add_background_noise, amount_background_noise, add_photobleaching, photobleaching_exp_constant,use_triangulation):
            matrix_background = make_matrix_background(polygon_array,use_triangulation, image_size, minimal_background_outside_cell, minimal_background_inside_cell)
            tensor_image = np.zeros((n_frames,image_size[0],image_size[1]),dtype='uint8')
            for t_p in np.arange(0, n_frames, step_size):
                if type(simulated_trajectories) == int:
                    using_ssa = 0
                    simulated_trajectories_tp = 0
                    max_SSA_value = 0
                else:
                    using_ssa = 1
                    simulated_trajectories_tp = simulated_trajectories[:,t_p]
                    max_SSA_value = simulated_trajectories.max()
                # Making the pixelated spots    
                tensor_image[t_p,:,:] = make_replacement_pixelated_spots(matrix_background, spot_positions_movement[t_p,:,:], size_spot, spot_sigma, using_ssa, simulated_trajectories_tp,max_SSA_value, minimal_background_inside_cell)
            
                # Adding background noise, notice that this operation normalizes the intensity values between zero and one, then it re-normalizes to 255
                if add_background_noise == 1:
                    temp_tensor_image= random_noise(tensor_image[t_p,:,:], mode='gaussian', mean=amount_background_noise, var=amount_background_noise)
                    tensor_image[t_p,:,:] = np.array(255 * temp_tensor_image, dtype=np.uint8) 
                    
                # Adding  photobleaching
                if add_photobleaching  == 1:
                    photobleaching = int(np.exp(photobleaching_exp_constant*t_p))
                    photobleaching = np.min([photobleaching,255]) # this section makes the maxmum value equal to 255, that is the maximum value allowed on uint8 data type.       
                    temp_tensor_image_pb = tensor_image[t_p,:,:]  + photobleaching # adding the photobleaching
                    temp_tensor_image_pb [temp_tensor_image_pb>255] = int(255) # Making sure that the pixels on the image are never larger than 255, that is the maximum value allowed on uint8 data type.        
                    tensor_image[t_p,:,:] = np.array(temp_tensor_image_pb, dtype=np.uint8) 
            return tensor_image
            
        # Create the spots for all channels. Return array with 3-dimensions: T,Sop, XY-Coord
        spot_positions_movement = make_spots_movement(self.polygon_array, self.number_spots, self.n_frames, self.step_size, self.image_size, self.diffusion_coeffient)
    
        # This section of the code runs the for each channel    
        # Channel 1
        if self.ignore_ch1 == 0:
            tensor_image_ch1 = make_simulation(spot_positions_movement, self.n_frames, self.step_size, self.polygon_array, self.image_size, self.minimal_background_outside_cell_ch1, self.minimal_background_inside_cell_ch1, self.size_spot_ch1, self.spot_sigma_ch1, self.simulated_trajectories_ch1, self.add_background_noise_ch1, self.amount_background_noise_ch1, self.add_photobleaching_ch1, self.photobleaching_exp_constant_ch1, self.use_triangulation_ch1)
            tensor_mean_intensity_in_figure_ch1, tensor_std_intensity_in_figure_ch1 = calculate_intensity_in_figure(tensor_image_ch1,self.n_frames,self.number_spots, spot_positions_movement, self.size_spot_ch1, self.step_size)
        else:
            tensor_image_ch1 = np.zeros((self.n_frames,self.image_size[0],self.image_size[1]),dtype='uint8') 
            tensor_mean_intensity_in_figure_ch1 = np.zeros((self.n_frames,self.number_spots),dtype='int')
            tensor_std_intensity_in_figure_ch1 = np.zeros((self.n_frames,self.number_spots),dtype='int')
        # Channel 2
        if self.ignore_ch2 == 0:
            tensor_image_ch2 = make_simulation(spot_positions_movement, self.n_frames, self.step_size, self.polygon_array, self.image_size, self.minimal_background_outside_cell_ch2, self.minimal_background_inside_cell_ch2, self.size_spot_ch2, self.spot_sigma_ch2, self.simulated_trajectories_ch2, self.add_background_noise_ch2, self.amount_background_noise_ch2, self.add_photobleaching_ch2, self.photobleaching_exp_constant_ch2, self.use_triangulation_ch2)
            tensor_mean_intensity_in_figure_ch2, tensor_std_intensity_in_figure_ch2 = calculate_intensity_in_figure(tensor_image_ch2,self.n_frames,self.number_spots, spot_positions_movement, self.size_spot_ch2, self.step_size)
        else:
            tensor_image_ch2 = np.zeros((self.n_frames,self.image_size[0],self.image_size[1]),dtype='uint8')    
            tensor_mean_intensity_in_figure_ch2 = np.zeros((self.n_frames,self.number_spots),dtype='int')
            tensor_std_intensity_in_figure_ch2 = np.zeros((self.n_frames,self.number_spots),dtype='int')
        # Channel 3
        if self.ignore_ch3 == 0:   
            tensor_image_ch3 = make_simulation(spot_positions_movement, self.n_frames, self.step_size, self.polygon_array, self.image_size, self.minimal_background_outside_cell_ch3, self.minimal_background_inside_cell_ch3, self.size_spot_ch3, self.spot_sigma_ch3, self.simulated_trajectories_ch3, self.add_background_noise_ch3, self.amount_background_noise_ch3, self.add_photobleaching_ch3, self.photobleaching_exp_constant_ch3, self.use_triangulation_ch3)
            tensor_mean_intensity_in_figure_ch3, tensor_std_intensity_in_figure_ch3 = calculate_intensity_in_figure(tensor_image_ch3,self.n_frames,self.number_spots, spot_positions_movement, self.size_spot_ch3, self.step_size)
        else:
            tensor_image_ch3 = np.zeros((self.n_frames,self.image_size[0],self.image_size[1]),dtype='uint8')   
            tensor_mean_intensity_in_figure_ch3 = np.zeros((self.n_frames,self.number_spots),dtype='int')
            tensor_std_intensity_in_figure_ch3 = np.zeros((self.n_frames,self.number_spots),dtype='int')                                                     
        
        # Creating a tensor with the final video as a tensor with 4D the order TXYC
        
        tensor_video =  np.zeros((self.n_frames,self.image_size[0],self.image_size[1], self.n_channels),dtype='uint8')
        tensor_video [:,:,:,0] =  tensor_image_ch1
        tensor_video [:,:,:,1] =  tensor_image_ch2
        tensor_video [:,:,:,2] =  tensor_image_ch3
        
        # This section saves the tensor as a imagej array 5D. In the orderd :  TZCYX    
        
        tensor_for_image_j = np.zeros((self.n_frames, self.z_slices, self.n_channels, self.image_size[0], self.image_size[1]),dtype='uint8')
        for i in range(0, self.n_frames):
            for ch in range(0, 3):
                if ch ==0:
                    tensor_for_image_j [i,0,0,:,:] = tensor_image_ch1 [i,:,:]
                if ch ==1:
                    tensor_for_image_j [i,0,1,:,:] = tensor_image_ch2 [i,:,:]
                if ch ==2:
                    tensor_for_image_j [i,0,2,:,:] = tensor_image_ch3 [i,:,:]
        # Creating tensors with real intensity values in the order TSC
        tensor_mean_intensity_in_figure = np.zeros((self.n_frames,self.number_spots, self.n_channels),dtype='float')
        tensor_mean_intensity_in_figure[:,:,0] = tensor_mean_intensity_in_figure_ch1
        tensor_mean_intensity_in_figure[:,:,1] = tensor_mean_intensity_in_figure_ch2
        tensor_mean_intensity_in_figure[:,:,2] = tensor_mean_intensity_in_figure_ch3
        # The same for the std    
        tensor_std_intensity_in_figure = np.zeros((self.n_frames,self.number_spots, self.n_channels),dtype='float')
        tensor_std_intensity_in_figure[:,:,0] = tensor_std_intensity_in_figure_ch1
        tensor_std_intensity_in_figure[:,:,1] = tensor_std_intensity_in_figure_ch2
        tensor_std_intensity_in_figure[:,:,2] = tensor_std_intensity_in_figure_ch3
        
        return tensor_video, tensor_for_image_j, spot_positions_movement, tensor_mean_intensity_in_figure, tensor_std_intensity_in_figure

