#*
#    SLAM.py: the implementation of SLAM
#    created and maintained by Ty Nguyen
#    tynguyen@seas.upenn.edu
#    Feb 2020
#*
import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import logging
if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle

import pdb

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
 

class SLAM(object):
    def __init__(self):
        self._characterize_sensor_specs()
    
    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_= str(dataset)
        if split_name.lower() not in src_dir:
            src_dir  = src_dir + '/' + split_name
        print('\n------Reading Lidar and Joints (IMU)------')
        self.lidar_  = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_lidar'+ self.dataset_)

        print ('\n------Reading Joints Data------')
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_joint'+ self.dataset_)

        self.num_data_ = len(self.lidar_.data_)
        # Position of odometry
        self.odo_indices_ = np.empty((2,self.num_data_),dtype=np.int64)

    def _characterize_sensor_specs(self, p_thresh=None):
        # High of the lidar from the ground (meters)
        self.h_lidar_ = 0.93 + 0.33 + 0.15
        # Accuracy of the lidar
        self.p_true_ = 9
        self.p_false_ = 1.0/9
        
        #TODO: set a threshold value of probability to consider a map's cell occupied  
        self.p_thresh_ = 0.6 if p_thresh is None else p_thresh # > p_thresh => occupied and vice versa
        # Compute the corresponding threshold value of logodd
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(self.p_thresh_)
        

    def _init_particles(self, num_p=0, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        # Particles representation
        self.num_p_ = num_p
        #self.percent_eff_p_thresh_ = percent_eff_p_thresh
        self.particles_ = np.zeros((3,self.num_p_),dtype=np.float64) if particles is None else particles
        
        # Weights for particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        self.best_p_indices_ = np.empty((2,self.num_data_),dtype=np.int64)
        # Best particles
        self.best_p_ = np.empty((3,self.num_data_))
        # Corresponding time stamps of best particles
        self.time_ =  np.empty(self.num_data_)
       
        # Covariance matrix of the movement model
        tiny_mov_cov   = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        self.mov_cov_  = mov_cov if mov_cov is not None else tiny_mov_cov
        # To generate random noise: x, y, z = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).T
        # this return [x], [y], [z]

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh

    def _init_map(self, map_resolution=0.05):
        '''*Input: resolution of the map - distance between two grid cells (meters)'''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -20  #meters
        MAP['ymin']  = -20
        MAP['xmax']  =  20
        MAP['ymax']  =  20
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)


    def _build_first_map(self,t0=0,use_lidar_yaw=True):
        """Build the first map using first lidar"""
        self.t0 = t0
        # Extract a ray from lidar data
        MAP = self.MAP_
        print('\n--------Doing build the first map--------')

        #TODO: student's input from here 



        #End student's input 

        self.MAP_ = MAP



    def _predict(self,t,use_lidar_yaw=True):
        logging.debug('\n-------- Doing prediction at t = {0}------'.format(t))
        #TODO: student's input from here 



        #End student's input 

    def _update(self,t,t0=0,fig='on'):
        """Update function where we update the """
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return

        #TODO: student's input from here 



        #End student's input 

        self.MAP_ = MAP
        return MAP
