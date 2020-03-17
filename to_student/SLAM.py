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
from bresenham2D import *

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
        self.p_true_ = 0.9
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

        # binary representation of math
        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.MAP_ = MAP

        # log_odds representation to simplify the math
        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)

        # occupancy probability
        self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64) * 0.5

        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)

    def good_lidar_idx(self, lidar_scan, L_MIN = 0.001, L_MAX = 30):
        #TODO: Truncate > L_Max and less than L_MIN
        lidar_scan = np.squeeze(lidar_scan)
        good_idxs = (lidar_scan > L_MIN) & (lidar_scan < L_MAX)
        return good_idxs

    def _polar_to_cart(self, lidar_scan, res_rad = 0.004359):
        """
        lidar_scan: m
        returns: 3 x m
        """
        angle_min = -2.35619 #rad
        angle_max = +2.35619 #rad
        lidar_angles = np.arange(angle_min, angle_max, res_rad)
        x = lidar_scan * np.cos(lidar_angles)
        y = lidar_scan * np.sin(lidar_angles)
        z = np.zeros_like(x)
        lidar_cart = np.vstack((x, y, z))
        return lidar_cart
    
    def _global_to_map_cell(self, g_pose, MAP):
        cell_x = max(min(MAP['xmax'], int(np.ceil(g_pose[0] / MAP['res'] + 1))), MAP['xmin'])
        cell_y = max(min(MAP['ymax'], int(np.ceil(g_pose[1] / MAP['res'] + 1))), MAP['ymin'])
        return (cell_x + 20, cell_y + 20)


    def _build_first_map(self,t0=0,use_lidar_yaw=True):
        """Build the first map using first lidar"""
        self.t0 = t0
        # Extract a ray from lidar data
        MAP = self.MAP_
        print('\n--------Doing build the first map--------')

        #TODO: student's input from here 
        #0) Extract Params from LiDAR and Joints
        lidar_scan, lidar_ts = self.lidar_._get_scan(idx=0)
        neck_angle, head_angle, _ = self.joints_._get_head_angles(ts=lidar_ts)
        good_lidar_idxs = self.good_lidar_idx(lidar_scan)
        l_lidar_pts = self._polar_to_cart(lidar_scan, res_rad=self.lidar_.res_rad)
        l_lidar_pts = l_lidar_pts[:, good_lidar_idxs]
        homo_l_lidar_pts = np.ones((4, l_lidar_pts.shape[1]), dtype=np.float64)
        homo_l_lidar_pts[:3, :] = l_lidar_pts
        yaw = self.lidar_.data_[0]['pose'][0, 2]

        #1) Transform LiDAR Scan to global world frame
        #a) lidar -> body
        R_bl = np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle))
        T_bl = np.array([0, 0, 0.15], dtype=np.float64)
        H_bl = tf.homo_transform(R_bl, T_bl)

        #b) body -> global (only considering yaw atm)
        R_gb = tf.rot_z_axis(yaw)
        T_gb = np.array([0.0, 0.0, self.h_lidar_])
        H_gb = tf.homo_transform(R_gb, T_gb)

        #c) apply to lidar_pts
        H_gl = H_gb @ H_bl
        g_lidar_pts = H_gl @ homo_l_lidar_pts

        #d) remove ground (all points with global y < 0.0)
        non_ground_idx = g_lidar_pts[2, : ] > 0.0
        g_lidar_pts = g_lidar_pts[:, non_ground_idx]

        #e) Use bresenham2D to get free/occupied cell locations 
        g_curr_pose = self.particles_[:, 10]
        m_curr_pose = self.lidar_._physicPos2Pos(MAP, g_curr_pose[:2])
        for ray in range(g_lidar_pts.shape[1]):
            m_lidar_pt = self.lidar_._physicPos2Pos(MAP, g_lidar_pts[:2, ray])
            ret = bresenham2D(m_curr_pose[0], m_curr_pose[1], m_lidar_pt[0], m_lidar_pt[1]).astype(int)
            free_coords = ret[:, :-1]
            occupied_coords = ret[:, -1]

            #f) Update Log Odds Map (increase all the free cells)
            log_pos = np.log(self.p_true_ / (1 - self.p_true_))
            log_neg = np.log(self.p_false_ / (1 - self.p_false_))
            self.log_odds_[tuple(occupied_coords)] += log_pos
            self.log_odds_[tuple(free_coords)] += log_neg

            MAP['map'][self.log_odds_ >= self.logodd_thresh_] = 1
            MAP['map'][self.log_odds_ < self.logodd_thresh_] = 0

        # # plt.imshow(MAP['map'])
        # plt.show()
        self.MAP_ = MAP

    def _mapping(self, idx=0, use_lidar_yaw=True):
        """Build the map """
        # Extract a ray from lidar data
        MAP = self.MAP_
        print('\n--------Building the map--------')

        #0) Extract Params from LiDAR and Joints
        lidar_scan, lidar_ts = self.lidar_._get_scan(idx=idx)
        neck_angle, head_angle, _ = self.joints_._get_head_angles(ts=lidar_ts)
        good_lidar_idxs = self.good_lidar_idx(lidar_scan)
        l_lidar_pts = self._polar_to_cart(lidar_scan, res_rad=self.lidar_.res_rad)
        l_lidar_pts = l_lidar_pts[:, good_lidar_idxs]
        homo_l_lidar_pts = np.ones((4, l_lidar_pts.shape[1]), dtype=np.float64)
        homo_l_lidar_pts[:3, :] = l_lidar_pts
        yaw = self.lidar_.data_[0]['pose'][0, 2]


        #1) Transform LiDAR Scan to global world frame
        #a) lidar -> body
        R_bl = np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle))
        T_bl = np.array([0, 0, 0.15], dtype=np.float64)
        H_bl = tf.homo_transform(R_bl, T_bl)

        #b) body -> global (only considering yaw atm)
        R_gb = tf.rot_z_axis(yaw)
        T_gb = np.array([0.0, 0.0, self.h_lidar_])
        H_gb = tf.homo_transform(R_gb, T_gb)

        #c) apply to lidar_pts
        H_gl = H_gb @ H_bl
        g_lidar_pts = H_gl @ homo_l_lidar_pts

        #d) remove ground (all points with global y < 0.0)
        non_ground_idx = g_lidar_pts[2, : ] > 0.0
        g_lidar_pts = g_lidar_pts[:, non_ground_idx]

        #e) Use bresenham2D to get free/occupied cell locations 
        g_curr_pose = self.particles_[:, 10]
        m_curr_pose = self.lidar_._physicPos2Pos(MAP, g_curr_pose[:2])
        for ray in range(g_lidar_pts.shape[1]):
            m_lidar_pt = self.lidar_._physicPos2Pos(MAP, g_lidar_pts[:2, ray])
            ret = bresenham2D(m_curr_pose[0], m_curr_pose[1], m_lidar_pt[0], m_lidar_pt[1]).astype(int)
            free_coords = ret[:, :-1]
            occupied_coords = ret[:, -1]

            #f) Update Log Odds Map (increase all the free cells)
            log_pos = np.log(self.p_true_ / (1 - self.p_true_))
            log_neg = np.log(self.p_false_ / (1 - self.p_false_))
            self.log_odds_[tuple(occupied_coords)] += log_pos
            self.log_odds_[tuple(free_coords)] += log_neg

            MAP['map'][self.log_odds_ >= self.logodd_thresh_] = 1
            MAP['map'][self.log_odds_ < self.logodd_thresh_] = 0
        plt.imshow(MAP['map'])
        plt.show()
        self.MAP_ = MAP


    def _predict(self,t,use_lidar_yaw=True):
        logging.debug('\n-------- Doing prediction at t = {0}------'.format(t))
        #TODO: student's input from here 
        #End student's input 

    def _update(self,t,t0=0,fig='on'):
        """Update function where we update the """
        MAP = self.MAP_
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return

        #TODO: student's input from here 
        #End student's input 

        self.MAP_ = MAP
        return MAP
