# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
from lib.data.datareader_VEHSR3 import DataReaderVEHSR3
random.seed(0)
    
class DataReaderVEHSHand(DataReaderVEHSR3):
    """
    0: Wrist
    1-4: Thumb
    5-8: Index
    9-12: Middle
    13-16: Ring
    17-20: Pinky
    """
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl', test_set_keyword='test', num_joints=21):
        '''
        Args:
            n_frames: frames in each clip
            sample_stride: downsample the data by this stride
            data_stride_train: avoid redundancy in training data by moving starting frame forward by this stride
            data_stride_test:
            read_confidence:
            dt_root:
            dt_file:
            test_set_keyword: dictionary key for the dataset pickle, set to 'test' or 'validate'
        '''
        super().__init__(n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence, dt_root, dt_file)
        self.dt_dataset['test'] = self.dt_dataset[test_set_keyword]
        self.dt_dataset['test']['action'] = list(map(str.lower, self.dt_dataset['test']['action']))
        self.res_w = 1000
        self.res_h = 1000
        self.num_joints = num_joints

class DataReaderVEHSUpperBodyHand(DataReaderVEHSR3):
    """
    0: left shoulder
    1: right shoulder
    2: left arm
    3: right arm
    4: left forearm 
    5: righ forearm
    6: left wrist
    7-10: left thumb
    11-14: left index
    15-18: left middle
    19-22: left ring
    23-26: left pinky
    27: right wrist
    28-31: right thumb
    32-35: right index
    36-39: right middle
    40-43: right ring
    44-47: right pinky
    """
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl', test_set_keyword='test', num_joints=28):
        '''
        Args:
            n_frames: frames in each clip
            sample_stride: downsample the data by this stride
            data_stride_train: avoid redundancy in training data by moving starting frame forward by this stride
            data_stride_test:
            read_confidence:
            dt_root:
            dt_file:
            test_set_keyword: dictionary key for the dataset pickle, set to 'test' or 'validate'
        '''
        super().__init__(n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence, dt_root, dt_file)
        self.dt_dataset['test'] = self.dt_dataset[test_set_keyword]
        self.dt_dataset['test']['action'] = list(map(str.lower, self.dt_dataset['test']['action']))
        self.res_w = 3840
        self.res_h = 2160
        self.num_joints = num_joints
