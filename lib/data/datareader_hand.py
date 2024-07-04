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

    
