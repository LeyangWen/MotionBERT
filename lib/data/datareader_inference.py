# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
from lib.data.datareader_VEHSR3 import DataReaderVEHSR3
random.seed(0)
    
class DataReaderInference(DataReaderVEHSR3):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl', test_set_keyword='test', num_joints=17, res_w=1920, res_h=1080):
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
        super().__init__(n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence, dt_root, dt_file, test_set_keyword=test_set_keyword, num_joints=num_joints)
        # todo: set res_w and res_h according to video input
        self.res_w = res_w
        self.res_h = res_h
        
    def read_2d(self):
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            res_w, res_h = self.res_w, self.res_h
            trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            res_w, res_h = self.res_w, self.res_h
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
                if len(train_confidence.shape)==2: # (1559752, 17)
                    train_confidence = train_confidence[:,:,None]
                    test_confidence = test_confidence[:,:,None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:,:,0:1]
                test_confidence = np.ones(testset.shape)[:,:,0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset

    def read_3d(self):
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)    # [N, 17, 3]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            res_w, res_h = self.res_w, self.res_h
            train_labels[idx, :, :2] = train_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[idx, :, 2:] = train_labels[idx, :, 2:] / res_w * 2
            
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            res_w, res_h = self.res_w, self.res_h
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2
            
        return train_labels, test_labels

    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            res_w, res_h = self.res_w, self.res_h
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw

    def denormalize(self, test_data):
        #       data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)
        n_clips = test_data.shape[0]
        test_hw = self.read_hw()
        data = test_data.reshape([n_clips, -1, self.num_joints, 3])
        assert len(data) == len(test_hw), f"len(data): {len(data)} != len(test_hw): {len(test_hw)}"
        # denormalize (x,y,z) coordiantes for results
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data  # [n_clips, -1, 17, 3]
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        split_id_train, split_id_test = self.get_split_id()
        print(f"train_data: {train_data.shape} test_data: {test_data.shape} train_labels: {train_labels.shape} test_labels: {test_labels.shape}")
        print(f"split_id_train: {split_id_train.shape} split_id_test: {split_id_test.shape}")
        # print(f"split_id_train[0]: {split_id_train[0]} split_id_test[0]: {split_id_test[0]}")
        print()
        try:
            train_data = train_data[split_id_train]
            train_labels = train_labels[split_id_train]
        except:
            print(f"Warning: Error slicing train_data and train_labels")
            train_data = []
            train_labels = []
        test_data =test_data[split_id_test]                # (N, 27, 17, 3)
        test_labels = test_labels[split_id_test]        # (N, 27, 17, 3)
        return train_data, test_data, train_labels, test_labels