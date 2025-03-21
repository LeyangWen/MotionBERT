import torch
import numpy as np
import glob
import os
import io
import random
import pickle
import tqdm

from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data
from lib.data.datareader_edge_inference import DataReaderEdgeInference
    
class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        print(f"l29-file_list: {len(self.file_list)}")
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        print(f"l49-index: {index}")
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test" or self.data_split=="validate":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)


# class MotionInference3D(Dataset):
#     """
#     For inference data input without the convert to small pkl step
#     """
#     def __init__(self, args, test_data, test_labels=None):
#         self.test_data = test_data
#         self.test_labels = test_labels
#         self.flip = args.flip
#         self.synthetic = args.synthetic
#         self.aug = Augmenter3D(args)
#         self.gt_2d = args.gt_2d
#
#     def convert_dataset(self):
#         datareader = DataReaderEdgeInference(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
#                                     dt_file='h36m_sh_conf_cam_source_final.pkl', dt_root='data/motion3d/')
#         test_data, test_labels = datareader.get_sliced_data()
#         print(test_data.shape)
#         assert len(test_data) == len(test_labels)
#
#         self.test_data = test_data  # Nx243xJx3
#         self.test_labels = test_labels  # Nx243xJx3
#
#     def __getitem__(self, index):
#         """Generates one sample clip of data, 243xJx3, for both 2d and 3d"""
#
#         # Select sample
#         file_path = self.file_list[index]
#         motion_file = read_pkl(file_path)
#
#
#         motion_3d = self.test_data[index]
#         motion_2d = self.test_labels[index]
#         if self.gt_2d:
#             motion_2d[:,:,:2] = motion_3d[:,:,:2]
#             motion_2d[:,:,2] = 1
#         return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)
