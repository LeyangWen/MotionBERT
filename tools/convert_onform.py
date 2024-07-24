print("running convert VEHS")
import os
import sys
import pickle
import numpy as np
import random
sys.path.insert(0, os.getcwd())
from lib.utils.tools import read_pkl
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.datareader_VEHSR3 import DataReaderVEHSR3
from lib.data.datareader_onform import DataReaderOnform
from tqdm import tqdm
import argparse
import datetime

def save_clips(subset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:  
            pickle.dump(data_dict, myprofile)

parser = argparse.ArgumentParser()
parser.add_argument('--dt_root', type=str, default='data/motion3d/')
parser.add_argument('--dt_file', type=str, default='h36m_sh_conf_cam_source_final.pkl')
parser.add_argument('--root_path', type=str, default='data/motion3d/MB3D_f243s81/H36M-SH"')
parser.add_argument('--test_set_keyword', default='test', type=str, help='eval set name, either test or validate, only for VEHS')
args = parser.parse_args()

datareader = DataReaderOnform(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, dt_file=args.dt_file, dt_root=args.dt_root, test_set_keyword=args.test_set_keyword)
train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
print(f"train_data: {train_data.shape}, test_data: {test_data.shape}")
print(f"train_labels: {train_labels.shape}, test_labels: {test_labels.shape}")
iteration_time = 1  #s
print(f'Estimated training time per epoch @ {iteration_time} it/s: {datetime.timedelta(seconds=iteration_time*len(train_data))}')
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

root_path = args.root_path
if not os.path.exists(root_path):
    os.makedirs(root_path)

save_clips("train", root_path, train_data, train_labels)
save_clips(str(args.test_set_keyword), root_path, test_data, test_labels)

