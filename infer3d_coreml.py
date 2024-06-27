import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable
import wandb
import coremltools as ct
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.datareader_VEHSR3 import DataReaderVEHSR3
from lib.data.datareader_inference import DataReaderInference
from lib.model.loss import *
from edge.edge_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_train_h36m.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-o', '--out_path', type=str, help='eval pose output path', default=r'experiment/coreml_h36m')
    parser.add_argument('--test_set_keyword', default='test', type=str, help='eval set name, either test or validate, only for VEHS')
    parser.add_argument('--coreml_file', type=str, default=r'edge/MB_h36m.mlpackage')
    parser.add_argument('--residual_mode', type=str, default=r'discard')
    parser.add_argument('--res_hw', default=(1000,10000))

    parser.add_argument('--wandb_mode', default='disabled', type=str, help=r'"online", "offline" or "disabled"')
    parser.add_argument('--wandb_project', default='MotionBert_train', type=str, help='wandb project name')
    parser.add_argument('--wandb_name', default='VEHS_ft_train', type=str, help='wandb run name')
    parser.add_argument('--note', default='', type=str, help='wandb notes')
    opts = parser.parse_args()
    return opts


def infer(args, coreml_model, test_clips):
    """
    For all clips, run model, concat results
    test_clips: Nx243xJx3 (norm_px_x, norm_px_y, confidence), normalized
    """
    print('INFO: Inference')
    results_all = []
    for batch_input in tqdm(test_clips):  # batch (N) is always 1 now in infer
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if args.flip:
            batch_input_flip = flip_data(batch_input, args)
            predicted_3d_pos_1 = model_pos_coreml(coreml_model, batch_input)
            predicted_3d_pos_flip = model_pos_coreml(coreml_model, batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip, args)                   # Flip back
            predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
        else:
            predicted_3d_pos = model_pos_coreml(coreml_model, batch_input)
        if args.rootrel:
            predicted_3d_pos[:,:,args.root_idx,:] = 0     # [N,T,17,3]
        if args.gt_2d:
            predicted_3d_pos[...,:2] = batch_input[...,:2]
        results_all.append(predicted_3d_pos)
    results_all = np.concatenate(results_all)
    return results_all


def infer_with_config(args, opts):
    print(args)
    args_all = vars(opts)
    args_all['yaml_config'] = args
    wandb.init(project=opts.wandb_project, name=opts.wandb_name, config=args_all, mode=opts.wandb_mode)  # Initialize a new run


    print('Loading dataset...')

    coreml_model = ct.models.MLModel(opts.coreml_file)
    res_h, res_w = opts.res_hw
    args.test_set_keyword = opts.test_set_keyword
    input_2d_conf = mock_input_pkl(args)

    start_time = time.time()
    frames = input_2d_conf.shape[0]
    input_2d_conf = normalize_2d(input_2d_conf, res_h, res_w)
    # no downsampling or stride
    input_2d_conf, split_info = split_infer_clips(input_2d_conf, n_frames=args.clip_len, residual_mode=opts.residual_mode)

    results_all = infer(args, coreml_model, input_2d_conf)
    results_all = denormalize(results_all, res_h, res_w)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    fps = frames / elapsed_time
    print(f"Elapsed time: {int(minutes):02d} min:{int(seconds):02d}s")
    print(f"fps: {fps:.2f}, for {frames} frames")

    wandb.log({
        'Elapsed time (min:s)': f"{int(minutes):02d} min:{int(seconds):02d}s",
        'Frames per second (FPS)': fps
    })

    os.makedirs(opts.out_path, exist_ok=True)
    np.save('%s/X3D.npy' % (opts.out_path), results_all)
    # render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=50)
    print(f"Output saved to {'%s/X3D.npy' % (opts.out_path)}")
    wandb.finish()

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    try:
        args.joint_format
    except:
        warnings.warn("no joint_format in your config file, defaulting to h36m")
        args.joint_format = 'h36m'
        args.root_idx = 0
        # raise ValueError("Add joint_format in your config file, used for loss.py --> limb_loss & utils_data.py --> flip")
    infer_with_config(args, opts)

    # todo: test speed & accuracy with args.flip on and off
