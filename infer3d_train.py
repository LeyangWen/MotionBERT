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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-o', '--out_path', type=str, help='eval pose output path', default=r'experiment/VEHS-7M_6D')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--test_set_keyword', default='validate', type=str, help='eval set name, either test or validate, only for VEHS')
    parser.add_argument('--wandb_project', default='MotionBert_train', type=str, help='wandb project name')
    parser.add_argument('--wandb_name', default='VEHS_ft_train', type=str, help='wandb run name')
    parser.add_argument('--note', default='', type=str, help='wandb notes')
    parser.add_argument('--res_w', type=int, default=1920)
    parser.add_argument('--res_h', type=int, default=1080)
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def evaluate(args, model_pos, test_loader, datareader):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()            
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input, args)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip, args)                   # Flip back
                predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,args.root_idx,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,args.root_idx,2] = 0

            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    
    return results_all


def train_with_config(args, opts):

    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    args_all = vars(opts)
    args_all['yaml_config'] = args
    this_run = wandb.init(project=opts.wandb_project, name=opts.wandb_name, config=args_all)  # Initialize a new run
    
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers':2,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 2,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, opts.test_set_keyword)
    if not opts.evaluate:      
        train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    
    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)
    if "VEHS" in opts.config:
        test_set_keyword = opts.test_set_keyword
        path_components = args.data_root.split('/')
        this_dt_root = '/'.join(path_components[:-2])  # this_dt_root='data/motion3d'
        datareader = DataReaderVEHSR3(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = this_dt_root, dt_file=args.dt_file, test_set_keyword=test_set_keyword, num_joints=args.num_joints)
    elif "infer" in opts.config:
        test_set_keyword = opts.test_set_keyword
        path_components = args.data_root.split('/')
        # todo: res_w and res_h here
        this_dt_root = '/'.join(path_components[:-2])  # this_dt_root='data/motion3d'
        datareader = DataReaderInference(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = this_dt_root, dt_file=args.dt_file, test_set_keyword=test_set_keyword, num_joints=args.num_joints)
    elif "h36m" in opts.config:  # H36M
        datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root='data/motion3d',
                                    dt_file=args.dt_file)
    else:
        raise ValueError('make sure dataset name (e.g., h36m, VEHS) is in opts.config')
     
    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            if args.num_joints != 17:  # 6D pose
                # for key, value in checkpoint['model_pos'].items():
                #     print(key, value.shape)
                del checkpoint['model_pos']['module.pos_embed']  # deleting the last layer
                print('INFO: Starting new 6D pose using a 3D pose checkpoint, deleting the last layer')
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
            model_pos = model_backbone
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone
        
    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if opts.evaluate:
        results_all = evaluate(args, model_pos, test_loader, datareader)
        os.makedirs(opts.out_path, exist_ok=True)
        np.save('%s/X3D.npy' % (opts.out_path), results_all)
        # render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=50)
    else:
        raise ValueError('This script is for inference only, please set --evaluate')

        
    wandb.finish()

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    try:
        args.joint_format
    except:
        raise ValueError("Add joint_format in your config file")
    train_with_config(args, opts)
