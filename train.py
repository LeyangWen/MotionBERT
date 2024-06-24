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
    parser.add_argument('--save_trace', action='store_true')
    parser.add_argument('--test_set_keyword', default='validate', type=str, help='eval set name, either test or validate, only for VEHS')
    parser.add_argument('--wandb_project', default='MotionBert_train', type=str, help='wandb project name')
    parser.add_argument('--wandb_name', default='VEHS_ft_train', type=str, help='wandb run name')
    parser.add_argument('--note', default='', type=str, help='wandb notes')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)
    
def evaluate(args, model_pos, test_loader, datareader, save_trace=False):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()
    idx = 0
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
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip, args)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if save_trace:
                if idx in [0, 50, 100]:
                    predicted_3d_pos = model_pos(batch_input)
                    traced = torch.jit.trace(model_pos.module, batch_input)
                    # traced.save(f"model.pt")
                    # save input and output
                    input_output = {"input": batch_input.cpu(), "gt": batch_gt.cpu(), "output":predicted_3d_pos.cpu()}
                    with open(f"input_output.pkl", "wb") as f:
                        pickle.dump(input_output, f)
                elif idx == 101:
                    raise NotImplementedError("Tracing is done, exiting")
                idx += 1
                continue


            if args.rootrel:
                predicted_3d_pos[:,:,args.root_idx,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,args.root_idx,2] = 0

            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])

    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    
    # quick fix if you are not using 2.5d factor
    #factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])*0 +1
    #gts = np.array(datareader.dt_dataset['test']['joint3d_image'])
    
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    assert len(results_all)==len(action_clips)
    
    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02', 
                  's_09_act_10_subact_02', 
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:,None,None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor
        
        # Root-relative Errors
        # wen: root_idx 57 -- VEHS-7M 66 keypoint
        # Veeru: root_idx 21 -- RTMPose 24 keypoint
        pred = pred - pred[:,args.root_idx:args.root_idx+1,:]
        gt = gt - gt[:,args.root_idx:args.root_idx+1,:]

        # wen: convert to h36m temp fix
        # h36m_convert_id = [6, 8, 46, 47, 48, 50, 51, 52, 53, 54, 57, 58, 59, 60, 62, 63, 64]
        # pred = pred[:, h36m_convert_id, :]
        # gt = gt[:, h36m_convert_id, :]
        # wen: convert to only cacualted keypoints
        # pred = pred[:, 50:, :]
        # gt = gt[:, 50:, :]
                
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    print(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')
    return e1, e2, results_all
        
def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    model_pos.train()
    total_len = len(train_loader)
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        if idx/1000 == 0:
            print(f'INFO: Training batch {idx+1}/{total_len} @ {100*(idx+1)/total_len:.2f}%.', end='\r')
        batch_size = len(batch_input)        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(batch_input[:,:,:,2:])    # For 2D data, weight/confidence is at the last channel
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:,:,args.root_idx:args.root_idx+1,:]
            else:
                batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,args.root_idx:args.root_idx+1,2]  # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Predict 3D poses
        predicted_3d_pos = model_pos(batch_input)    # (N, T, 17, 3)
        
        optimizer.zero_grad()
        if has_3d:
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos, args)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt, args)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            loss_total = loss_3d_pos + \
                         args.lambda_scale       * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         args.lambda_lv          * loss_lv + \
                         args.lambda_lg          * loss_lg + \
                         args.lambda_a           * loss_a  + \
                         args.lambda_av          * loss_av
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()

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
                # todo: use a args flag or other checks instead, sometimes we dont need to delete the last layer
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

    if not opts.evaluate:        
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d), len(instav_loader_2d) + len(posetrack_loader_2d)))
        else:
            print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
                
        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)
        
        print("Start training loop")
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            N = 0
                        
            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True) 
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)
                print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1, e2))
                train_writer.add_scalar('Error P1', e1, epoch + 1)
                train_writer.add_scalar('Error P2', e2, epoch + 1)
                train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
                
                wandb.log({
                    'Error P1': e1,
                    'Error P2': e2,
                    'loss_3d_pos': losses['3d_pos'].avg,
                    'loss_2d_proj': losses['2d_proj'].avg,
                    'loss_3d_scale': losses['3d_scale'].avg,
                    'loss_3d_velocity': losses['3d_velocity'].avg,
                    'loss_lv': losses['lv'].avg,
                    'loss_lg': losses['lg'].avg,
                    'loss_a': losses['angle'].avg,
                    'loss_av': losses['angle_velocity'].avg,
                    'loss_total': losses['total'].avg
                }, step=epoch + 1)
                
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)
            
    if opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader, save_trace=opts.save_trace)
        os.makedirs(opts.out_path, exist_ok=True)
        np.save('%s/X3D.npy' % (opts.out_path), results_all)
        # render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=50)
        
        wandb.log({
            'Error P1': e1,
            'Error P2': e2})
        
    wandb.finish()

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    try:
        args.joint_format
    except:
        raise ValueError("Add joint_format in your config file, used for loss.py --> limb_loss & utils_data.py --> flip")
    train_with_config(args, opts)
