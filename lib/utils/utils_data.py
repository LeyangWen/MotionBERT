import os
import torch
import torch.nn.functional as F
import numpy as np
import copy
import warnings

def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result

def crop_scale_3d(motion, scale_range=[1, 1]):
    '''
        Motion: [T, 17, 3]. (x, y, z)
        Normalize to [-1, 1]
        Z is relative to the first frame's root.
    '''
    result = copy.deepcopy(motion)
    result[:,:,2] = result[:,:,2] - result[0,0,2]
    xmin = np.min(motion[...,0])
    xmax = np.max(motion[...,0])
    ymin = np.min(motion[...,1])
    ymax = np.max(motion[...,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) / ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,2] = result[...,2] / scale
    result = (result - 0.5) * 2
    return result

def flip_data(data, args=False):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    if args == False or args.joint_format.upper() == 'H36M':
        left_joints = [4, 5, 6, 11, 12, 13] # Human 3.6
        right_joints = [1, 2, 3, 14, 15, 16]
    elif args.joint_format.upper() == 'RTM-24':
        left_joints = [3, 5, 7, 9, 11, 13,16]
        right_joints = [4, 6, 8, 10, 12, 14,19]
    elif args.joint_format.upper() == 'Hand-21':
        left_joints = []
        right_joints = []
    else:
        raise ValueError("args.joint_format not recognized")
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1                                               # flip x of all joints
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data

def resample(ori_len, target_len, replay=False, randomness=True):
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result

def split_clips(vid_list, n_frames, data_stride):
    '''
    Args:
        vid_list: source name list
        n_frames: default 243
        data_stride: default 81

    Returns:

    '''
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i<len(vid_list):
        i += 1
        if i-st == n_frames:  # if clip is long enough, save it and move forward by stride
            result.append(range(st,i))
            saved.add(vid_list[i-1])
            st = st + data_stride
            n_clips += 1
        if i==len(vid_list):  # last clip
            break
        if vid_list[i]!=vid_list[i-1]:  # if source name changes, start a new clip
            if not (vid_list[i-1] in saved):
                resampled = resample(i-st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i-1])
            st = i
    return result