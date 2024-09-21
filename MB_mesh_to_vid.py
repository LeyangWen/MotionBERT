import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
# import matplotlib
# matplotlib.use('Qt5Agg')
import os

from lib.utils.tools import *
from lib.utils.utils_smpl import *
from lib.utils.vismo import render_and_save


def prase_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--motion_smpl_file', type=str, default=r'/Users/leyangwen/Documents/mesh/H36M Results_small.pkl')
    parser.add_argument('--motion_smpl_file', type=str, default=r'/Volumes/Z/mesh/H36M Results_1000.pkl')
    parser.add_argument('--out_path', type=str, default=r'/Users/leyangwen/Documents/mesh/')
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()
    args.dataset_name = args.motion_smpl_file.split('/')[-1].split('.')[0]
    return args

def save_first_frames(results, frames=1000):
    for key in results.keys():
        results[key] = results[key][:frames]
    with open(args.motion_smpl_file.replace('.pkl', f'_{frames}.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    args = prase_args()
    with open(args.motion_smpl_file, 'rb') as f:
        results = pickle.load(f)

    # save_first_frames(results, frames=1000)

    vertices = results['verts'].reshape(-1, 6890, 3)
    joints = results['kp_3d'].reshape(-1, 17, 3)
    # global_orient = data['global_orient']
    # body_pose = data['body_pose']



    render_and_save(vertices, osp.join(args.out_path, f'{args.dataset_name}.mp4'), keep_imgs=False, fps=args.fps, draw_face=True)

