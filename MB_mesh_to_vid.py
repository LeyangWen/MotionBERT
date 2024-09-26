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
    parser.add_argument('--motion_smpl_file', type=str, default=r'/Users/leyangwen/Documents/mesh/H36M Results_300.pkl')
    parser.add_argument('--out_path', type=str, default=r'/Users/leyangwen/Documents/mesh/')
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()
    args.dataset_name = args.motion_smpl_file.split('/')[-1].split('.')[0]
    return args

def save_first_frames(results, frames=1000):
    for key in results.keys():
        results[key] = results[key][:frames]
    short_file_name = args.motion_smpl_file.split('/')[-1].replace('.pkl', f'_{frames}.pkl')
    with open(short_file_name, 'wb') as f:
        pickle.dump(results, f)
    raise ValueError(f'Intended force break, saved to {short_file_name}')

def mb_result_view_rotate(verts):
    """
    Rotation for the MB results: Rotation matrix for 90 degrees around x-axis
    """
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    return np.dot(verts, rotation_matrix.T)

if __name__ == '__main__':
    args = prase_args()
    with open(args.motion_smpl_file, 'rb') as f:
        results = pickle.load(f)

    # save_first_frames(results, frames=300)

    vertices = results['verts'].reshape(-1, 6890, 3)
    joints = results['kp_3d'].reshape(-1, 17, 3)
    vertices_gt = results['verts_gt'].reshape(-1, 6890, 3)
    joints_gt = results['kp_3d_gt'].reshape(-1, 17, 3)

    # rotate all 3D verticies in x axis by 90 degree
    rotated_vertices = np.array([mb_result_view_rotate(v) for v in vertices])
    rotated_joints = np.array([mb_result_view_rotate(j) for j in joints])/1000
    rotated_vertices_gt = np.array([mb_result_view_rotate(v) for v in vertices_gt])
    rotated_joints_gt = np.array([mb_result_view_rotate(j) for j in joints_gt])/1000

    render_and_save(rotated_vertices, osp.join(args.out_path, f'{args.dataset_name}_verts.mp4'), keep_imgs=False, fps=args.fps, draw_face=True)
    render_and_save(rotated_vertices_gt, osp.join(args.out_path, f'{args.dataset_name}_verts_gt.mp4'), keep_imgs=False, fps=args.fps, draw_face=True)
    render_and_save(rotated_joints, osp.join(args.out_path, f'{args.dataset_name}_joints.mp4'), keep_imgs=False, fps=args.fps, draw_face=False)
    render_and_save(rotated_joints_gt, osp.join(args.out_path, f'{args.dataset_name}_joints_gt.mp4'), keep_imgs=False, fps=args.fps, draw_face=False)

