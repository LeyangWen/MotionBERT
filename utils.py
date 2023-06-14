import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation

def bland_altman_plot(method_a, method_b,name = ''):
    # Calculate differences and mean difference
    differences = method_a - method_b
    mean_difference = np.mean(differences)

    # Calculate limits of agreement
    std_difference = np.std(differences)
    lower_limit = mean_difference - 1.96 * std_difference
    upper_limit = mean_difference + 1.96 * std_difference

    # Create the Bland-Altman plot
    plt.scatter((method_a + method_b) / 2, differences, color='black', s=2)
    plt.axhline(mean_difference, color='red', linestyle='--', label='Mean Difference')
    plt.axhline(upper_limit, color='blue', linestyle='--', label='Upper Limit of Agreement')
    plt.axhline(lower_limit, color='blue', linestyle='--', label='Lower Limit of Agreement')

    # Add labels and legend
    plt.xlabel('Average of Measurements (SMPL + Vicon)/2')
    plt.ylabel('Difference (SMPL - Vicon)')
    plt.title(f'Bland-Altman Plot: {name} ({mean_difference:.2f}±{std_difference*1.96:.2f})')
    plt.legend()

    # Show the plot
    plt.show()

def load_csv(csv_filename):
    df = pd.read_csv(csv_filename, skiprows=[0, 1, 2,3, 4], header=None)
    header_df = pd.read_csv(csv_filename, skiprows=[0, 1], nrows=1, header=None)
    # remove nan in the header
    # iterate over header_df
    header = ['Frame', 'Sub Frame']
    axis = ['X', 'Y', 'Z']
    joint_names = []
    # LAnkleAngles, LElbowAngles, LHeadAngles, LHipAngles, LKneeAngles, LNeckAngles, LPelvisAngles, LShoulderAngles, LSpineAngles, LThoraxAngles, LWristAngles, RAnkleAngles, RElbowAngles, RHeadAngles, RHipAngles, RKneeAngles, RNeckAngles, RPelvisAngles, RShoulderAngles, RSpineAngles, RThoraxAngles, RWristAngles
    for x in header_df.iloc[0]:
        # print(x)
        if type(x) == str:
            joint_name = x.split(':')[-1]
            joint_names.append(joint_name)
            for ax in axis:
                header.append(f'{joint_name}-{ax}')
    # add header to df
    df.columns = header
    return df, joint_names


def euler_to_quaternion(angles, order='xyz'):
    # Create rotation object from Euler angles
    rotation = Rotation.from_euler(order, angles)

    # Get the quaternion representation
    quaternion = rotation.as_quat()

    return quaternion


def quaternion_magnitude(quaternion):
    # Create rotation object from quaternion
    rotation = Rotation.from_quat(quaternion)

    # Get the single angle representation
    angle = rotation.magnitude()

    return angle


if __name__ == '__main__':
    exp_dir = r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\experiment\Vicon_Gunwoo_Test_movement02'
    camera_DEVICEID = '66920731'
    exp_name = 'Gunwoo movement 02'
    vicon_csv_file = os.path.join(exp_dir, f'{exp_name}.angles.csv')
    smpl_csv_file = os.path.join(exp_dir,'output', f'{exp_name}.{camera_DEVICEID}','smpl_pose.csv')

    # read smpl pose
    smpl_pose = np.loadtxt(smpl_csv_file, delimiter=',')
    # read vicon pose
    vicon_angles, joint_names = load_csv(vicon_csv_file)

    smpl_vicon_compare = {}
    smpl_vicon_compare['LElbow'] = [joint_names.index('LElbowAngles'), 18]
    smpl_vicon_compare['RElbow'] = [joint_names.index('RElbowAngles'), 19]
    # smpl_vicon_compare['LShoulder'] = [joint_names.index('LShoulderAngles'), 16]
    # smpl_vicon_compare['RShoulder'] = [joint_names.index('RShoulderAngles'), 17]
    smpl_vicon_compare['LKnee'] = [joint_names.index('LKneeAngles'), 4]
    smpl_vicon_compare['RKnee'] = [joint_names.index('RKneeAngles'), 5]

    for k, v in smpl_vicon_compare.items():
        print(k)
        vicon_idx, smpl_idx = v
        smpl_angle = smpl_pose[:, smpl_idx * 3:smpl_idx * 3 + 3]
        vicon_angle = vicon_angles.iloc[0::2, vicon_idx * 3 + 2: vicon_idx*3+5] / 180 * np.pi
        vicon_angle = vicon_angle.values
        vicon_q = euler_to_quaternion(vicon_angle)
        vicon_magnitude = quaternion_magnitude(vicon_q)
        smpl_q = euler_to_quaternion(smpl_angle)
        smpl_magnitude = quaternion_magnitude(smpl_q)
        bland_altman_plot(smpl_magnitude, vicon_magnitude, name = k)
        # plot smpl and vicon
        plt.plot(smpl_magnitude, label='smpl')
        plt.plot(vicon_magnitude, label='vicon')
        plt.title(f'Time Series Plot: {k}')
        plt.legend()
        plt.show()

