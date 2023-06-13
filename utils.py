import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def bland_altman_plot(method_a, method_b):
    # Calculate differences and mean difference
    differences = method_a - method_b
    mean_difference = np.mean(differences)

    # Calculate limits of agreement
    std_difference = np.std(differences)
    lower_limit = mean_difference - 1.96 * std_difference
    upper_limit = mean_difference + 1.96 * std_difference

    # Create the Bland-Altman plot
    plt.scatter((method_a + method_b) / 2, differences, color='black')
    plt.axhline(mean_difference, color='red', linestyle='--', label='Mean Difference')
    plt.axhline(upper_limit, color='blue', linestyle='--', label='Upper Limit of Agreement')
    plt.axhline(lower_limit, color='blue', linestyle='--', label='Lower Limit of Agreement')

    # Add labels and legend
    plt.xlabel('Average of Measurements (Method A + Method B)/2')
    plt.ylabel('Difference (Method A - Method B)')
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    exp_dir = r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\experiment\Vicon_Gunwoo_Test_movement02'
    camera_DEVICEID = '66920731'
    exp_name = 'Gunwoo movement 02'
    vicon_csv_file = os.path.join(exp_dir, f'{exp_name}.csv')
    smpl_csv_file = os.path.join(exp_dir,'output', f'{exp_name}.{camera_DEVICEID}','smpl_pose.csv')

    # read smpl pose
    smpl_pose = np.loadtxt(smpl_csv_file, delimiter=',')