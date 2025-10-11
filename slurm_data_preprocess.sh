#!/bin/bash -l
#SBATCH --job-name=VEHS_3D_dataset_preprocess
#SBATCH --output=output_slurm/preprocess_log.txt
#SBATCH --error=output_slurm/preprocess_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --time=00:30:00
#SBATCH --account=shdpm0
#SBATCH --partition=standard
##### END preamble
##### Run in MotionBert dir

my_job_header
module purge
# module load python3.10-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
module load python/3.10.4
module load pytorch/2.0.1
module load numpy
module load matplotlib
module list

#source /home/wenleyan/.conda/envs/motionbert/bin/activate motionbert
# conda activate motionbert


#################### Viewpoint augmentation experiments

# ########## Single camera
# python -u tools/convert_VEHSR3.py \
# --dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/viewpoint_augmentation/single_cam/normal/' \
# --dt_file 'VEHS_6D_downsample5_keep1_37_oneCam.pkl' \
# --test_set_keyword 'validate' \
# --root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/viewpoint_augmentation/single_cam/normal/'

#### pitch correct version
# /nfs/turbo/coe-shdpm/leyang/VEHS_MB/viewpoint_augmentation/single_cam/pitch_correct/VEHS_6D_downsample5_keep1_37_oneCam_pitch_correct.pkl

# python -u tools/convert_VEHSR3.py \
# --dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/viewpoint_augmentation/single_cam/pitch_correct/' \
# --dt_file 'VEHS_6D_downsample5_keep1_37_oneCam_pitch_correct.pkl' \
# --test_set_keyword 'validate' \
# --root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/viewpoint_augmentation/single_cam/pitch_correct/'


#################### Hand
# Lab vid - Hand-21
#python -u tools/convert_VEHS_hand.py \
#--dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/Lab_hand' \
#--dt_file 'lab_rtmpose_hand_for_MB.pkl' \
#--test_set_keyword 'test' \
#--root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/Lab_hand' > output_slurm/Rokoko_hand.out
#
#python -u tools/convert_VEHS_hand.py \
#--dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/Lab_hand' \
#--dt_file 'lab_rtmpose_hand_for_MB.pkl' \
#--test_set_keyword 'validate' \
#--root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/Lab_hand'

## Rokoko - Hand-21
#python -u tools/convert_VEHS_hand.py \
#--dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/Rokoko_hand' \
#--dt_file 'Rokoko_hand_3D_downsample1_keep1.pkl' \
#--test_set_keyword 'test' \
#--root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/Rokoko_hand' > output_slurm/Rokoko_hand.out

################ upper body gesture control
## Rokoko - UBHand48 gesture control
# python -u tools/convert_VEHS_UBHand.py \
# --dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/Upper_body_48' \
# --dt_file 'Gesture_3D_downsample1_keep3.pkl' \
# --test_set_keyword 'test' \
# --root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/Upper_body_48'


################## RTMPose +
# VEHS - RTMPose37 - VEHSR3
# python -u tools/convert_VEHSR3.py \
# --dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_37kpts_VEHS7M/normal' \
# --dt_file 'VEHS_6D_downsample5_keep1_37_v2_RTM2D.pkl' \
# --test_set_keyword 'test' \
# --root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_37kpts_VEHS7M/normal' > output_slurm/preprocess_RTM6.out

#### pitch correct version
# python -u tools/convert_VEHSR3.py \
# --dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_37kpts_VEHS7M/V5_2-b/pitch_correct' \
# --dt_file 'VEHS_6D_downsample5_keep1_37_v2_pitch_correct_modified_RTM2D.pkl' \
# --test_set_keyword 'validate' \
# --root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_37kpts_VEHS7M/V5_2-b/pitch_correct' > output_slurm/preprocess_RTM6.out

# VEHS - RTMPose37 - industry inference
# python -u tools/convert_inference.py \
# --dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_industry/37kpts_v5-2b' \
# --dt_file 'rtmpose_v5-2b_20fps_industry_37kpts_v2.pkl' \
# --test_set_keyword 'validate' \
# --root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_industry/37kpts_v5-2b' > output_slurm/preprocess_RTM6.out

## VEHS - RTMPose24 - VEHSR3
#python -u tools/convert_VEHSR3.py \
#--dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_VEHS_tilt_correct' \
#--dt_file 'VEHS_6D_downsample5_keep1_config6_tilt_corrected_modified_RTM2D.pkl' \
#--test_set_keyword 'validate' \
#--root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_VEHS_tilt_correct' > output_slurm/preprocess_RTM6.out

################## paper
# H36M
#python -u tools/convert_h36m.py > output_slurm/preprocess_H36M.out

# VEHSR3
# python -u tools/convert_VEHSR3.py \
# --dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/3DPose/20fps' \
# --dt_file 'VEHS_3D_downsample5_keep1_H36M_17.pkl' \
# --test_set_keyword 'test' \
# --root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/3DPose/20fps' > output_slurm/preprocess.out

# VEHSR3 - 6D
# python -u tools/convert_VEHSR3.py \
# --dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/6DPose' \
# --dt_file 'VEHS_6D_downsample2_keep1.pkl' \
# --test_set_keyword 'validate' \
# --root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/6DPose' > output_slurm/preprocess_6D.out


# VEHSR3R4 - 6D - 66v2
python -u tools/convert_VEHSR3.py \
--dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_R3_R4_MB/66kptv2/GT2D/50fps' \
--dt_file 'VEHS_6D_downsample2_keep1_pitch_correct_0.pkl' \
--test_set_keyword 'validate' \
--root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_R3_R4_MB/66kptv2/GT2D/50fps' > output_slurm/preprocess_RTM6.out