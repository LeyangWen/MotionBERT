#!/bin/bash
#SBATCH --job-name=MB_train_RTM_VEHS
#SBATCH --output=output_slurm/train_log_3D.txt
#SBATCH --error=output_slurm/train_error_3D.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20g
#SBATCH --gres=gpu:5
#SBATCH --time=24:00:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu
##### END preamble
##### Run in MotionBert dir

my_job_header
module load python3.10-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
module load python/3.10.4
module load pytorch/2.0.1
module load numpy
module load matplotlib
module list

#conda activate motionbert



#################### Viewpoint augmentation experiments

########## Single camera
### Pretrain  # set gt_2d in config to True
# python train.py \
# --pretrained /nfs/turbo/coe-shdpm/leyang/MB_checkpoints/pose3d/MB_ft_VEHSR3_6DPose/ \
# --test_set_keyword validate \
# --wandb_project "MotionBert_train_RTM2D" \
# --selection best_epoch.bin \
# --discard_last_layer \
# --config configs/pose3d/RTMPose_exp/37kpts_v1/viewpoint_aug/MB_ft_VEHS_20fps.yaml \
# --wandb_name "37kpts_v2_20fps-pretrain-normal-oneCam-1" \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/20fps-pretrain-normal-oneCam-1" \


#################### Hand
# # Rokoko - Hand-21
# python train.py \
# --config configs/pose3d/hand/MB_train_Rokoko.yaml \
# --test_set_keyword validate \
# --wandb_project "MotionBert_train_Hand" \
# --wandb_name "Rokoko_2" \
# --checkpoint checkpoint/pose3d/MB_train_Rokoko_hand_21 > output_slurm/train_hand.out

# Rokoko - UBHand48 gesture control
# python train.py \
# --config configs/pose3d/hand/MB_train_Rokoko.yaml \
# --pretrained checkpoint/pose3d/MB_ft_VEHSR3_6DPose/ \
# --test_set_keyword validate \
# --wandb_project "MotionBert_train_Hand" \
# --wandb_name "Rokoko_WB_exp1" \
# --selection best_epoch.bin \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_WBHand48_checkpoints/exp1" \
# --discard_last_layer \
# --resume "/scratch/shdpm_root/shdpm0/wenleyan/MB_WBHand48_checkpoints/exp1/latest_epoch.bin"\


# finetune RTMPose37 - VEHS

### Pretrain  # set gt_2d in config to True
# python train.py \
# --pretrained /nfs/turbo/coe-shdpm/leyang/MB_checkpoints/pose3d/MB_ft_VEHSR3_6DPose/ \
# --test_set_keyword validate \
# --wandb_project "MotionBert_train_RTM2D" \
# --selection best_epoch.bin \
# --discard_last_layer \
# --config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps.yaml \
# --wandb_name "37kpts_v2_20fps-pretrain-normal-2" \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/20fps-pretrain-normal-1" \


### Finetune   # set gt_2d in config to False
# python train.py \
# --pretrained /nfs/turbo/coe-shdpm/leyang/MB_checkpoints/pose3d/MB_ft_VEHSR3_3DPose/ \
# --selection latest_epoch.bin \
# --discard_last_layer \
# --test_set_keyword validate \
# --wandb_project "MotionBert_train_RTM2D" \
# --note "default loss" \
# --config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps_pitch_correct.yaml \
# --wandb_name "RTMW37kpts_v2_20fps-finetune-pitch-correct-1-OG" \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/RTMW37kpts_v2_20fps-finetune-pitch-correct-1-OG" \


python train.py \
--pretrained /nfs/turbo/coe-shdpm/leyang/MB_checkpoints/pose3d/MB_ft_VEHSR3_3DPose/ \
--selection latest_epoch.bin \
--discard_last_layer \
--test_set_keyword validate \
--wandb_project "MotionBert_train_RTM2D" \
--note "OG loss" \
--config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps_pitch_correct_3.yaml \
--wandb_name "Try2-RTMW37kpts_v2_20fps-finetune-pitch-correct-1-OG" \
--checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/Try2/RTMW37kpts_v2_20fps-finetune-pitch-correct-1-OG" \

# --note "limb loss V1" \
# --config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps_pitch_correct_2.yaml \
# --wandb_name "Try2-RTMW37kpts_v2_20fps-finetune-pitch-correct-2-limbLoss" \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/Try2/RTMW37kpts_v2_20fps-finetune-pitch-correct-2-limbLoss" \









# --note "angle loss V2 only" \
# --config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps_pitch_correct.yaml \
# --wandb_name "Try2-RTMW37kpts_v2_20fps-finetune-pitch-correct-5-angleLossV2-only" \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/Try2/RTMW37kpts_v2_20fps-finetune-pitch-correct-5-angleLossV2-only" \

# --note "OG loss" \
# --config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps_pitch_correct.yaml \
# --wandb_name "RTMW37kpts_v2_20fps-finetune-pitch-correct-1-OG" \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/RTMW37kpts_v2_20fps-finetune-pitch-correct-1-OG" \

# --note "bone loss V2" \
# --config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps_pitch_correct.yaml \
# --wandb_name "RTMW37kpts_v2_20fps-finetune-pitch-correct-2-limbLoss" \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/RTMW37kpts_v2_20fps-finetune-pitch-correct-2-limbLoss" \
#
# --resume /scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/20fps-finetune-pitch-correct-9/latest_epoch.bin \

# --config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps.yaml \
# --wandb_name "37kpts_v2_20fps-finetune-normal-7" \
# --checkpoint "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/20fps-finetune-normal-7" \










#--discard_last_layer \

# --checkpoint checkpoint/pose3d/FT_RTM_VEHS_37kpts_v1 \
# --resume checkpoint/pose3d/FT_RTM_VEHS_37kpts_v1/latest_epoch.bin \



## finetune RTMPose24 - VEHS  (config change 4 location)
#python train.py \
#--config configs/pose3d/RTMPose_exp/MB_ft_VEHS_tilt_correct.yaml \
#--pretrained checkpoint/pose3d/MB_train_VEHSR3_3DPose/ \
#--test_set_keyword validate \
#--wandb_project "MotionBert_train_RTM2D" \
#--wandb_name "tilt_corrected" \
#--checkpoint checkpoint/pose3d/FT_RTM_VEHS_tilt_corrected \
#--selection latest_epoch.bin > output_slurm/train_RTM.out

#--checkpoint checkpoint/pose3d/FT_RTM_VEHS_config6 \
#--resume checkpoint/pose3d/FT_RTM_VEHS_config6/latest_epoch.bin \
#--selection latest_epoch.bin > output_slurm/train_RTM.out

# train scratch - VEHS
#python -u train.py \
#--config configs/pose3d/MB_train_VEHSR3.yaml \
#--test_set_keyword validate \
#--checkpoint checkpoint/pose3d/MB_train_VEHSR3_3DPose > output_slurm/train_2.out

# train scratch - H36M
#python -u train.py \
#--config configs/pose3d/MB_train_h36m.yaml \
#--test_set_keyword test \
#--wandb_name H36M_scratch_gt2d_train \
#--checkpoint checkpointpoint/pose3d/MB_train_H36M_gt2D_3DPose > output_slurm/train_h36m.out


# finetune 3D
#python train.py \
#--config configs/pose3d/MB_ft_VEHSR3_3DPose.yaml \
#--pretrained checkpoint/pose3d/FT_MB_release_MB_ft_h36m \
#--test_set_keyword validate \
#--checkpoint checkpoint/pose3d/MB_ft_VEHSR3_3DPose \
#--selection best_epoch.bin > output_slurm/train_2.out

# fine tune 6D
#python train.py \
#--config configs/pose3d/MB_ft_VEHSR3_6DPose.yaml \
#--pretrained checkpoint/pose3d/FT_MB_release_MB_ft_h36m \
#--test_set_keyword validate \
#--checkpoint checkpoint/pose3d/MB_ft_VEHSR3_6DPose \
#--selection best_epoch.bin > output_slurm/train_6D.out