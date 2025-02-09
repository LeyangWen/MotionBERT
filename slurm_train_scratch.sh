#!/bin/bash
#SBATCH --job-name=MB_train_RTM_VEHS
#SBATCH --output=output_slurm/train_log_3D.txt
#SBATCH --error=output_slurm/train_error_3D.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10g
#SBATCH --gres=gpu:6
#SBATCH --time=30:00:00
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
module list

#conda activate motionbert

# # Rokoko - Hand-21
# python train.py \
# --config configs/pose3d/hand/MB_train_Rokoko.yaml \
# --test_set_keyword validate \
# --wandb_project "MotionBert_train_Hand" \
# --wandb_name "Rokoko_2" \
# --checkpoint checkpoint/pose3d/MB_train_Rokoko_hand_21 > output_slurm/train_hand.out


# finetune RTMPose37 - VEHS
python train.py \
--config configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS.yaml \
--pretrained checkpoint/pose3d/MB_ft_VEHSR3_6DPose/ \
--test_set_keyword validate \
--wandb_project "MotionBert_train_RTM2D" \
--wandb_name "37kpts_v1_try" \
--checkpoint checkpoint/pose3d/FT_RTM_VEHS_37kpts_v1 \
--selection best_epoch.bin \
--discard_last_layer \

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