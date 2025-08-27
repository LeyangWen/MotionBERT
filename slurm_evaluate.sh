#!/bin/bash -l
#SBATCH --job-name=MB_eval
#SBATCH --output=output_slurm/eval_log.txt
#SBATCH --error=output_slurm/eval_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --account=shdpm0
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

echo "cpu-2, gpu-1, mem-20"

# Dataset
#config_file="configs/pose3d/MB_train_h36m.yaml"
#config_file="configs/pose3d/MB_ft_h36m.yaml"
#config_file="configs/pose3d/MB_train_VEHSR3.yaml"
# config_file="configs/pose3d/MB_ft_VEHSR3_6DPose.yaml"
# config_file="configs/pose3d/MB_ft_VEHSR3_3DPose.yaml"
#config_file="configs/pose3d/RTMPose_exp/MB_ft_VEHS_config6.yaml"
#config_file="configs/pose3d/RTMPose_exp/MB_ft_inference.yaml"
#config_file="configs/pose3d/RTMPose_exp/MB_ft_VEHS_tilt_correct.yaml"
#config_file="configs/pose3d/hand/MB_train_Rokoko.yaml"
#config_file="configs/pose3d/hand/MB_infer_lab_RTMinput.yaml"  # infer should use infer code
# config_file="configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS.yaml"
# config_file="configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps.yaml"
config_file="configs/pose3d/RTMPose_exp/37kpts_v1/MB_ft_VEHS_20fps_pitch_correct.yaml"
# 
# Checkpoint
# checkpoint_bin="/nfs/turbo/coe-shdpm/leyang/MB_checkpoints/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"  # from h36m MB website
# checkpoint_bin="/nfs/turbo/coe-shdpm/leyang/MB_checkpoints/pose3d/MB_train_H36M_gt2D_3DPose/best_epoch.bin"  # from h36m gt2d custom trained
# checkpoint_bin="checkpoint/pose3d/MB_train_h36m/best_epoch.bin" 
# checkpoint_bin="checkpoint/pose3d/MB_ft_VEHSR3_6DPose/best_epoch.bin"
# checkpoint_bin="checkpoint/pose3d/MB_ft_VEHSR3_3DPose/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_config6/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_config6_GT2d_true/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_tilt_corrected/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/MB_train_Rokoko_hand_21/latest_epoch.bin"
# checkpoint_bin="/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/exp6/best_epoch.bin"
# checkpoint_bin="/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/20fps-pretrain-normal-oneCam-1/best_epoch.bin"
# checkpoint_bin="/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/20fps-pretrain-normal-1/best_epoch.bin"

### RTMW
# checkpoint_bin="/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/RTMW37kpts_v2_20fps-finetune-pitch-correct-1-OG/best_epoch.bin" # RTMWV5 2-b 20fps, og loss
# checkpoint_bin="/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/RTMW37kpts_v2_20fps-finetune-pitch-correct-2-limbLoss/best_epoch.bin" # RTMWV5 2-b 20fps, + limb loss
checkpoint_bin="/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/RTMW37kpts_v2_20fps-finetune-pitch-correct-3-angleLoss/best_epoch.bin" # RTMWV5 2-b 20fps, og loss

echo "config_file: $config_file"
echo "checkpoint_bin: $checkpoint_bin"

# evaluate - train.py [pose3d]
python -u train.py \
--config "$config_file" \
--wandb_project "MotionBert_eval" \
--wandb_name "cpt_RTMWV5-2b-OGLoss_data_VEHS6D_validate-1920x1200"  \
--note "gt2d_False-20fps-1920x1200" \
--out_path "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/RTMW/RTMW37kpts_v2_20fps-finetune-pitch-correct-3-angleLoss/VEHS7M-Validate" \
--test_set_keyword validate \
--evaluate "$checkpoint_bin" \

    
# 
# --out_path "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/FT_MB_release_MB_ft_h36m/VEHS7M-test-1920x1200" \

#--save_trace \

## evaluate - train.py [hand]
#python -u train.py \
#--config "$config_file" \
#--wandb_project "MotionBert_eval_Hand" \
#--wandb_name "Rokoko_2"  \
#--note "" \
#--out_path "experiment/handPose/Rokoko_2/right" \
#--test_set_keyword test \
#--evaluate "$checkpoint_bin" \

#{'L': 'validate', 'R': 'test'}
