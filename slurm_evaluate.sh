#!/bin/bash -l
#SBATCH --job-name=MB_eval
#SBATCH --output=output_slurm/eval_pose_log.txt
#SBATCH --error=output_slurm/eval_pose_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --partition=debug
#SBATCH --time=00:10:00
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
module list

#conda activate motionbert

echo "cpu-2, gpu-1, mem-20"

# Dataset
#config_file="configs/pose3d/MB_train_h36m.yaml"
#config_file="configs/pose3d/MB_ft_h36m.yaml"
#config_file="configs/pose3d/MB_train_VEHSR3.yaml"
#config_file="configs/pose3d/MB_ft_VEHSR3_6DPose.yaml"
config_file="configs/pose3d/MB_ft_VEHSR3_3DPose.yaml"
#config_file="configs/pose3d/RTMPose_exp/MB_ft_VEHS_config6.yaml"
#config_file="configs/pose3d/RTMPose_exp/MB_ft_inference.yaml"
#config_file="configs/pose3d/RTMPose_exp/MB_ft_VEHS_tilt_correct.yaml"
#config_file="configs/pose3d/hand/MB_train_Rokoko.yaml"
#config_file="configs/pose3d/hand/MB_infer_lab_RTMinput.yaml"  # infer should use infer code

# Checkpoint
#checkpoint_bin="checkpoint/pose3d/MB_train_h36m/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/MB_ft_VEHSR3_6DPose/best_epoch.bin"
checkpoint_bin="checkpoint/pose3d/MB_ft_VEHSR3_3DPose/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_config6/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_config6_GT2d_true/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_tilt_corrected/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/MB_train_Rokoko_hand_21/latest_epoch.bin"


echo "config_file: $config_file"
echo "checkpoint_bin: $checkpoint_bin"

# evaluate - train.py [pose3d]
python -u train.py \
--config "$config_file" \
--wandb_project "MotionBert_eval" \
--wandb_name "cpt_VEHS3D_ft_dataset_VEHS3D"  \
--note "" \
--out_path "experiment/VEHS-7M_17" \
--test_set_keyword test \
--evaluate "$checkpoint_bin" \

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
