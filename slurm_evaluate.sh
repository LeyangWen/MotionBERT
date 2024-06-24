#!/bin/bash
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
module list

#conda activate motionbert

echo "cpu-2, gpu-1, mem-20"

# Dataset
#config_file="configs/pose3d/MB_train_h36m.yaml"
#config_file="configs/pose3d/MB_train_VEHSR3.yaml"
#config_file="configs/pose3d/MB_ft_VEHSR3_6DPose.yaml"
#config_file="configs/pose3d/RTMPose_exp/MB_ft_VEHS_config6.yaml"
#config_file="configs/pose3d/RTMPose_exp/MB_ft_inference.yaml"
config_file="configs/pose3d/RTMPose_exp/MB_ft_VEHS_tilt_correct.yaml"

# Checkpoint
#checkpoint_bin="checkpoint/pose3d/MB_train_h36m/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/MB_ft_VEHSR3_6DPose/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_config6/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_config6_GT2d_true/best_epoch.bin"
checkpoint_bin="checkpoint/pose3d/FT_RTM_VEHS_tilt_corrected"



echo "config_file: $config_file"
echo "checkpoint_bin: $checkpoint_bin"

# evaluate - train.py
python -u train.py \
--config "$config_file" \
--wandb_project "MotionBert_eval" \
--wandb_name "RTM_input_MB_ft_validate_tilt_corrected" \
--note "gt_2d false" \
--out_path "experiment/RTM2D_ft/validate_RTM2d_tilt_corrected" \
--test_set_keyword validate \
--evaluate "$checkpoint_bin" \
#--save_trace \

# inference only
#python -u infer3d_train.py \
#--config "$config_file" \
#--wandb_project "MotionBert_eval" \
#--wandb_name "RTM_input_MB_ft_validate" \
#--note "save pose output" \
#--out_path "experiment/RTM2D_ft/config6" \
#--test_set_keyword validate \
#--evaluate "$checkpoint_bin" > "output_slurm/eval_${SLURM_JOB_ID}_output.out"


# (config change 4 location)