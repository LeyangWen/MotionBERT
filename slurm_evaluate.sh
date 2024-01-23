#!/bin/bash
#SBATCH --job-name=MB_eval
#SBATCH --output=output_slurm/eval_log_${SLURM_JOB_ID}.txt
#SBATCH --error=output_slurm/eval_error_${SLURM_JOB_ID}.txt
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
#config_file="configs/pose3d/MB_train_VEHSR3.yaml"
config_file="configs/pose3d/MB_train_h36m.yaml"

# Checkpoint
#checkpoint_bin="checkpoint/pose3d/MB_train_VEHSR3_3DPose/latest_epoch.bin"
checkpoint_bin="checkpoint/pose3d/MB_train_h36m/best_epoch.bin"
#checkpoint_bin="checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"

echo "config_file: $config_file"
echo "checkpoint_bin: $checkpoint_bin"
python -u train.py \
--config "$config_file" \
--evaluate "$checkpoint_bin" > "output_slurm/eval_${SLURM_JOB_ID}_output.out"
