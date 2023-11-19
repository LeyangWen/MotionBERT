#!/bin/bash
#SBATCH --job-name=MB_train
#SBATCH --output=output_slurm/train_log_2.txt
#SBATCH --error=output_slurm/train_error_2.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30g
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --account=engin1
#SBATCH --partition=gpu
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

echo "cpu-8, gpu-2, mem-30"

python -u train.py \
--config configs/pose3d/MB_train_VEHSR3.yaml \
--checkpoint checkpoint/pose3d/MB_train_VEHSR3_3DPose > output_slurm/train_2.out
