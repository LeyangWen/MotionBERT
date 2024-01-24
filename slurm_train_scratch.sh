#!/bin/bash
#SBATCH --job-name=MB_train_real
#SBATCH --output=output_slurm/train_log_1.txt
#SBATCH --error=output_slurm/train_error_1.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20g
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --account=shdpm0
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

echo "cpu-2, gpu-2, mem-20"

python -u train.py \
--config configs/pose3d/MB_train_VEHSR3.yaml \
--test_set_keyword validate \
--checkpoint checkpoint/pose3d/MB_train_VEHSR3_3DPose > output_slurm/train_1.out
