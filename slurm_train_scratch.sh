#!/bin/bash
#SBATCH --job-name=MotionBert_train_scratch_VEHS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20g
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
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

conda activate motionbert
#python tools/convert_VEHSR3.py \
#--dt_root 'data/motion3d/MB3D_VEHS_R3_small/3DPose' \
#--dt_file 'VEHS_3D_downsample_4.pkl_small.pkl' \
#--root_path 'data/motion3d/MB3D_VEHS_R3_small/3DPose'


python -u train.py \
--config configs/pose3d/MB_train_VEHSR3.yaml \
--checkpoint checkpoint/pose3d/MB_train_VEHSR3_3DPose > train_log.out
