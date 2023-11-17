#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1g
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:20
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
python tools/convert_VEHSR3.py \
--dt_root 'data/motion3d/MB3D_VEHS_R3_small/3DPose' \
--dt_file 'VEHS_3D_downsample_4.pkl_small.pkl' \
--root_path 'data/motion3d/MB3D_VEHS_R3_small/3DPose'


python train.py \
--config configs/pose3d/MB_train_h36m.yaml \
--checkpoint checkpoint/pose3d/MB_train_h36m
