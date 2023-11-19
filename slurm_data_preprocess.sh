#!/bin/bash
#SBATCH --job-name=MotionBert_train_scratch_VEHS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --time=00:30:00
#SBATCH --account=engin1
#SBATCH --partition=debug
##### END preamble
##### Run in MotionBert dir

my_job_header
module load python3.10-anaconda
module load python/3.10.4
module list

conda activate motionbert
python tools/convert_VEHSR3.py \
--dt_root 'data/motion3d/MB3D_VEHS_R3/3DPose' \
--dt_file 'VEHS_3D_downsample1_keep1.pkl' \
--root_path 'data/motion3d/MB3D_VEHS_R3/3DPose
