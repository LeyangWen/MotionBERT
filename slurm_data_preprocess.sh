#!/bin/bash
#SBATCH --job-name=VEHS_3D_dataset_preprocess
#SBATCH --output=output_slurm/preprocess_log.txt
#SBATCH --error=output_slurm/preprocess_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --time=00:30:00
#SBATCH --account=engin1
#SBATCH --partition=standard
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

#source /home/wenleyan/.conda/envs/motionbert/bin/activate motionbert
#conda activate motionbert

# H36M
python -u tools/convert_h36m.py > output_slurm/preprocess_H36M.out

# VEHSR3
#python -u tools/convert_VEHSR3.py \
#--dt_root 'data/motion3d/MB3D_VEHS_R3/3DPose' \
#--dt_file 'VEHS_3D_downsample1_keep1.pkl' \
#--root_path 'data/motion3d/MB3D_VEHS_R3/3DPose' > output_slurm/preprocess.out