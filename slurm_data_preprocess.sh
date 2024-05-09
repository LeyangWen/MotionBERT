#!/bin/bash
#SBATCH --job-name=VEHS_3D_dataset_preprocess
#SBATCH --output=output_slurm/preprocess_log.txt
#SBATCH --error=output_slurm/preprocess_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=00:30:00
#SBATCH --account=shdpm0
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

# VEHS - RTMPose24 - industry inference
python -u tools/convert_inference.py \
--dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_industry' \
--dt_file 'rtmpose_industry_no3d_j24_f20_s1_RTM2D.pkl' \
--test_set_keyword 'validate' \
--res_w 1920 \
--res_h 1080 \
--root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_industry' > output_slurm/preprocess_RTM6.out

# VEHS - RTMPose24 - VEHSR3
#python -u tools/convert_VEHSR3.py \
#--dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_VEHS_config6' \
#--dt_file 'VEHS_6D_downsample5_keep1_config6_modified_RTM2D.pkl' \
#--test_set_keyword 'validate' \
#--root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_VEHS_config6' > output_slurm/preprocess_RTM6.out


# H36M
#python -u tools/convert_h36m.py > output_slurm/preprocess_H36M.out

# VEHSR3
#python -u tools/convert_VEHSR3.py \
#--dt_root 'data/motion3d/MB3D_VEHS_25d/3DPose' \
#--dt_file 'VEHS_3D_downsample2_keep1.pkl' \
#--root_path 'data/motion3d/MB3D_VEHS_25d/3DPose' > output_slurm/preprocess.out

# VEHSR3 - 6D
#python -u tools/convert_VEHSR3.py \
#--dt_root '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/6DPose' \
#--dt_file 'VEHS_6D_downsample2_keep1.pkl' \
#--root_path '/nfs/turbo/coe-shdpm/leyang/VEHS_MB/6DPose' > output_slurm/preprocess_6D.out