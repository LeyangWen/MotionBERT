#!/bin/bash
#SBATCH --job-name=MB_mesh_eval
#SBATCH --output=output_slurm/eval_mesh_log_spgpu.txt
#SBATCH --error=output_slurm/eval_mesh_error_spgpu.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180g
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu
##SBATCH --partition=debug
#SBATCH --time=02:00:00
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

# Dataset -- GT2D
# config_file="configs/mesh/MB_train_VEHS_3D.yaml"  # 3D kpts 17
#config_file="configs/mesh/MB_train_VEHS_6D.yaml"  # 6D kpts 66 v2

#config_file="configs/mesh/MB_ft_h36m.yaml"  # H36M inference 2D

# Dataset -- RTM2D

#config_file="configs/mesh/MB_train_VEHS_6D.yaml"  # 6D kpts 66 v2



# Checkpoint
# checkpoint_bin="checkpoint/mesh/MB_train_VEHSR3/latest_epoch.bin"  # 3D kpts 17
#checkpoint_bin="checkpoint/mesh/MB_train_VEHS66kpts/latest_epoch.bin"  # 6D kpts 66 v2
#checkpoint_bin="checkpoint/mesh/FT_Mb_release_MB_ft_pw3d/best_epoch.bin"

# config_file="configs/mesh/RTM2D_train_17kpts_3D.yaml"  # 3D kpts 17
# checkpoint_bin="/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/mesh_compare/SMPL_RTM17kpts_V1/epoch_99.bin" # RTM2D 17kpts SMPL


config_file="configs/mesh/RTM2D_train_37kpts_6D.yaml"
checkpoint_bin="/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/mesh_compare/SMPL_RTM37kpts_temp/latest_epoch.bin" # RTM2D 37kpts SMPL



echo "config_file: $config_file"
echo "checkpoint_bin: $checkpoint_bin"

# evaluate - train.py
python -u train_mesh.py \
--config "$config_file" \
--evaluate "$checkpoint_bin" \
--test_set_keyword validate \
--wandb_project "MotionBert_mesh_eval" \
--wandb_name "SMPL_RTM37kpts_V1" \
--note "RTM2D 37kpts, SMPL model, temp" \
--fps 20 \
--out_path "/scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/mesh_compare/SMPL_RTM37kpts_temp/" \

#--wandb_name "GT_input_MB_mesh_test_66_VEHS7M" \
#--out_path "/scratch/shdpm_root/shdpm0/wenleyan/66kpts" \




# if too big, save to scratch: /scratch/shdpm_root/shdpm0/wenleyan/

