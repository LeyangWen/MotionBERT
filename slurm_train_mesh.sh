#!/bin/bash
#SBATCH --job-name=MB_train_mesh
#SBATCH --output=output_slurm/train_log_mesh.txt
#SBATCH --error=output_slurm/train_error_mesh.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu
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

echo "test"


python train_mesh.py \
--config configs/mesh/MB_train_VEHS_3D.yaml \
--pretrained checkpoint/mesh/FT_Mb_release_MB_ft_pw3d/ \
--selection best_epoch.bin \
--checkpoint checkpoint/mesh/MB_train_VEHSR3 \
--test_set_keyword validate \
--wandb_project "MotionBert_train_mesh" \
--wandb_name "gt2d_17kpts" > output_slurm/train_mesh.out


#--resume checkpoint/pose3d/FT_RTM_VEHS_config2/latest_epoch.bin \


