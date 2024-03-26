#!/bin/bash
#SBATCH --job-name=MB_train_h36m_gt_scratch
#SBATCH --output=output_slurm/train_log_3D.txt
#SBATCH --error=output_slurm/train_error_3D.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40g
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
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

echo "spgpu test"

# train scratch - VEHS
#python -u train.py \
#--config configs/pose3d/MB_train_VEHSR3.yaml \
#--test_set_keyword validate \
#--checkpoint checkpoint/pose3d/MB_train_VEHSR3_3DPose > output_slurm/train_2.out

# train scratch - H36M
python -u train.py \
--config configs/pose3d/MB_train_h36m.yaml \
--test_set_keyword test \
--wandb_name H36M_scratch_gt2d_train \
--checkpoint checkpoint/pose3d/MB_train_H36M_gt2D_3DPose > output_slurm/train_h36m.out


# fine tune
#python train.py \
#--config configs/pose3d/MB_ft_VEHSR3_3DPose.yaml \
#--pretrained checkpoint/pose3d/FT_MB_release_MB_ft_h36m \
#--test_set_keyword validate \
#--checkpoint checkpoint/pose3d/MB_ft_VEHSR3_3DPose \
#--selection best_epoch.bin > output_slurm/train_2.out

# fine tune 6D
#python train.py \
#--config configs/pose3d/MB_ft_VEHSR3_6DPose.yaml \
#--pretrained checkpoint/pose3d/FT_MB_release_MB_ft_h36m \
#--test_set_keyword validate \
#--checkpoint checkpoint/pose3d/MB_ft_VEHSR3_6DPose \
#--selection best_epoch.bin > output_slurm/train_6D.out