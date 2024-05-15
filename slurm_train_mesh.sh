#!/bin/bash
#SBATCH --job-name=MB_train_RTM_VEHS
#SBATCH --output=output_slurm/train_log_3D.txt
#SBATCH --error=output_slurm/train_error_3D.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10g
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

echo "test"


python train_mesh.py \
--config configs/mesh/MB_train_VEHSR3.yaml \
--checkpoint checkpoint/mesh/MB_train_VEHSR3

# finetune RTMPose24 - VEHS  (config change 4 location)
python train.py \
--config configs/pose3d/RTMPose_exp/MB_ft_VEHS_config2.yaml \
--pretrained checkpoint/pose3d/MB_train_VEHSR3_3DPose/ \
--test_set_keyword validate \
--checkpoint checkpoint/pose3d/FT_RTM_VEHS_config2 \
--wandb_project "MotionBert_train_RTM2D" \
--wandb_name "config2_unprocessed_gt2d_false" \
--resume checkpoint/pose3d/FT_RTM_VEHS_config2/latest_epoch.bin \
--selection latest_epoch.bin > output_slurm/train_RTM.out

