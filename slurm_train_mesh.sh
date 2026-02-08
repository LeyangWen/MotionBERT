#!/bin/bash -l
#SBATCH --job-name=MB_train_mesh
#SBATCH --output=output_slurm/train_mesh_log.txt
#SBATCH --error=output_slurm/train_mesh_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --gres=gpu:3
#SBATCH --time=100:00:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu
##### END preamble
##### Run in MotionBert dir

### gpu_mig40 -->1000GB, but can only use one
my_job_header
module load python3.10-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
module load python/3.10.4
module load pytorch/2.0.1
module list

conda activate motionbert

echo "test"
# rm -rf checkpoint/mesh/MB_train_VEHS66kpts_try1

# for 6D
# python train_mesh.py \
# --config configs/mesh/MB_train_VEHS_6D.yaml \
# --pretrained checkpoint/mesh/FT_Mb_release_MB_ft_pw3d/ \
# --selection best_epoch.bin \
# --checkpoint checkpoint/mesh/MB_train_VEHS66kpts \
# --test_set_keyword validate \
# --wandb_project "MotionBert_train_mesh" \
# --wandb_name "gt2d_66kpts_try2_section2" \
# --resume checkpoint/mesh/MB_train_VEHS66kpts/latest_epoch.bin \

# TODO: To train 37 kpt model (/nfs/turbo/coe-shdpm/leyang/VEHS_MB/mesh/RTM2D_37kpts_SMPL/), change file loc, change J_regressor in multiple places

## for 3D
# python train_mesh.py \
# --config configs/mesh/RTM2D_train_17kpts_3D.yaml \
# --pretrained checkpoint/mesh/FT_Mb_release_MB_ft_pw3d/ \
# --selection best_epoch.bin \
# --checkpoint /scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/mesh_compare/SMPL_RTM17kpts_V3 \
# --test_set_keyword validate \
# --wandb_project "MotionBert_train_mesh" \
# --wandb_name "SMPL_RTM17kpts_V3" \
# --note "RTM2D 17kpts, SMPL model, 200 epochs" \

# --resume /scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/mesh_compare/SMPL_RTM17kpts_V1/latest_epoch.bin \


## for 6D
python train_mesh.py \
--config configs/mesh/RTM2D_train_37kpts_6D.yaml \
--pretrained checkpoint/mesh/FT_Mb_release_MB_ft_pw3d/ \
--selection best_epoch.bin \
--checkpoint /scratch/shdpm_root/shdpm0/wenleyan/MB_checkpoints/mesh_compare/SMPL_RTM37kpts_V1 \
--test_set_keyword validate \
--wandb_project "MotionBert_train_mesh" \
--wandb_name "SMPL_RTM37kpts_V1" \
--note "RTM2D 37kpts, SMPL model, 200 epochs" \
