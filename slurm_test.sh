#!/bin/bash
#SBATCH --job-name=Test
#SBATCH --output=output_slurm/test_log.txt
#SBATCH --error=output_slurm/test_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1g
#SBATCH --time=00:02:00
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

CONDA_ROOT="/home/wenleyan/.conda/envs/motionbert/"

. $CONDA_ROOT/etc/profile.d/conda.sh
conda activate motionbert

