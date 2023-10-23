# Great Lake prompts

## ssh & login
```bash
ssh wenleyan@greatlakes.arc-ts.umich.edu
```

## Interactive session
### Load module


```bash
module purge  # clear all modules
#module --default avail  # list all available modules
#module keyword string  # search for modules with keyword

module load python3.10-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
module list

```


## Create batch file for Slurm

### Batch file template

```bash
##### Slurm preamble
#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1g
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --account=training  # account to charge
#SBATCH --partition=gpu # debug
##### END preamble

my_job_header
module load python3.10-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
module list
cd
activate 
python xxx.py
```

### Other useful options

```bash
#SBATCH --no-requeue  # do not requeue job if node fails
#SBATCH --test-only  # test job without running
```

### Submit job    
```bash
sbatch batch_file.sh
```

### Check job status

```bash
squeue -u wenleyan
```

### Cancel a job

```bash
scancel job_id
```

### useful commands

```bash
my_account wenleyan  # list resources
my_job_estimate xxx.sbat #-c 4 -n 1 -m 4g - p gpu -t 1:00:00
my_job_statistics jobid
my_account_billing -v -a wenleyan 
sinfo  # node status by partition

```


# Resources

### [Great Lake Cheat Sheet](https://arc.umich.edu/wp-content/uploads/sites/4/2020/05/Great-Lakes-Cheat-Sheet.pdf)
### [Online Dashboard](https://greatlakes.arc-ts.umich.edu/pun/sys/dashboard)

### [Training Videos](https://www.mivideo.it.umich.edu/channel/ARC-TS%2BTraining/181860561/)