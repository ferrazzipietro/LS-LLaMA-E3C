#!/bin/bash -l
### job name
#SBATCH --job-name=model_predictions_generation

### 
#SBATCH --mail-user=ferrazzi@math.unipd.it

### Standard output and standard error for our job
#SBATCH --error=model_predictions_generation.err
#SBATCH --output=model_predictions_generation.out

### queue/partition choosed
#SBATCH --partition=testing
### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=32G

### Time limit for our job (ten minutes here: HH:MM:SS)
#SBATCH --time=96:00:00

### GPU request
#SBATCH --gres=gpu
#SBATCH --constraint=A6000
####SBATCH --constraint=vgpu7-0

### Some useful informative commands
echo -n 'Date: '
date
echo -n 'Directory: '
pwd
echo -n 'This job will be executed on th following nodes: '
echo ${SLURM_NODELIST}
echo

SHELL=/bin/bash

### Jobs execution commands
conda activate gpu_venv_lates
which python
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python  model_predictions_generation.py
echo -n 'Date: '
date
