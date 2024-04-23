#!/bin/bash -l
### job name
#SBATCH --job-name=train_ls_llama

### 
#SBATCH --mail-user=ferrazzi@math.unipd.it

### Standard output and standard error for our job
#SBATCH --error=train_ls_llama.err
#SBATCH --output=train_ls_llama.out

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
####SBATCH --constraint=A6000
####SBATCH --constraint=vgpu1-1

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
conda activate gpu_venv_lates

# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python --version
# !export TORCH_USE_CUDA_DSA=1
python train_ls_llama.py #llama_token_clf_copy.py e3c 7b  #  llama_token_clf.py wnut_17 7b #
