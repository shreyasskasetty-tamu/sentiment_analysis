#!/bin/bash
#SBATCH --time=0-05:00:00           # max job run time dd-hh:mm:ss
#SBATCH --ntasks-per-node=1         # tasks (commands) per compute node
#SBATCH --cpus-per-task=8          # CPUs (threads) per command
#SBATCH --mem=360G                  # total memory per node
#SBATCH --gres=gpu:t40          # request 2 GPUs
#SBATCH --output=stdout.%x.%j       # save stdout to file
#SBATCH --error=stderr.%x.%j        # save stderr to file

source deactivate 
source activate menv
python train.py -m roberta