#!/bin/bash
#SBATCH --job-name=test_distributed   # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=4                   # Number of tasks (1 per GPU)
#SBATCH --gpus=4                     # Number of GPUs
#SBATCH --cpus-per-task=6            # Number of CPU cores per task
#SBATCH --account=qingqu
#SBATCH --partition=qingqu           # Use the GPU partition
#SBATCH --time=00:30:00              # Time limit
#SBATCH --mem=32G                    # Memory per node
#SBATCH --output=output_%j.log       # Output log file
#SBATCH --error=error_%j.log         # Error log file


# /scratch/qingqu_root/qingqu/jiayx/test1/bin/python test_dist_3.py 10 5

/scratch/qingqu_root/qingqu/jiayx/test1/bin/python test_dist_2.py