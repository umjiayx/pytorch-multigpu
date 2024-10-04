#!/bin/bash
#SBATCH --job-name=test_dist   # Job name
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

# Load necessary modules
module load cuda/11.7
module load python/3.9

# Print the environment details to debug
echo "Conda environment path: /scratch/qingqu_root/qingqu/jiayx/test1"
echo "Activating environment..."

# Directly run Python from the Conda environment
/scratch/qingqu_root/qingqu/jiayx/test1/bin/python -c "import torch; print(torch.__version__)"

# Run the Python script with PyTorch from the same Python interpreter
/scratch/qingqu_root/qingqu/jiayx/test1/bin/python train.py --gpu_device 0 1 2 3 --batch_size 768
