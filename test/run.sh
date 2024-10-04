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

# Load necessary modules
module load cuda/11.7
module load python/3.9

# source activate /scratch/qingqu_root/qingqu/jiayx/test1

# Print the environment details to debug
echo "Conda environment path: /scratch/qingqu_root/qingqu/jiayx/test1"
echo "Activating environment..."

# Directly run Python from the Conda environment
/scratch/qingqu_root/qingqu/jiayx/test1/bin/python -c "import torch; print(torch.__version__)"

# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=$((10000 + RANDOM % 55536))

# Run the Python script with PyTorch from the same Python interpreter

# srun /scratch/qingqu_root/qingqu/jiayx/test1/bin/python test_distributed.py

# /scratch/qingqu_root/qingqu/jiayx/test1/bin/python test_distributed.py

# /scratch/qingqu_root/qingqu/jiayx/test1/bin/python -m torch.distributed.run  --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 test_distributed.py

# /scratch/qingqu_root/qingqu/jiayx/test1/bin/python test_dist_3.py 10 5

/scratch/qingqu_root/qingqu/jiayx/test1/bin/python test_dist_2.py