import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time

import logging
from datetime import datetime

def setup_logger(rank):
    os.makedirs('./results', exist_ok=True)
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'./results.log_rank_{rank}_{current_time}.txt'
    pass

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def demo_model(rank, world_size):
    print(f"Running on rank {rank}. Using GPU {rank}")
    
    # Setup distributed environment
    setup(rank, world_size)
    
    # Dummy tensor to demonstrate GPU utilization
    tensor = torch.ones(1, device=rank)
    
    # Wrap the model in DDP
    model = torch.nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Loop to keep the process running longer
    for _ in range(5):  # This will run the loop 600 times
        tensor = tensor * 2
        # torch.cuda.synchronize(rank)  # Ensures that GPU operations are completed
        print(f"Rank {rank}, Tensor value: {tensor[0].item()}")
        time.sleep(0.2)  # Wait 1 second between iterations

    # Clean up the distributed environment
    cleanup()

def run_demo(world_size):
    mp.spawn(demo_model,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    print(f"Finished!")

if __name__ == "__main__":
    world_size = 4
    run_demo(world_size)