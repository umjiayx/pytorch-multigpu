import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from model import pyramidnet
import argparse
# from tensorboardX import SummaryWriter

import logging
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP


# gpu_devices = ','.join([str(id) for id in args.gpu_devices])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
# os.environ['MASTER_PORT'] = '5678'


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def args_parser():
    parser = argparse.ArgumentParser(description='cifar10 classification models')
    parser.add_argument('--lr', default=0.1, help='')
    parser.add_argument('--resume', default=None, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    return parser


def main():
    '''
    # set up logging
    log_folder = Path('./results')
    log_folder.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = log_folder / f"{current_time}.txt"

    level = getattr(logging, "INFO", None)
    if not isinstance(level, int):
        raise ValueError(f"level {level} not supported")
    
    logger = logging.getLogger()
    logger.setLevel(level=level)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    handeler1 = logging.StreamHandler()
    handeler2 = logging.FileHandler(log_path, mode='w')

    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handeler1.setFormatter(formatter)
    handeler2.setFormatter(formatter)

    logger.addHandler(handeler1)
    logger.addHandler(handeler2)    
    
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # logging.info(f"Using device: {device}")    
    
    '''

    parser = args_parser()
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count() # 4
    args.world_size = ngpus_per_node * args.world_size # 4 * 1 = 4
    
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    print("Finished!")


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    setup(gpu, ngpus_per_node)
    print(f"Using GPU: {gpu} for training")
    
    # Update rank for each process
    args.rank = args.rank * ngpus_per_node + gpu    # 0 * 4 + r

    print(" ==> Making model...")
    net = pyramidnet()

    # Move the model to the GPU
    torch.cuda.set_device(args.gpu)
    net.cuda(args.gpu)
    net = DDP(net, device_ids=[args.gpu]) # Wrap the model in DDP, with device_ids set to [gpu]

    # Adjust the batch size for DDP
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    
    '''
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'The number of parameters of model is {num_params}')
    # logging.info(f'The number of parameters of model is {num_params}')    
    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    '''

    print(" ==> Preparing data...")
    #logging.info('==> Preparing data..')
    
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root='../data', train=True, download=True, transform=transforms_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None), num_workers=args.num_workers, 
                              sampler=train_sampler)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0.9, weight_decay=1e-4)

    # Prepare validation data
    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_val = CIFAR10(root='../data', train=False, download=True, transform=transforms_val)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train(net, criterion, optimizer, train_loader, val_loader, args.gpu)

    cleanup()


def validate(net, criterion, val_loader, device):
    net.eval()  # Set the model to evaluation mode
    
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_acc = 100 * correct / total
    print(f"    Validation Loss: {val_loss / (batch_idx+1):.3f} | Validation Accuracy: {val_acc:.3f}")
    net.train()  # Switch back to training mode


def train(net, criterion, optimizer, train_loader, val_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100 * correct / total
        
        batch_time = time.time() - start
        
        if batch_idx % 10 == 0:
            # logging.info('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(batch_idx, len(train_loader), train_loss/(batch_idx+1), acc, batch_time))
            print('Iteration: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(batch_idx, len(train_loader), train_loss/(batch_idx+1), acc, batch_time))
            
            validate(net, criterion, val_loader, device)
    
    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))
    #logging.info("Training time {}".format(elapse_time))
    

if __name__=='__main__':
    main()