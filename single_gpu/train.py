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

from model import pyramidnet
import argparse
# from tensorboardX import SummaryWriter

import logging
from pathlib import Path

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--num_worker', type=int, default=4, help='')
args = parser.parse_args()


def main():
    # best_acc = 0

    # Set up logger
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root='../data', train=True, download=True, 
                            transform=transforms_train)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_worker)

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    logging.info('==> Making model..')

    net = pyramidnet()
    net = net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f'The number of parameters of model is {num_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, 
                          momentum=0.9, weight_decay=1e-4)
    
    train(net, criterion, optimizer, train_loader, device)
            

def train(net, criterion, optimizer, train_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        
        inputs = inputs.to(device)
        targets = targets.to(device)
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
        
        if batch_idx % 20 == 0:
            logging.info('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_idx, len(train_loader), train_loss/(batch_idx+1), acc, batch_time))
    
    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    logging.info("Training time {}".format(elapse_time))
    
if __name__=='__main__':
    main()