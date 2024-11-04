import os
import random
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np

from models import myresnet, vit
from utils import get_DataLoader
from utils.utils_scheduler import WarmupCosineSchedule
from mydataset import Food

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

parser = argparse.ArgumentParser(description='Training RGB-D Fusion Network for Nutrition Estimation')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--wd', default=0.9, type=float, help='Weight decay')
parser.add_argument('--min_lr', default=2e-4, type=float, help='Minimum learning rate')
parser.add_argument('--dataset', choices=["nutrition_rgbd", "nutrition_rgb"], default='nutrition_rgbd', help='Dataset choice')
parser.add_argument('--b', type=int, default=8, help='Batch size')
parser.add_argument('--resume', '-r', type=str, help='Resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained network')
parser.add_argument('--model', default='resnet101', type=str, help='Model architecture')
parser.add_argument('--data_root', type=str, default='path/to/dataset', help='Dataset root path')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

def initialize_model(args):
    if args.model == 'resnet101':
        net = myresnet.resnet101(rgbd=args.rgbd)
        pretrained_dict = torch.load("path/to/pretrained/model.pth")
        model_dict = net.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        net.load_state_dict(model_dict)
    elif args.model == 'vit_base':
        net = vit.vit_base_patch16_224(pretrained=args.pretrained)
    else:
        raise ValueError("Unsupported model type")
    
    return net.to('cuda' if torch.cuda.is_available() else 'cpu')

def configure_optimizer(net):
    optimizer = optim.Adam(
        [{'params': net.parameters(), 'lr': args.lr, 'weight_decay': args.wd}],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    return optimizer, scheduler

def calculate_loss(outputs, labels):
    criterion = nn.L1Loss()
    return criterion(outputs, labels)

def train(epoch, net, trainloader, optimizer):
    net.train()
    running_loss = 0.0
    epoch_iterator = tqdm(trainloader, desc=f"Training Epoch {epoch}", dynamic_ncols=True)
    
    for inputs, labels in epoch_iterator:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = calculate_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch}] - Training Loss: {running_loss / len(trainloader)}")

def test(epoch, net, testloader):
    net.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = calculate_loss(outputs, labels)
            test_loss += loss.item()
    print(f"Epoch [{epoch}] - Test Loss: {test_loss / len(testloader)}")

if __name__ == "__main__":
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net = initialize_model(args)
    optimizer, scheduler = configure_optimizer(net)
    trainloader, testloader = get_DataLoader(args)
    
    for epoch in range(300):
        train(epoch, net, trainloader, optimizer)
        test(epoch, net, testloader)
        scheduler.step()