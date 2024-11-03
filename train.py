# Required Libraries and Imports
import os
import argparse
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models import myresnet, vit, T2TNutrition, ViTNutrition, Inception3, Inception3_concat
from timm.models import create_model
from utils.utils import progress_bar, load_for_transfer_learning, logtxt, check_dirs
from utils_data import get_DataLoader
from utils.utils_scheduler import WarmupCosineSchedule
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from mydataset import Food

# Argument Parsing and Settings
parser = argparse.ArgumentParser(description='PyTorch Nutrition Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=0.9, type=float, help='weight decay')
parser.add_argument('--min_lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', choices=["nutrition_rgbd", "nutrition_rgb", "food101"], default='cifar10')
parser.add_argument('--b', type=int, default=8, help='batch size')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained model if available')
parser.add_argument('--num_classes', type=int, default=1024, metavar='N', help='number of label classes')
parser.add_argument('--model', default='T2t_vit_t_14', type=str, metavar='MODEL', help='Model name to train')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate')
parser.add_argument('--img_size', type=int, default=224, metavar='N', help='Image patch size')
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--data_root', type=str, default="/path/to/data", help="Dataset root")
parser.add_argument('--run_name', type=str, default="run_name")
parser.add_argument('--print_freq', type=int, default=200, help="Log frequency")

args = parser.parse_args()

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# Prepare device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_loss = float('inf')
global_step = 0

# Model Initialization
def initialize_model(args):
    if args.model == 'resnet101':
        model = myresnet.resnet101()
    elif args.model == 'inceptionv3':
        model = Inception3(aux_logits=False, transform_input=False)
    elif 'vit_base' in args.model:
        model = create_model(args.model, pretrained=args.pretrained, img_size=args.img_size)
    else:
        raise ValueError("Unknown model specified.")
    return model.to(device)

net = initialize_model(args)

# Optimizer and Loss Setup
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Data Preparation
trainloader, testloader = get_DataLoader(args)

# Training and Testing Functions
def train(epoch, model):
    global global_step
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step += 1

        if (batch_idx + 1) % args.print_freq == 0:
            print(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {running_loss / (batch_idx + 1):.5f}")

def test(epoch, model):
    global best_loss
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    avg_loss = test_loss / len(testloader)
    if avg_loss < best_loss:
        print("Saving new best model")
        torch.save(model.state_dict(), 'best_model.pth')
        best_loss = avg_loss

    print(f"Test Epoch {epoch}, Average Loss: {avg_loss:.5f}")

# Main Training Loop
for epoch in range(300):
    train(epoch, net)
    test(epoch, net)
    scheduler.step()