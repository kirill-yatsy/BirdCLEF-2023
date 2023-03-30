import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from torch import utils as t_utils
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import sys
import os
from torchaudio.utils import download_asset
from helpers import train_loop, test_loop
from IPython.display import Audio
from os import walk
import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
import torch.autograd.profiler as profiler
sys.dont_write_bytecode = True
torch.manual_seed(17)
from datetime import datetime



class NeuralNetwork(nn.Module):
    def __init__(self, numberOfClasses: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential( 
            nn.BatchNorm2d(3),
            nn.Conv2d(kernel_size=6, stride=3, padding=0, in_channels=3, out_channels=3),
            # nn.Dropout(0.25), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3), 
            nn.Flatten(),
            # nn.Linear(5544, 5544),
            # nn.ReLU(),
            nn.Linear(5544, numberOfClasses)
        )

    def forward(self, x: torch.Tensor): 
        logits = self.linear_relu_stack(x) 
        return logits
    


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    device = torch.device("cuda")