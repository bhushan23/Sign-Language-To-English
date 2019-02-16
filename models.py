import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class M1(nn.Module):
    def __init__(self):
        super(M1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4)
        self.maxpool1 = nn.MaxPool2d(64)
        self.fc1 = nn.Linear(238144, 1024)
        self.fc2 = nn.Linear(1024, 24)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, padding = 2)
        self.maxpool1 = nn.MaxPool2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.maxpool2 = nn.MaxPool2d(128)
        self.fc1 = nn.Linear(254016, 1024)
        self.fc2 = nn.Linear(1024, 24)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x
