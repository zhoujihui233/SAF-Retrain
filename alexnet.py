# coding = utf-8

import torch
import torch.nn as nn 

class AlexNet(nn.Module):
    """docstring for AlexNet"""
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 11, 1, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 96, 5, 1, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 120, 3, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.dense = nn.Sequential(
            nn.Linear(4*4*120, 512),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
            # nn.BatchNorm2d(10)
            )
        
    def forward(self, x) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out
