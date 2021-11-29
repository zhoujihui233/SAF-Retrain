import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input / batch_size
                out_channels=20,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=0,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 50, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(800, 300),  # 只有800是需要计算的，其他数值都是指定的
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(300, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # x.size(0)：(batch_size, 32 * 4 * 4)   # view是Torch中的reshape
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output
