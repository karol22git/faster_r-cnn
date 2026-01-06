
import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride = 1, padding = 1 , bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out  = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity

        return F.relu(out)