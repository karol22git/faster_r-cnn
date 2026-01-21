import sys
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../source/')
from ResNet import ResNet # type: ignore

model = ResNet()
x = torch.randn(1, 3, 800, 800)
y = model(x)
assert y.shape[1] == 512
assert y.shape[2] < 60 and y.shape[3] < 60
