import sys
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../source/')
from BasicBlock import BasicBlock # type: ignore

block = BasicBlock(64, 128, stride=2)
x = torch.randn(1, 64, 100, 100)
y = block(x)
assert y.shape == (1, 128, 50, 50)
