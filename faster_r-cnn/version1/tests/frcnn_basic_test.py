import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../source/')
sys.path.append("../../../resnet18/version1/source")
sys.path.append("../../../RPN/version1/source")
from FasterRCnn import FasterRCnn  #type: ignore
from ResNet import ResNet #type: ignore
from RPN import RPN #type: ignore

model = FasterRCnn()
x = torch.randn(1, 3, 800, 800)
cls_logits, bbox_deltas, anchors = model(x)
assert isinstance(model.backbone, ResNet)
assert isinstance(model.rpn, RPN)
assert cls_logits.shape[0] == bbox_deltas.shape[0]
assert anchors.shape[0] == cls_logits.shape[0]
x = torch.randn(1, 3, 64, 64)
model(x)
x = torch.randn(1, 3, 1200, 1200)
model(x)
x = torch.randn(2, 3, 800, 800)
model(x)
model.cuda()
x = torch.randn(1, 3, 800, 800).cuda()
model(x)
torch.cuda.memory_allocated()
