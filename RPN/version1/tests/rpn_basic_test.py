import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../source/')
from RPN import RPN

rpn = RPN()
feat = torch.randn(1, 512, 25, 25)
image = torch.randn(1, 3, 800, 800)
cls_logits, bbox_deltas = rpn(feat)

assert cls_logits.shape[1] == 2
assert bbox_deltas.shape[1] == 4
assert cls_logits.shape[0] == 5625
assert bbox_deltas.shape[0] == 5625
anchors = rpn.generate_anchors(image, feat)
assert anchors.shape[1] == 4
assert anchors.shape[0] == 5625
anchors = torch.tensor([[0,0,10,10]])
gt = torch.tensor([[0,0,10,10]])
iou = rpn.box_iou(anchors, gt)
assert iou.item() == 1.0
anchors = torch.tensor([[0,0,10,10], [100,100,110,110]])
gt = torch.tensor([[0,0,10,10]])
iou = rpn.box_iou(anchors, gt)
labels = rpn.assign_labels(anchors, gt, iou)

assert labels[0] == 1   # perfect match
assert labels[1] == 0   # background
