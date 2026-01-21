import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../source/')
#sys.path.append("../../../resnet18/version1/source")
#sys.path.append("../../../RPN/version1/source")
from FasterRCnn import FasterRCnn  #type: ignore
#from ResNet import ResNet #type: ignore
#from RPN import RPN #type: ignore

def test_faster_rcnn_end_to_end_inference():
    model = FasterRCnn(num_classes=5)
    model.eval()

    # sztuczny obraz
    images = torch.randn(1, 3, 800, 800)

    with torch.no_grad():
        cls_logits_roi, bbox_deltas_roi, proposals = model(images)

    # 1. proposals muszą istnieć
    assert proposals.ndim == 2
    assert proposals.shape[1] == 4
    assert len(proposals) > 0

    # 2. klasy ROI
    assert cls_logits_roi.ndim == 2
    assert cls_logits_roi.shape[1] == 5

    # 3. bbox deltas ROI
    assert bbox_deltas_roi.ndim == 2
    assert bbox_deltas_roi.shape[1] == 5 * 4

    # 4. brak NaN
    assert torch.isfinite(cls_logits_roi).all()
    assert torch.isfinite(bbox_deltas_roi).all()
    assert torch.isfinite(proposals).all()

    print("Inference test passed.")
    print("num proposals:", len(proposals))
    print("cls_logits_roi shape:", cls_logits_roi.shape)
    print("bbox_deltas_roi shape:", bbox_deltas_roi.shape)
test_faster_rcnn_end_to_end_inference()