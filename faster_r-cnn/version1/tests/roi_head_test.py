
import sys
import torch
sys.path.append('../source/')
from RoiHead import RoiHead #type: ignore
def test_roi_head_shapes():
    head = RoiHead(in_channels=512, num_classes=5)
    roi_features = torch.randn(10, 512, 7, 7)

    cls_logits, bbox_deltas = head(roi_features)

    assert cls_logits.shape == (10, 5)
    assert bbox_deltas.shape == (10, 5 * 4)

test_roi_head_shapes()