import sys
import torch
sys.path.append('../source/')
from RoiAlign import RoiAlign #type: ignore

def test_roi_align_runs():
    roi = RoiAlign()
    feat = torch.randn(1, 512, 50, 50)
    proposals = torch.tensor([[10., 10., 100., 100.]])
    out = roi(feat, proposals)
    assert out.shape == (1, 512, 7, 7)
def test_roi_align_multiple_boxes():
    roi = RoiAlign()
    feat = torch.randn(1, 512, 50, 50)
    proposals = torch.tensor([
        [10., 10., 100., 100.],
        [200., 200., 300., 300.]
    ])
    out = roi(feat, proposals)
    assert out.shape == (2, 512, 7, 7)

test_roi_align_runs()
test_roi_align_multiple_boxes()