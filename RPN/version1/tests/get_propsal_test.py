import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../source/')
from RPN import RPN
def test_get_proposals_runs():
    rpn = RPN()
    image = torch.randn(1, 3, 800, 800)
    feat = torch.randn(1, 512, 50, 50)

    cls_logits = torch.randn(50*50*9, 2)
    bbox_deltas = torch.randn(50*50*9, 4)
    anchors = rpn.generate_anchors(image, feat)

    proposals, scores = rpn.get_proposals(cls_logits, bbox_deltas, anchors, (800, 800))

    assert proposals.ndim == 2
    assert scores.ndim == 1
def test_proposals_not_more_than_anchors():
    rpn = RPN()
    image = torch.randn(1, 3, 800, 800)
    feat = torch.randn(1, 512, 50, 50)

    cls_logits = torch.randn(50*50*9, 2)
    bbox_deltas = torch.randn(50*50*9, 4)
    anchors = rpn.generate_anchors(image, feat)

    proposals, scores = rpn.get_proposals(cls_logits, bbox_deltas, anchors, (800, 800))

    assert len(proposals) <= len(anchors)
def test_proposals_within_image_bounds():
    rpn = RPN()
    image = torch.randn(1, 3, 800, 800)
    feat = torch.randn(1, 512, 50, 50)

    cls_logits = torch.randn(50*50*9, 2)
    bbox_deltas = torch.randn(50*50*9, 4)
    anchors = rpn.generate_anchors(image, feat)

    proposals, scores = rpn.get_proposals(cls_logits, bbox_deltas, anchors, (800, 800))

    assert (proposals[:, 0] >= 0).all()
    assert (proposals[:, 1] >= 0).all()
    assert (proposals[:, 2] <= 800).all()
    assert (proposals[:, 3] <= 800).all()
def test_nms_removes_duplicates():
    rpn = RPN()

    proposals = torch.tensor([
        [10., 10., 50., 50.],
        [10., 10., 50., 50.],  # duplikat
    ])

    scores = torch.tensor([0.9, 0.8])

    from torchvision.ops import nms
    keep = nms(proposals, scores, 0.7)

    assert len(keep) == 1
def test_small_boxes_are_removed():
    rpn = RPN()

    proposals = torch.tensor([
        [10., 10., 12., 12.],  # 2x2 → za mały
        [20., 20., 40., 40.],  # OK
    ])

    scores = torch.tensor([0.9, 0.8])

    ws = proposals[:, 2] - proposals[:, 0]
    hs = proposals[:, 3] - proposals[:, 1]
    keep = (ws >= 4) & (hs >= 4)

    filtered = proposals[keep]

    assert len(filtered) == 1
def test_decode_boxes_identity():
    rpn = RPN()

    anchors = torch.tensor([[0., 0., 10., 10.]])
    deltas = torch.zeros((1, 4))

    decoded = rpn.decode_boxes(anchors, deltas)

    assert torch.allclose(decoded, anchors)
def test_scores_sorted_after_nms():
    rpn = RPN()
    image = torch.randn(1, 3, 800, 800)
    feat = torch.randn(1, 512, 50, 50)

    cls_logits = torch.randn(50*50*9, 2)
    bbox_deltas = torch.randn(50*50*9, 4)
    anchors = rpn.generate_anchors(image, feat)

    proposals, scores = rpn.get_proposals(cls_logits, bbox_deltas, anchors, (800, 800))

    assert torch.all(scores[:-1] >= scores[1:])
test_get_proposals_runs()
test_proposals_not_more_than_anchors()
test_proposals_within_image_bounds()
test_nms_removes_duplicates()
test_small_boxes_are_removed()
test_scores_sorted_after_nms()
test_decode_boxes_identity()