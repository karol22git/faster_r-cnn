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


def test_faster_rcnn_end_to_end_training():
    # 1. Model
    model = FasterRCnn(num_classes=5)
    model.train()

    # 2. Sztuczny obraz
    images = torch.randn(1, 3, 800, 800)

    # 3. Sztuczne ground truth
    gt_boxes = torch.tensor([[100., 100., 300., 300.]])   # jeden box
    gt_labels = torch.tensor([1])                         # klasa 1

    targets = {
        "boxes": gt_boxes,
        "labels": gt_labels
    }

    # 4. Forward (trening)
    losses = model(images, targets)

    # 5. Sprawdzenie, czy wszystkie straty istnieją
    assert "loss_rpn_cls" in losses
    assert "loss_rpn_reg" in losses
    assert "loss_roi_cls" in losses
    assert "loss_roi_reg" in losses
    print("loss_rpn_cls:", losses["loss_rpn_cls"].item())
    print("loss_rpn_reg:", losses["loss_rpn_reg"].item())
    print("loss_roi_cls:", losses["loss_roi_cls"].item())
    print("loss_roi_reg:", losses["loss_roi_reg"].item())

    # 6. Sprawdzenie, czy straty są liczbami > 0
    for k, v in losses.items():
        assert torch.is_tensor(v)
        assert v.item() >= 0

    # 7. Backprop
    total_loss = sum(losses.values())
    total_loss.backward()

    # 8. Optymalizator
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()
test_faster_rcnn_end_to_end_training()