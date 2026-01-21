import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

sys.path.append("../../../resnet18/version1/source")
sys.path.append("../../../RPN/version1/source")

from ResNet import ResNet # type: ignore
from RPN import RPN
from RoiAlign import RoiAlign
from RoiHead import RoiHead

class FasterRCnn(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = ResNet()
        self.rpn = RPN()
        self.roi_align = RoiAlign(output_size=7, spatial_scale=1/16)
        self.roi_head = RoiHead(in_channels=512, num_classes=num_classes)

    def forward(self, images, targets=None):
        # 1. Backbone
        device = images.device

        feat = self.backbone(images)

        # 2. RPN
        cls_logits_rpn, bbox_deltas_rpn = self.rpn(feat)

        H, W = images.shape[2], images.shape[3]
        anchors = self.rpn.generate_anchors(images, feat)

        H, W = images.shape[2], images.shape[3]
        proposals, rpn_scores = self.rpn.get_proposals(
            cls_logits_rpn, bbox_deltas_rpn, anchors, (H, W)
        )

        # Jeśli inference → od razu ROI Head
        if targets is None:
            roi_features = self.roi_align(feat, proposals)
            cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)
            return cls_logits_roi, bbox_deltas_roi, proposals

        # 3. Trening: dopasowanie proposals ↔ GT
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]

        roi_labels, roi_reg_targets = self.assign_roi_targets(
            proposals, gt_boxes, gt_labels
        )

        # 4. ROI Align
        roi_features = self.roi_align(feat, proposals)

        # 5. ROI Head
        cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)

        # 6. Strata klasyfikacji ROI
        loss_roi_cls = F.cross_entropy(cls_logits_roi, roi_labels)

        # 7. Strata regresji ROI
        pos_mask = roi_labels > 0
        num_classes = cls_logits_roi.shape[1]

        bbox_deltas_roi = bbox_deltas_roi.view(-1, num_classes, 4)
        bbox_deltas_pos = bbox_deltas_roi[pos_mask, roi_labels[pos_mask]]

        #loss_roi_reg = F.smooth_l1_loss(
        #    bbox_deltas_pos,
        #    roi_reg_targets[pos_mask]
        #)  
        if pos_mask.sum() == 0:
            loss_roi_reg = torch.tensor(0.0, device=feat.device)
        else:
            loss_roi_reg = F.smooth_l1_loss(
            bbox_deltas_pos,
            roi_reg_targets[pos_mask])

        # 8. Straty RPN (już masz)
        #loss_rpn_cls, loss_rpn_reg = self.rpn.rpn_loss(
        #    cls_logits_rpn, bbox_deltas_rpn, anchors, gt_boxes
        #)
        # 8. Straty RPN

        # IoU anchors ↔ GT
        iou_mat = self.rpn.box_iou(anchors, gt_boxes)

        # labels: [N]  (0, 1, -1)
        labels = self.rpn.assign_labels(anchors, gt_boxes, iou_mat)

        # matched GT box for each anchor
        _, argmax_iou_per_anchor = iou_mat.max(dim=1)
        matched_gt = gt_boxes[argmax_iou_per_anchor]   # [N, 4]

        # target deltas: [N, 4]
        target_deltas = self.rpn.encode_boxes(anchors, matched_gt)

        # teraz dopiero liczymy stratę
        loss_rpn_total, loss_rpn_cls, loss_rpn_reg = self.rpn.rpn_loss(
            cls_logits_rpn,
            bbox_deltas_rpn,
            labels,
            target_deltas
        )
        #print("num proposals:", len(proposals))
        #print("num positive roi:", pos_mask.sum().item())
        #print("roi_labels unique:", roi_labels.unique())

        # 9. Zwracamy wszystkie straty
        losses = {
            "loss_rpn_cls": loss_rpn_cls,
            "loss_rpn_reg": loss_rpn_reg,
            "loss_roi_cls": loss_roi_cls,
            "loss_roi_reg": loss_roi_reg
        }

        return losses

    def assign_roi_targets(self, proposals, gt_boxes, gt_labels, iou_threshold=0.5):
        iou = self.rpn.box_iou(proposals, gt_boxes)  # [N, M]
        max_iou, max_idx = iou.max(dim=1)

        # label = 0 → background
        roi_labels = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)

        # pozytywne ROI
        pos_mask = max_iou >= iou_threshold
        roi_labels[pos_mask] = gt_labels[max_idx[pos_mask]]

        # regresja bboxów tylko dla pozytywnych
        roi_reg_targets = torch.zeros_like(proposals)
        roi_reg_targets[pos_mask] = self.rpn.encode_boxes(
            proposals[pos_mask],
            gt_boxes[max_idx[pos_mask]]
        )

        return roi_labels, roi_reg_targets

    def predict(self, images, score_thresh=0.5, nms_thresh=0.5):
        self.eval()
        with torch.no_grad():
            # 1. Backbone
            feat = self.backbone(images)

            # 2. RPN
            cls_logits_rpn, bbox_deltas_rpn = self.rpn(feat)
            H, W = images.shape[2], images.shape[3]
            anchors = self.rpn.generate_anchors(images, feat)
            proposals, rpn_scores = self.rpn.get_proposals(
                cls_logits_rpn, bbox_deltas_rpn, anchors, (H, W)
            )

            # 3. ROI Align
            roi_features = self.roi_align(feat, proposals)

            # 4. ROI Head
            cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)

            # 5. Softmax → scores
            scores = cls_logits_roi.softmax(dim=1)  # [N, num_classes]
            max_scores, labels = scores.max(dim=1)  # [N]

            # 6. Decode bbox deltas
            num_classes = scores.shape[1]
            bbox_deltas_roi = bbox_deltas_roi.view(-1, num_classes, 4)

            # wybieramy delty dla najlepszej klasy
            chosen_deltas = bbox_deltas_roi[torch.arange(len(labels)), labels]

            # dekodujemy
            boxes = self.rpn.decode_boxes(proposals, chosen_deltas)

            # 7. Filtrowanie po score_thresh
            keep = max_scores >= score_thresh
            boxes = boxes[keep]
            labels = labels[keep]
            max_scores = max_scores[keep]

            # 8. Per-class NMS
            final_boxes = []
            final_labels = []
            final_scores = []

            for cls in labels.unique():
                cls_mask = labels == cls

                cls_boxes = boxes[cls_mask]
                cls_scores = max_scores[cls_mask]

                keep_idx = nms(cls_boxes, cls_scores, nms_thresh)

                final_boxes.append(cls_boxes[keep_idx])
                final_labels.append(labels[cls_mask][keep_idx])
                final_scores.append(cls_scores[keep_idx])

            # 9. Łączenie wyników
            if len(final_boxes) == 0:
                return torch.empty((0,4)), torch.empty((0,),dtype=torch.long), torch.empty((0,))

            final_boxes = torch.cat(final_boxes)
            final_labels = torch.cat(final_labels)
            final_scores = torch.cat(final_scores)

        return final_boxes, final_labels, final_scores
