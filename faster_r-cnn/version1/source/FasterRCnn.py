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
        self.roi_align = RoiAlign(output_size=7, spatial_scale=1/32)
        self.roi_head = RoiHead(in_channels=512, num_classes=num_classes)
    def forward(self, images, targets=None):
        # 1. Backbone
        device = images.device
        feat = self.backbone(images)
        # 2. RPN
        cls_logits_rpn, bbox_deltas_rpn = self.rpn(feat)
        H_img, W_img = images.shape[2], images.shape[3]
        anchors = self.rpn.generate_anchors(images, feat)
        proposals, rpn_scores = self.rpn.get_proposals(
            cls_logits_rpn, bbox_deltas_rpn, anchors, (H_img, W_img)
        )
        # Tryb INFERENCJI
        if not self.training or targets is None:
            roi_features = self.roi_align(feat, proposals)
            cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)
            return cls_logits_roi, bbox_deltas_roi, proposals
        # Tryb TRENINGU
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]
        # 3. Dodanie GT do propozycji (gwarantuje positive samples)
        proposals = torch.cat([proposals, gt_boxes], dim=0)
        # 4. Dopasowanie etykiet i targetów regresji do wszystkich propozycji
        roi_labels, roi_reg_targets = self.assign_roi_targets(
            proposals, gt_boxes, gt_labels
        )
        # 5. Subsampling (wybieramy zbalansowane 128 próbek)
        keep_idx = self.subsample_labels(roi_labels, num_samples=128)
        proposals = proposals[keep_idx]
        roi_labels = roi_labels[keep_idx]
        roi_reg_targets = roi_reg_targets[keep_idx]
        # 6. ROI Align i ROI Head (tylko na wybranych 128 próbkach)
        roi_features = self.roi_align(feat, proposals)
        cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)
        # 7. Strata klasyfikacji ROI
        loss_roi_cls = F.cross_entropy(cls_logits_roi, roi_labels)
        # 8. Strata regresji ROI
        pos_mask = roi_labels > 0
        num_classes = cls_logits_roi.shape[1]
        if pos_mask.sum() > 0:
            bbox_deltas_roi = bbox_deltas_roi.view(-1, num_classes, 4)
            bbox_deltas_pos = bbox_deltas_roi[pos_mask, roi_labels[pos_mask]]

            loss_roi_reg = F.smooth_l1_loss(
                bbox_deltas_pos, 
                roi_reg_targets[pos_mask], 
                reduction='mean'
            )
        else:
            loss_roi_reg = torch.tensor(0.0, device=device, requires_grad=True)
        # 9. Straty RPN
        iou_mat = self.rpn.box_iou(anchors, gt_boxes)
        rpn_labels = self.rpn.assign_labels(anchors, gt_boxes, iou_mat)
        _, argmax_iou_per_anchor = iou_mat.max(dim=1)
        matched_gt_for_rpn = gt_boxes[argmax_iou_per_anchor]
        rpn_target_deltas = self.rpn.encode_boxes(anchors, matched_gt_for_rpn)
        _, loss_rpn_cls, loss_rpn_reg = self.rpn.rpn_loss(
            cls_logits_rpn, bbox_deltas_rpn, rpn_labels, rpn_target_deltas
        )
        return {
            "loss_rpn_cls": loss_rpn_cls,
            "loss_rpn_reg": loss_rpn_reg,
            "loss_roi_cls": loss_roi_cls,
            "loss_roi_reg": loss_roi_reg
        }
    #def forward(self, images, targets=None):
        #    # 1. Backbone - Ekstrakcja cech
    #    device = images.device
    #    feat = self.backbone(images)
#
    #    # 2. RPN - Generowanie propozycji
    #    cls_logits_rpn, bbox_deltas_rpn = self.rpn(feat)
    #    H_img, W_img = images.shape[2], images.shape[3]
    #    anchors = self.rpn.generate_anchors(images, feat)
#
    #    proposals, rpn_scores = self.rpn.get_proposals(
    #        cls_logits_rpn, bbox_deltas_rpn, anchors, (H_img, W_img)
    #    )
#
    #    # Tryb INFERENCJI (Testowania)
    #    if not self.training or targets is None:
    #        roi_features = self.roi_align(feat, proposals)
    #        cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)
    #        return cls_logits_roi, bbox_deltas_roi, proposals
#
    #    # Tryb TRENINGU
    #    # 3. Dopasowanie proposals ↔ Ground Truth
    #    #gt_boxes = targets["boxes"]
    #    #gt_labels = targets["labels"]
##
    #    #roi_labels, roi_reg_targets = self.assign_roi_targets(
    #    #    proposals, gt_boxes, gt_labels
    #    #)
##
    #    ## --- NOWOŚĆ: Próbkowanie ROI (Subsampling) ---
    #    ## Wybieramy np. 128 propozycji, żeby zbalansować tło i obiekty
    #    #keep_idx = self.subsample_labels(roi_labels, num_samples=128, positive_fraction=0.25)
    #    #
    #    #sampled_proposals = proposals[keep_idx]
    #    #sampled_roi_labels = roi_labels[keep_idx]
    #    #sampled_roi_reg_targets = roi_reg_targets[keep_idx]
# Fa#sterRCnn.py -> metoda forward
#
    #    # ... (kod po wygenerowaniu proposals przez RPN) ...
#
    #    # 3. Trening: dopasowanie proposals ↔ GT
    #    gt_boxes = targets["boxes"]
    #    gt_labels = targets["labels"]
#
    #    # --- KLUCZOWY DODATEK: Dopisujemy GT do listy propozycji ---
    #    if self.training:
    #        proposals = torch.cat([proposals, gt_boxes], dim=0)
    #    # -----------------------------------------------------------
#
    #    roi_labels, roi_reg_targets = self.assign_roi_targets(
    #        proposals, gt_boxes, gt_labels
    #    )
#
    #    # 4. ROI Subsampling (wybiera 128 z połączonej puli RPN + GT)
    #    if self.training:
    #        keep_idx = self.subsample_labels(roi_labels, num_samples=128)
    #        proposals = proposals[keep_idx]
    #        roi_labels = roi_labels[keep_idx]
    #        roi_reg_targets = roi_reg_targets[keep_idx]
#
    #    # 5. ROI Align (teraz na 100% będzie tu przynajmniej jeden pozytywny obiekt)
    #    roi_features = self.roi_align(feat, proposals)
    #    
    #    # ... (reszta kodu bez zmian) ...
    #    # 4. ROI Align - wycinamy cechy tylko dla wybranych 128 propozycji
    #    roi_features = self.roi_align(feat, sampled_proposals)
#
    #    # 5. ROI Head - klasyfikacja i regresja końcowa
    #    cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)
#
    #    # 6. Strata klasyfikacji ROI
    #    loss_roi_cls = F.cross_entropy(cls_logits_roi, sampled_roi_labels)
#
    #    # 7. Strata regresji ROI
    #    pos_mask = sampled_roi_labels > 0
    #    num_classes = cls_logits_roi.shape[1]
#
    #    if pos_mask.sum() > 0:
    #        # Formatujemy delty na [N, Num_Classes, 4] i wybieramy te dla właściwej klasy
    #        bbox_deltas_roi = bbox_deltas_roi.view(-1, num_classes, 4)
    #        bbox_deltas_pos = bbox_deltas_roi[pos_mask, sampled_roi_labels[pos_mask]]
    #        
    #        loss_roi_reg = F.smooth_l1_loss(
    #            bbox_deltas_pos, 
    #            sampled_roi_reg_targets[pos_mask], 
    #            reduction='mean'
    #        )
    #    else:
    #        loss_roi_reg = torch.tensor(0.0, device=device)
#
    #    # 8. Straty RPN
    #    # Przygotowanie danych do straty RPN (na wszystkich kotwicach)
    #    iou_mat = self.rpn.box_iou(anchors, gt_boxes)
    #    rpn_labels = self.rpn.assign_labels(anchors, gt_boxes, iou_mat)
    #    
    #    _, argmax_iou_per_anchor = iou_mat.max(dim=1)
    #    matched_gt_for_rpn = gt_boxes[argmax_iou_per_anchor]
    #    rpn_target_deltas = self.rpn.encode_boxes(anchors, matched_gt_for_rpn)
#
    #    _, loss_rpn_cls, loss_rpn_reg = self.rpn.rpn_loss(
    #        cls_logits_rpn,
    #        bbox_deltas_rpn,
    #        rpn_labels,
    #        rpn_target_deltas
    #    )
#
    #    # 9. Wynik końcowy - słownik strat
    #    losses = {
    #        "loss_rpn_cls": loss_rpn_cls,
    #        "loss_rpn_reg": loss_rpn_reg,
    #        "loss_roi_cls": loss_roi_cls,
    #        "loss_roi_reg": loss_roi_reg
    #    }
#
    #    return losses
        #def forward(self, images, targets=None):
        #    # 1. Backbone
    #    device = images.device
#
    #    feat = self.backbone(images)
#
    #    # 2. RPN
    #    cls_logits_rpn, bbox_deltas_rpn = self.rpn(feat)
#
    #    H, W = images.shape[2], images.shape[3]
    #    anchors = self.rpn.generate_anchors(images, feat)
#
    #    H, W = images.shape[2], images.shape[3]
    #    proposals, rpn_scores = self.rpn.get_proposals(
    #        cls_logits_rpn, bbox_deltas_rpn, anchors, (H, W)
    #    )
#
    #    # Jeśli inference → od razu ROI Head
    #    if targets is None:
    #        roi_features = self.roi_align(feat, proposals)
    #        cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)
    #        return cls_logits_roi, bbox_deltas_roi, proposals
#
    #    # 3. Trening: dopasowanie proposals ↔ GT
    #    gt_boxes = targets["boxes"]
    #    gt_labels = targets["labels"]
#
    #    roi_labels, roi_reg_targets = self.assign_roi_targets(
    #        proposals, gt_boxes, gt_labels
    #    )
#
    #    # 4. ROI Align
    #    roi_features = self.roi_align(feat, proposals)
#
    #    # 5. ROI Head
    #    cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)
#
    #    # 6. Strata klasyfikacji ROI
    #    loss_roi_cls = F.cross_entropy(cls_logits_roi, roi_labels)
#
    #    # 7. Strata regresji ROI
    #    pos_mask = roi_labels > 0
    #    num_classes = cls_logits_roi.shape[1]
#
    #    bbox_deltas_roi = bbox_deltas_roi.view(-1, num_classes, 4)
    #    bbox_deltas_pos = bbox_deltas_roi[pos_mask, roi_labels[pos_mask]]
#
    #    #loss_roi_reg = F.smooth_l1_loss(
    #    #    bbox_deltas_pos,
    #    #    roi_reg_targets[pos_mask]
    #    #)  
    #    if pos_mask.sum() == 0:
    #        loss_roi_reg = torch.tensor(0.0, device=feat.device)
    #    else:
    #        loss_roi_reg = F.smooth_l1_loss(
    #        bbox_deltas_pos,
    #        roi_reg_targets[pos_mask])
#
    #    # 8. Straty RPN (już masz)
    #    #loss_rpn_cls, loss_rpn_reg = self.rpn.rpn_loss(
    #    #    cls_logits_rpn, bbox_deltas_rpn, anchors, gt_boxes
    #    #)
    #    # 8. Straty RPN
#
    #    # IoU anchors ↔ GT
    #    iou_mat = self.rpn.box_iou(anchors, gt_boxes)
#
    #    # labels: [N]  (0, 1, -1)
    #    labels = self.rpn.assign_labels(anchors, gt_boxes, iou_mat)
#
    #    # matched GT box for each anchor
    #    _, argmax_iou_per_anchor = iou_mat.max(dim=1)
    #    matched_gt = gt_boxes[argmax_iou_per_anchor]   # [N, 4]
#
    #    # target deltas: [N, 4]
    #    target_deltas = self.rpn.encode_boxes(anchors, matched_gt)
#
    #    # teraz dopiero liczymy stratę
    #    loss_rpn_total, loss_rpn_cls, loss_rpn_reg = self.rpn.rpn_loss(
    #        cls_logits_rpn,
    #        bbox_deltas_rpn,
    #        labels,
    #        target_deltas
    #    )
    #    #print("num proposals:", len(proposals))
    #    #print("num positive roi:", pos_mask.sum().item())
    #    #print("roi_labels unique:", roi_labels.unique())
#
    #    # 9. Zwracamy wszystkie straty
    #    losses = {
    #        "loss_rpn_cls": loss_rpn_cls,
    #        "loss_rpn_reg": loss_rpn_reg,
    #        "loss_roi_cls": loss_roi_cls,
    #        "loss_roi_reg": loss_roi_reg
    #    }
#
    #    return losses

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

    #def predict(self, images, score_thresh=0.5, nms_thresh=0.5):
    #    self.eval()
    #    with torch.no_grad():
    #        # 1. Backbone
    #        feat = self.backbone(images)
#
    #        # 2. RPN
    #        cls_logits_rpn, bbox_deltas_rpn = self.rpn(feat)
    #        H, W = images.shape[2], images.shape[3]
    #        anchors = self.rpn.generate_anchors(images, feat)
    #        proposals, rpn_scores = self.rpn.get_proposals(
    #            cls_logits_rpn, bbox_deltas_rpn, anchors, (H, W)
    #        )
#
    #        # 3. ROI Align
    #        roi_features = self.roi_align(feat, proposals)
#
    #        # 4. ROI Head
    #        cls_logits_roi, bbox_deltas_roi = self.roi_head(roi_features)
#
    #        # 5. Softmax → scores
    #        scores = cls_logits_roi.softmax(dim=1)  # [N, num_classes]
    #        max_scores, labels = scores.max(dim=1)  # [N]
#
    #        # 6. Decode bbox deltas
    #        num_classes = scores.shape[1]
    #        bbox_deltas_roi = bbox_deltas_roi.view(-1, num_classes, 4)
#
    #        # wybieramy delty dla najlepszej klasy
    #        chosen_deltas = bbox_deltas_roi[torch.arange(len(labels)), labels]
#
    #        # dekodujemy
    #        boxes = self.rpn.decode_boxes(proposals, chosen_deltas)
#
    #        # 7. Filtrowanie po score_thresh
    #        keep = max_scores >= score_thresh
    #        boxes = boxes[keep]
    #        labels = labels[keep]
    #        max_scores = max_scores[keep]
#
    #        # 8. Per-class NMS
    #        final_boxes = []
    #        final_labels = []
    #        final_scores = []
#
    #        for cls in labels.unique():
    #            cls_mask = labels == cls
#
    #            cls_boxes = boxes[cls_mask]
    #            cls_scores = max_scores[cls_mask]
#
    #            keep_idx = nms(cls_boxes, cls_scores, nms_thresh)
#
    #            final_boxes.append(cls_boxes[keep_idx])
    #            final_labels.append(labels[cls_mask][keep_idx])
    #            final_scores.append(cls_scores[keep_idx])
#
    #        # 9. Łączenie wyników
    #        if len(final_boxes) == 0:
    #            return torch.empty((0,4)), torch.empty((0,),dtype=torch.long), torch.empty((0,))
#
    #        final_boxes = torch.cat(final_boxes)
    #        final_labels = torch.cat(final_labels)
    #        final_scores = torch.cat(final_scores)
#
    #    return final_boxes, final_labels, final_scores
    @torch.no_grad()
    def predict(self, images, score_thresh=0.7):
        self.eval()
        device = images.device
        feat = self.backbone(images)
        
        # 1. Pobierz propozycje z RPN
        cls_logits_rpn, bbox_deltas_rpn = self.rpn(feat)
        anchors = self.rpn.generate_anchors(images, feat)
        proposals, _ = self.rpn.get_proposals(cls_logits_rpn, bbox_deltas_rpn, anchors, images.shape[2:])
        
        # 2. ROI Head
        roi_features = self.roi_align(feat, proposals)
        cls_logits, bbox_deltas = self.roi_head(roi_features)
        
        # 3. Softmax dla klasyfikacji
        probs = torch.softmax(cls_logits, dim=1)
        
        # 4. Dekodowanie ramek dla każdej klasy (oprócz tła)
        num_classes = cls_logits.shape[1]
        bbox_deltas = bbox_deltas.view(-1, num_classes, 4)
        
        final_boxes = []
        final_labels = []
        final_scores = []
    
        # Iterujemy po klasach (pomijamy 0 - background)
        for cls_idx in range(1, num_classes):
            cls_probs = probs[:, cls_idx]
            
            # Filtrujemy po progu pewności
            keep = cls_probs > score_thresh
            if not keep.any():
                continue
                
            cls_probs = cls_probs[keep]
            cls_deltas = bbox_deltas[keep, cls_idx]
            cls_proposals = proposals[keep]
            
            # Dekodujemy ramki (używamy Twojej funkcji z RPN)
            decoded_boxes = self.rpn.decode_boxes(cls_proposals, cls_deltas)
            
            # Przycinamy do rozmiaru obrazu
            h, w = images.shape[2:]
            decoded_boxes[:, [0, 2]] = decoded_boxes[:, [0, 2]].clamp(0, w)
            decoded_boxes[:, [1, 3]] = decoded_boxes[:, [1, 3]].clamp(0, h)
            
            # --- KLUCZOWY KROK: NMS ---
            # Musimy usunąć nakładające się ramki dla tej samej klasy
            from torchvision.ops import nms
            keep_nms = nms(decoded_boxes, cls_probs, iou_threshold=0.3)
            
            final_boxes.append(decoded_boxes[keep_nms])
            final_labels.append(torch.full((len(keep_nms),), cls_idx, device=device))
            final_scores.append(cls_probs[keep_nms])
    
        if not final_boxes:
            return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))
    
        return torch.cat(final_boxes), torch.cat(final_labels), torch.cat(final_scores)
    def subsample_labels(self, labels, num_samples=128, positive_fraction=0.25):
        # Znajdź indeksy pozytywne (klasy 1-20) i negatywne (0)
        pos_idx = torch.where(labels > 0)[0]
        neg_idx = torch.where(labels == 0)[0]

        # Ile chcemy pozytywnych? (np. 128 * 0.25 = 32)
        num_pos = int(num_samples * positive_fraction)
        num_pos = min(pos_idx.numel(), num_pos)

        # Ile chcemy negatywnych?
        num_neg = num_samples - num_pos
        num_neg = min(neg_idx.numel(), num_neg)

        # Losujemy indeksy
        perm_pos = torch.randperm(pos_idx.numel(), device=labels.device)[:num_pos]
        perm_neg = torch.randperm(neg_idx.numel(), device=labels.device)[:num_neg]

        keep_idx = torch.cat([pos_idx[perm_pos], neg_idx[perm_neg]])
        return keep_idx