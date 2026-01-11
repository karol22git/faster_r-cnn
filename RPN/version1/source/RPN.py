import torch
import torch.nn as nn
import torch.nn.functional as F
class RPN(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, n_anchors=9, scales = [8,16,32], aspect_ratio = [0.5, 1, 2]):
        super().__init__()
        self.scales = scales
        self.aspect_ratio = aspect_ratio
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size= 3,stride= 1, padding =1)
        
        self.cls_score = nn.Conv2d(out_channels, n_anchors * 2,kernel_size= 1,stride= 1, padding=0)
        
        self.bbox_pred = nn.Conv2d(out_channels, n_anchors * 4, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        rpn_cls_score = self.cls_score(x)
        rpn_bbox_pred = self.bbox_pred(x)
        return rpn_cls_score, rpn_bbox_pred
    
    def generate_anchors(self, image, feat):
            r"""
            Method to generate anchors. First we generate one set of zero-centred anchors
            using the scales and aspect ratios provided.
            We then generate shift values in x,y axis for all featuremap locations.
            The single zero centred anchors generated are replicated and shifted accordingly
            to generate anchors for all feature map locations.
            Note that these anchors are generated such that their centre is top left corner of the
            feature map cell rather than the centre of the feature map cell.
            :param image: (N, C, H, W) tensor
            :param feat: (N, C_feat, H_feat, W_feat) tensor
            :return: anchor boxes of shape (H_feat * W_feat * num_anchors_per_location, 4)
            """
            grid_h, grid_w = feat.shape[-2:]
            image_h, image_w = image.shape[-2:]

            # For the vgg16 case stride would be 16 for both h and w
            stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
            stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)

            scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
            aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)

            # Assuming anchors of scale 128 sq pixels
            # For 1:1 it would be (128, 128) -> area=16384
            # For 2:1 it would be (181.02, 90.51) -> area=16384
            # For 1:2 it would be (90.51, 181.02) -> area=16384

            # The below code ensures h/w = aspect_ratios and h*w=1
            h_ratios = torch.sqrt(aspect_ratios)
            w_ratios = 1 / h_ratios

            # Now we will just multiply h and w with scale(example 128)
            # to make h*w = 128 sq pixels and h/w = aspect_ratios
            # This gives us the widths and heights of all anchors
            # which we need to replicate at all locations
            ws = (w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h_ratios[:, None] * scales[None, :]).view(-1)

            # Now we make all anchors zero centred
            # So x1, y1, x2, y2 = -w/2, -h/2, w/2, h/2
            base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
            base_anchors = base_anchors.round()

            # Get the shifts in x axis (0, 1,..., W_feat-1) * stride_w
            shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w

            # Get the shifts in x axis (0, 1,..., H_feat-1) * stride_h
            shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h

            # Create a grid using these shifts
            shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            # shifts_x -> (H_feat, W_feat)
            # shifts_y -> (H_feat, W_feat)

            shifts_x = shifts_x.reshape(-1)
            shifts_y = shifts_y.reshape(-1)
            # Setting shifts for x1 and x2(same as shifts_x) and y1 and y2(same as shifts_y)
            shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)
            # shifts -> (H_feat * W_feat, 4)

            # base_anchors -> (num_anchors_per_location, 4)
            # shifts -> (H_feat * W_feat, 4)
            # Add these shifts to each of the base anchors
            anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
            # anchors -> (H_feat * W_feat, num_anchors_per_location, 4)
            anchors = anchors.reshape(-1, 4)
            # anchors -> (H_feat * W_feat * num_anchors_per_location, 4)
            return anchors
    def box_iou(anchors, boxes):
        """
        anchors: (N, 4) - wygenerowane kotwice
        boxes: (M, 4) - prawdziwe ramki (Ground Truth)
        Zwraca: (N, M) macierz IoU
        """

        # 1. Obliczamy powierzchnię obu zestawów ramek
        # Area = (x2 - x1) * (y2 - y1)
        area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # 2. Obliczamy współrzędne części wspólnej (Intersection)
        # Wykorzystujemy broadcasting: (N, 1, 2) vs (1, M, 2)
        # Lewy górny róg części wspólnej to MAX z lewych górnych rogów
        lt = torch.max(anchors[:, None, :2], boxes[:, :2])  # [N, M, 2]
        # Prawy dolny róg części wspólnej to MIN z prawych dolnych rogów
        rb = torch.min(anchors[:, None, 2:], boxes[:, 2:])  # [N, M, 2]

        # 3. Obliczamy szerokość i wysokość części wspólnej
        wh = (rb - lt).clamp(min=0)  # Jeśli się nie nakładają, clamp da 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # 4. IoU = Area of Intersection / Area of Union
        # Union = Area1 + Area2 - Intersection
        union = area_anchors[:, None] + area_boxes - inter

        return inter / union
    def assign_labels(anchors, gt_boxes, iou_matrix):
        # Domyślnie wszystkie kotwice to -1 (ignoruj)
        labels = torch.full((anchors.size(0),), -1, dtype=torch.int64)

        # Dla każdej kotwicy znajdź obiekt, z którym ma najwyższe IoU
        max_iou_per_anchor, argmax_iou_per_anchor = iou_matrix.max(dim=1)

        # 1. Kotwice z niskim IoU to tło (0)
        labels[max_iou_per_anchor < 0.3] = 0

        # 2. Kotwice z wysokim IoU to obiekt (1)
        labels[max_iou_per_anchor >= 0.7] = 1

        # 3. Specjalny przypadek: Zawsze daj 1 kotwicy, która ma 
        # absolutnie najwyższe IoU dla danego obiektu (nawet jeśli < 0.7)
        max_iou_per_gt, _ = iou_matrix.max(dim=0)
        for i in range(len(max_iou_per_gt)):
            labels[iou_matrix[:, i] == max_iou_per_gt[i]] = 1

        return labels
    def encode_boxes(anchors, gt_boxes):
        """
        anchors: [N, 4] (x1, y1, x2, y2)
        gt_boxes: [N, 4] (x1, y1, x2, y2) - przypisane obiekty dla każdej kotwicy
        """
        # Przeliczamy na format (środek_x, środek_y, szerokość, wysokość)
        w_a = anchors[:, 2] - anchors[:, 0]
        h_a = anchors[:, 3] - anchors[:, 1]
        ctr_x_a = anchors[:, 0] + 0.5 * w_a
        ctr_y_a = anchors[:, 1] + 0.5 * h_a

        w_g = gt_boxes[:, 2] - gt_boxes[:, 0]
        h_g = gt_boxes[:, 3] - gt_boxes[:, 1]
        ctr_x_g = gt_boxes[:, 0] + 0.5 * w_g
        ctr_y_g = gt_boxes[:, 1] + 0.5 * h_g

        # Obliczamy delty (zgodnie z publikacją Faster R-CNN)
        dx = (ctr_x_g - ctr_x_a) / w_a
        dy = (ctr_y_g - ctr_y_a) / h_a
        dw = torch.log(w_g / w_a)
        dh = torch.log(h_g / h_a)

        deltas = torch.stack([dx, dy, dw, dh], dim=1)
        return deltas
    def decode_boxes(anchors, deltas):
        """
        anchors: [N, 4]
        deltas: [N, 4] (wynik z sieci RPN)
        """
        w_a = anchors[:, 2] - anchors[:, 0]
        h_a = anchors[:, 3] - anchors[:, 1]
        ctr_x_a = anchors[:, 0] + 0.5 * w_a
        ctr_y_a = anchors[:, 1] + 0.5 * h_a

        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

        # Odwracamy wzory z kodowania
        pred_ctr_x = dx * w_a + ctr_x_a
        pred_ctr_y = dy * h_a + ctr_y_a
        pred_w = torch.exp(dw) * w_a
        pred_h = torch.exp(dh) * h_a

        # Powrót do formatu (x1, y1, x2, y2)
        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        return torch.stack([x1, y1, x2, y2], dim=1)
    import torch.nn.functional as F

    def rpn_loss(pred_cls_scores, pred_deltas, labels, target_deltas):
        """
        pred_cls_scores: [N, 2] - wyniki klasyfikacji z RPN (obiekt vs tło)
        pred_deltas: [N, 4] - przewidziane przez sieć poprawki
        labels: [N] - prawdziwe etykiety z funkcji assign_labels (1, 0, -1)
        target_deltas: [N, 4] - prawdziwe poprawki z funkcji encode_boxes
        """
        
        # 1. Filtrujemy tylko te kotwice, które nie są ignorowane (labels != -1)
        # Ignorujemy kotwice z etykietą -1, żeby nie psuły gradientu
        keep_idx = torch.where(labels != -1)[0]
        
        cls_loss = F.cross_entropy(pred_cls_scores[keep_idx], labels[keep_idx])
    
        # 2. Strata regresji (tylko dla kotwic, które są obiektami: labels == 1)
        pos_idx = torch.where(labels == 1)[0]
        
        if len(pos_idx) > 0:
            # Smooth L1 liczymy tylko tam, gdzie faktycznie jest obiekt
            reg_loss = F.smooth_l1_loss(
                pred_deltas[pos_idx], 
                target_deltas[pos_idx], 
                beta=1.0 / 9.0, 
                reduction='sum'
            )
            # Normalizujemy przez liczbę wszystkich pozytywnych kotwic
            reg_loss = reg_loss / (labels == 1).sum()
        else:
            reg_loss = torch.tensor(0.0).to(pred_deltas.device)
    
        # Całkowita strata
        total_loss = cls_loss + reg_loss
        return total_loss, cls_loss, reg_loss