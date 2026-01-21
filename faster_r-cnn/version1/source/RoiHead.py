import torch.nn as nn
import torch.nn.functional as F
import torch
class RoiHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=21, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, num_classes * 4)

    def forward(self, roi_features):
        # roi_features: [N, C, 7, 7]
        x = roi_features.flatten(start_dim=1)  # [N, C*7*7]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        cls_logits = self.cls_head(x)      # [N, num_classes]
        bbox_deltas = self.reg_head(x)     # [N, num_classes*4]

        return cls_logits, bbox_deltas
    