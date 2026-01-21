from torchvision.ops import roi_align
import torch.nn as nn

class RoiAlign(nn.Module):
    def __init__(self, output_size=7, spatial_scale=1/16, sampling_ratio=2):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, feature_map, proposals):
        # proposals: Tensor[N, 4] in image coordinates
        proposals_feat = proposals / (1 / self.spatial_scale)

        # roi_align expects a list of tensors
        boxes = [proposals_feat]

        roi_features = roi_align(
            feature_map,
            boxes,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio
        )

        return roi_features
