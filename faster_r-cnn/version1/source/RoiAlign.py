from torchvision.ops import roi_align
import torch.nn as nn

class RoiAlign(nn.Module):
    def __init__(self, output_size=7, spatial_scale=1/32, sampling_ratio=2):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, feature_map, proposals):
    # proposals: Tensor[N, 4] w skali obrazu (x1, y1, x2, y2)
    # boxes musi być listą tensorów - po jednym tensorze na każdy obraz w batchu
    # Jeśli trenujesz na batch_size=1, to [proposals] jest ok.
    
        roi_features = roi_align(
            feature_map,
            [proposals], # Tutaj podajemy ramki w skali obrazu
            output_size=self.output_size,
            spatial_scale=self.spatial_scale, # Tutaj musi być 1/32 (jeśli ResNet ma stride 32)
            sampling_ratio=self.sampling_ratio
        )
        return roi_features