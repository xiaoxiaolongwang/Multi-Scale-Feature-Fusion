import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from mmdet.models.builder import ROI_EXTRACTORS


class ChannelReorgAttention(nn.Module):
    """简单的通道重组注意力：1x1 -> ReLU -> 1x1 -> Sigmoid"""
    def __init__(self, channels, reduction=16):
        super(ChannelReorgAttention, self).__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (N, C, H, W)
        # 全局池化后产生通道权重，再逐通道放大
        avg = F.adaptive_avg_pool2d(x, 1)
        w = self.fc(avg)
        return x * w


@ROI_EXTRACTORS.register_module()
class CrossLevelFusionRoIExtractor(nn.Module):
    """
    Cross-level fusion RoI extractor:
      - extract RoI features from two selected pyramid levels (e.g. N3 and N4),
      - concatenate along channel dim,
      - compress with a 1x1 conv (implements hierarchical attention by learning channel weights),
      - optionally apply a channel reorganization attention to enhance representation.

    Args:
        in_channels (int): channels of each selected feature map (assumed same for both levels).
        out_channels (int): desired output channels after 1x1 compression.
        featmap_strides (list[int]): strides of the feature maps (e.g. [8, 16, 32, 64]).
        roi_size (tuple): (H, W) output size of RoI pooling (default (7,7) like Faster R-CNN).
        level_indices (tuple): indices (relative to feats list) to use for cross-level extraction,
                               e.g. (2, 3) for using N3, N4 when feats = [N1,N2,N3,N4,...].
        use_channel_attention (bool): whether to apply the dynamic reorganization channel attention.
        pooling_mode (str): 'align' or 'pool' (uses torchvision.ops.roi_align for 'align').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 featmap_strides,
                 roi_size=(7, 7),
                 level_indices=(2, 3),
                 use_channel_attention=True,
                 pooling_mode='align',
                 sampling_ratio=2):
        super(CrossLevelFusionRoIExtractor, self).__init__()
        assert pooling_mode in ('align', 'pool')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.roi_size = roi_size
        self.level_indices = tuple(level_indices)
        self.featmap_strides = featmap_strides
        self.use_channel_attention = use_channel_attention
        self.pooling_mode = pooling_mode
        self.sampling_ratio = sampling_ratio

        # 1x1 conv to compress concatenated (2*C) -> out_channels
        self.compress_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if use_channel_attention:
            self.reorg_att = ChannelReorgAttention(out_channels)
        else:
            self.reorg_att = None

    def _roi_pool_single_level(self, feat, proposals, spatial_scale):
        """
        Use torchvision.ops.roi_align to extract RoIs for a batch.
        proposals: list[Tensor] (len = batch_size), each Tensor is (num_boxes_i, 4) in (x1,y1,x2,y2) image coords.
        spatial_scale: float = 1.0 / stride (to convert image coords -> feat coords)
        Returns: Tensor (n_total, C, H, W)
        """
        # torchvision.ops.roi_align accepts boxes as a list[Tensor] of boxes per image (in absolute coords),
        # but with a given spatial_scale parameter it does the scaling internally.
        # roi_align returns (K, C, H, W) where K = total boxes across batch.
        # Here we call it directly.
        if self.pooling_mode == 'align':
            return roi_align(feat, proposals, output_size=self.roi_size,
                             spatial_scale=spatial_scale, sampling_ratio=self.sampling_ratio)
        else:
            # fallback: use roi_align as pool (no interpolation sampling)
            return roi_align(feat, proposals, output_size=self.roi_size,
                             spatial_scale=spatial_scale, sampling_ratio=-1)

    def forward(self, feats, proposals):
        """
        Args:
            feats (list[Tensor]): feature maps from neck/backbone, each of shape (B, C_i, H_i, W_i).
            proposals (list[Tensor] or Tensor): proposals for the batch.
                - Preferred: list[Tensor] with length B (batch size), each Tensor (num_props, 4) in (x1,y1,x2,y2) image coords.
                - Alternatively: a single Tensor of shape (num_props_total, 5) with (batch_idx, x1,y1,x2,y2).
                  This implementation will standardize to list[Tensor] form when possible.
        Returns:
            fused_rois: Tensor of shape (n_total, out_channels, H, W)
        """
        # Normalize proposals to list[Tensor] per image in image coords.
        # If user provided Tensor with batch_idx, convert to list:
        if isinstance(proposals, torch.Tensor):
            if proposals.numel() == 0:
                return torch.zeros(0, self.out_channels, self.roi_size[0], self.roi_size[1],
                                   device=feats[0].device)
            # assume proposals shape (K,5) where first col is batch_idx
            if proposals.size(1) == 5:
                # split by batch idx
                device = proposals.device
                batch_size = int(proposals[:, 0].max().item()) + 1
                props_per_image = []
                for b in range(batch_size):
                    mask = (proposals[:, 0] == b)
                    if mask.sum() == 0:
                        props_per_image.append(torch.zeros((0, 4), device=device))
                    else:
                        props_per_image.append(proposals[mask][:, 1:5])
                proposals_list = props_per_image
            else:
                raise ValueError("If proposals is a Tensor it must be (K,5) with batch_idx as first column.")
        else:
            # assume list[Tensor]
            proposals_list = proposals

        # pick the two levels specified (e.g. indices 2 and 3 for N3 and N4)
        idx0, idx1 = self.level_indices
        feat0 = feats[idx0]  # (B, C, H0, W0)
        feat1 = feats[idx1]  # (B, C, H1, W1)

        # compute spatial scales from strides
        stride0 = self.featmap_strides[idx0]
        stride1 = self.featmap_strides[idx1]
        spatial_scale0 = 1.0 / stride0
        spatial_scale1 = 1.0 / stride1

        # extract RoIs from each level
        # roi_align expects proposals as a list[Tensor(batch_i_boxes, 4)] in (x1,y1,x2,y2) relative to the original image.
        rois_lvl0 = self._roi_pool_single_level(feat0, proposals_list, spatial_scale0)  # (n_total, C, H, W)
        rois_lvl1 = self._roi_pool_single_level(feat1, proposals_list, spatial_scale1)  # (n_total, C, H, W)

        # concat along channel
        fused = torch.cat([rois_lvl0, rois_lvl1], dim=1)  # (n_total, 2*C, H, W)

        # compress with 1x1 conv to out_channels
        compressed = self.compress_conv(fused)  # (n_total, out_channels, H, W)

        # dynamic reorganization attention (optional)
        if self.reorg_att is not None:
            compressed = self.reorg_att(compressed)

        return compressed
