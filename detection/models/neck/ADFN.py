import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import NECKS


class ChannelAttention(nn.Module):
    """通道注意力模块 (用于特征预处理阶段)."""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力模块 (用于特征融合阶段)."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


@NECKS.register_module()
class AdaptiveDenseFusionPyramid(nn.Module):
    """Adaptive Dense Fusion Pyramid (ADFP)
    替代传统 FPN 的多尺度特征融合模块。
    包含两阶段：
      1. 通道注意力 + 1x1 conv 的特征预处理
      2. 密集连接 + 自适应通道-空间加权融合
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1):
        super(AdaptiveDenseFusionPyramid, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level if end_level != -1 else self.num_ins


        self.preprocess_convs = nn.ModuleList()
        for c in in_channels:
            self.preprocess_convs.append(
                nn.Sequential(
                    ChannelAttention(c),
                    nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0)
                )
            )

        # === Feature Fusion Stage ===
        self.fusion_convs = nn.ModuleList()
        self.spatial_att = SpatialAttention()

        for i in range(self.start_level, self.end_level - 1):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # Step 1: Feature Preprocessing with Channel Attention
        processed_feats = [conv(x) for conv, x in zip(self.preprocess_convs, inputs)]

        # Step 2: Dense Feature Fusion
        fused_feats = [processed_feats[0]]
        for i in range(1, len(processed_feats)):
            higher = F.interpolate(processed_feats[i], size=processed_feats[i - 1].shape[2:], mode='nearest')
            concat_feat = torch.cat([processed_feats[i - 1], higher], dim=1)
            fused = self.fusion_convs[i - 1](concat_feat)
            spatial_weight = self.spatial_att(fused)
            fused = fused * spatial_weight
            fused_feats.append(fused)


        outs = []
        for i in range(self.num_outs):
            if i < len(fused_feats):
                outs.append(fused_feats[i])
            else:
                outs.append(F.max_pool2d(fused_feats[-1], 1, stride=2))
        return tuple(outs)
