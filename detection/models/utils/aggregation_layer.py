# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.utils import ConfigDict
from mmdet.models.builder import MODELS
from torch import Tensor

# AGGREGATORS are used for aggregate features from different data
# pipelines in meta-learning methods, such as attention rpn.
AGGREGATORS = MODELS


def build_aggregator(cfg: ConfigDict) -> nn.Module:
    """Build aggregator."""
    return AGGREGATORS.build(cfg)


@AGGREGATORS.register_module()
class AggregationLayer(BaseModule):
    """Aggregate query and support features with single or multiple aggregator.
    Each aggregator return aggregated results in different way.

    Args:
        aggregator_cfgs (list[ConfigDict]): List of fusion function.
        init_cfg (ConfigDict | None): Initialization config dict. Default: None
    """

    def __init__(self,
                 aggregator_cfgs: List[ConfigDict],
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.aggregator_list = nn.ModuleList()
        self.num_aggregators = len(aggregator_cfgs)
        aggregator_cfgs_ = copy.deepcopy(aggregator_cfgs)
        for cfg in aggregator_cfgs_:
            self.aggregator_list.append(build_aggregator(cfg))

    def forward(self, query_feat: Tensor,
                support_feat: Tensor) -> List[Tensor]:
        """Return aggregated features of query and support through single or
        multiple aggregators.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with
                shape (N, C, H, W).

        Returns:
            list[Tensor]: List of aggregated features.
        """
        out = []
        for i in range(self.num_aggregators):
            out.append(self.aggregator_list[i](query_feat, support_feat))
        return out


@AGGREGATORS.register_module()
class DepthWiseCorrelationAggregator(BaseModule):
    """Depth-wise correlation aggregator.

    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
            Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert in_channels is not None, \
            "DepthWiseCorrelationAggregator require config of 'in_channels'."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_fc = with_fc
        if with_fc:
            assert out_channels is not None, 'out_channels is expected.'
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (1, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'

        feat = F.conv2d(
            query_feat,
            support_feat.permute(1, 0, 2, 3),
            groups=self.in_channels)
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DifferenceAggregator(BaseModule):
    """Difference aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        feat = query_feat - support_feat
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat

# SE模块的实现
class SEModule(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),  # Excitation: First FC
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),  # Excitation: Second FC
            nn.Sigmoid()  # Sigmoid to get weights between 0 and 1
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # Batch size, Channels, Height, Width
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Fully Connected Layers
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Multiply the attention weights with the input feature map
        return x * y

@AGGREGATORS.register_module()
# 点积聚集器
class DotProductAggregator(BaseModule):
    """Dot product aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
    使用全连接来完成点积,如果这样的话输入通道与输出通道不能为None,因为我们要确定全连接层的信息
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                #  2048
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                #  默认不使用全连接
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None,
                #  是否在聚集完成之后进行se通道注意力
                 with_SE: bool = False) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_SE=with_SE
        # 是否使用全连接
        self.with_fc = with_fc
        # 输入通道与输出通道
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 如果需要全连接需要构建全连接层
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        if with_SE:
            # 创建SE通道注意力模块
            self.SEblock=SEModule(in_channels)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        # 确保除了批次数之外剩余维度都相等
        assert query_feat.size()[1:] == support_feat.size()[1:], \
            'mismatch channel number between query and support features.'
        # 进行元素级别的乘法
        # 我猜这一步是每个查询集感兴趣特征向量与15个特征向量做元素级别的乘法
        # 2025.1.28好吧之前猜错了,这里其实是一个批次所有的感兴趣区域特征向量与一个支持集特征向量做元素级别的乘法
        feat = query_feat.mul(support_feat)
        # 如果有卷积则执行以下操作
        # 能看出来如果使用全连接的话,就没有支持集参与其中了
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            # 将feat去掉最后两个维度之后进行卷积
            feat = self.fc(feat.squeeze(3).squeeze(2))
            # 归一化于relu
            feat = self.norm(feat)
            feat = self.relu(feat)
        
        if self.with_SE:
            feat=self.SEblock(feat)
        # 返回聚集之后的特征图
        return feat

@AGGREGATORS.register_module()
# 点积相减拼接聚集器
class DotProductCatSubtractAggregator(BaseModule):
    """Dot product aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
    使用全连接来完成点积,如果这样的话输入通道与输出通道不能为None,因为我们要确定全连接层的信息
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                #  2048
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                #  默认不使用全连接
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # 是否使用全连接
        self.with_fc = with_fc
        # 输入通道与输出通道
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 如果需要全连接需要构建全连接层
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        # 确保除了批次数之外剩余维度都相等
        assert query_feat.size()[1:] == support_feat.size()[1:], \
            'mismatch channel number between query and support features.'
        # 进行元素级别的乘法
        # 我猜这一步是每个查询集感兴趣特征向量与15个特征向量做元素级别的乘法
        # 2025.1.28好吧之前猜错了,这里其实是一个批次所有的感兴趣区域特征向量与一个支持集特征向量做元素级别的乘法
        feat = query_feat.mul(support_feat)
        feat1=query_feat-support_feat
        # 如果有卷积则执行以下操作
        # 能看出来如果使用全连接的话,就没有支持集参与其中了
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            # 将feat去掉最后两个维度之后进行卷积
            feat = self.fc(feat.squeeze(3).squeeze(2))
            # 归一化于relu
            feat = self.norm(feat)
            feat = self.relu(feat)
        # 返回聚集之后的特征图
        return torch.cat((feat,feat1),dim=1)

@AGGREGATORS.register_module()
# 点积相减拼接聚集器
class DotProductCatAbsOfSubtractAggregator(BaseModule):
    """Dot product aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
    使用全连接来完成点积,如果这样的话输入通道与输出通道不能为None,因为我们要确定全连接层的信息
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                #  2048
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                #  默认不使用全连接
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # 是否使用全连接
        self.with_fc = with_fc
        # 输入通道与输出通道
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 如果需要全连接需要构建全连接层
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        # 确保除了批次数之外剩余维度都相等
        assert query_feat.size()[1:] == support_feat.size()[1:], \
            'mismatch channel number between query and support features.'
        # 进行元素级别的乘法
        # 我猜这一步是每个查询集感兴趣特征向量与15个特征向量做元素级别的乘法
        # 2025.1.28好吧之前猜错了,这里其实是一个批次所有的感兴趣区域特征向量与一个支持集特征向量做元素级别的乘法
        feat = query_feat.mul(support_feat)
        feat1=torch.abs(query_feat-support_feat)
        # 如果有卷积则执行以下操作
        # 能看出来如果使用全连接的话,就没有支持集参与其中了
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            # 将feat去掉最后两个维度之后进行卷积
            feat = self.fc(feat.squeeze(3).squeeze(2))
            # 归一化于relu
            feat = self.norm(feat)
            feat = self.relu(feat)
        # 返回聚集之后的特征图
        return torch.cat((feat,feat1),dim=1)