# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmdet.models.builder import SHARED_HEADS
from mmdet.models.roi_heads import ResLayer
from torch import Tensor


@SHARED_HEADS.register_module()
# miniconda3/envs/openmmlab2/lib/python3.7/site-packages/mmdet/models/roi_heads/shared_heads/res_layer.py
# 共享头只接受一种通道大小的特征图
class MetaRCNNResLayer(ResLayer):
    """Shared resLayer for metarcnn and fsdetview.

    It provides different forward logics for query and support images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 最大池化，卷积核大小为2，步长默认与卷积核大小相同
        self.max_pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    # 查询集的前向传播方法
    # 共享头输出结果就是向量，平均池化完之后的结果
    def forward(self, x: Tensor) -> Tensor:
        """Forward function for query images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        # 获取共享层(单独的一个残差阶段)对象
        res_layer = getattr(self, f'layer{self.stage + 1}')
        # res_layer是网络的主体卷积部分,我认为这是共享头网络的核心
        out = res_layer(x)
        # 先在第四个维度上平均池化，然后在第三个维度上平均池化，最后的结果的形状会是[batch_size,num_channl,1,1]
        out = out.mean(3).mean(2)
        return out

    # 对于支持集，先进行卷积核为2的最大池化
    def forward_support(self, x: Tensor) -> Tensor:
        """Forward function for support images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        x = self.max_pool(x)
        # 获取共享层(单独的一个残差阶段)对象
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        out = self.sigmoid(out)
        out = out.mean(3).mean(2)
        return out
