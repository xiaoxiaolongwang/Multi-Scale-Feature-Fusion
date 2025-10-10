# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmcv.cnn import build_conv_layer
from mmdet.models import ResNet
from mmdet.models.builder import BACKBONES
from torch import Tensor
import torch.nn as nn
import torch


@BACKBONES.register_module()
class ResNetWithMetaConv(ResNet):
    """ResNet with `meta_conv` to handle different inputs in metarcnn and
    fsdetview.

    When input with shape (N, 3, H, W) from images, the network will use
    `conv1` as regular ResNet. When input with shape (N, 4, H, W) from (image +
    mask) the network will replace `conv1` with `meta_conv` to handle
    additional channel.
    """

    def __init__(self, **kwargs) -> None:
        # 使用kwargs确保父类被正确初始化
        super().__init__(**kwargs)
        # 定义元卷积

        self.meta_conv = build_conv_layer(
            self.conv_cfg,  # from config of ResNet
            4,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        # 定义全局池化
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        # 定义激活函数
        self.sig=nn.Sigmoid()
        self.convS3=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.convS4=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))

    # 对于S3的通道处理
    def S3(self,x:Tensor):
        # 获得池化量
        gap=self.gap(x)
        gmp=self.gmp(x)
        # 将统计量相加
        channel_stats=gap+gmp
        # sigmoid激活函数获得通道注意力
        CA=self.sig(channel_stats)
        return self.convS3(CA*x)

    # 对于S4的通道处理
    def S4(self,x:Tensor):
        # 获得池化量
        gap=self.gap(x)
        gmp=self.gmp(x)
        # 将统计量相加
        channel_stats=gap+gmp
        # sigmoid激活函数获得通道注意力
        CA=self.sig(channel_stats)
        return self.convS4(CA*x)
    # metarcnn主干网络调用的前向传播方法
    def forward(self, x: Tensor, use_meta_conv: bool = False) -> Tuple[Tensor]:
        """Forward function.

        When input with shape (N, 3, H, W) from images, the network will use
        `conv1` as regular ResNet. When input with shape (N, 4, H, W) from
        (image + mask) the network will replace `conv1` with `meta_conv` to
        handle additional channel.

        Args:
            x (Tensor): Tensor with shape (N, 3, H, W) from images
                or (N, 4, H, W) from (images + masks).
            use_meta_conv (bool): If set True, forward input tensor with
                `meta_conv` which require tensor with shape (N, 4, H, W).
                Otherwise, forward input tensor with `conv1` which require
                tensor with shape (N, 3, H, W). Default: False.

        Returns:
            tuple[Tensor]: Tuple of features, each item with
                shape (N, C, H, W).
        """
        # 重写了前向传播的方法
        # 颈部卷积
        if use_meta_conv:
            x = self.meta_conv(x)
        else:
            x = self.conv1(x)
        # 装饰器的作用
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 列表
        outs = []
        # 循环取出残差层的名字
        for i, layer_name in enumerate(self.res_layers):
            # 获得残差块的引用
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            # 选择输出哪些残差块的输出
            if i in self.out_indices:
                outs.append(x)
                # import pdb; pdb.set_trace()
        # for i,x in enumerate(outs):
        #     outs[i]=self.share_after_resbackbone(x)

        # 对于S3与S4进行通道注意力处理
        outs[0]=self.S3(outs[0])
        outs[1]=self.S4(outs[1])

        # tuple方法将列表强转为元组
        return tuple(outs)
