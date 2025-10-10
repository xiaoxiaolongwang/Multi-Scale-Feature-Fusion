# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import BBoxHead
from torch import Tensor


@HEADS.register_module()
class MetaBBoxHead(BBoxHead):
    """BBoxHead with meta classification for metarcnn and fsdetview.

    Args:
    
    元分类 类别个数
        num_meta_classes (int): Number of classes for meta classification.
        
    元分类的输入通道（应该是共享头之后的结果）
        meta_cls_in_channels (int): Number of support feature channels.
        
    是否使用元分类损失，默认为真
        with_meta_cls_loss (bool): Use meta classification loss.
            Default: True.
            
    元分类损失权重，默认为None
        meta_cls_loss_weight (float | None): The loss weight of `loss_meta`.
            Default: None.
    
    元分类损失配置
        loss_meta (dict): Config for meta classification loss.
    """

    def __init__(self,
                 # 这些标出来的参数都是关于支持集分支的
                 num_meta_classes: int,
                #  2048
                 meta_cls_in_channels: int = 2048,
                 with_meta_cls_loss: bool = True,
                 # 配置中没有传入这部分，元分类损失损失权重
                 meta_cls_loss_weight: Optional[float] = None,
                 # 1
                 loss_meta: Dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 # 这部分应该是rcnn的部分
                 *args,
                 **kwargs) -> None:
        # 使用父类进一步初始化rcnn
        super().__init__(*args, **kwargs)
        
        # 网络是否带有元分类损失分支
        self.with_meta_cls_loss = with_meta_cls_loss
        
        if with_meta_cls_loss:
            # 全连接层，输出单元个数为分类数量
            self.fc_meta = nn.Linear(meta_cls_in_channels, num_meta_classes)
            # 元分类损失权重
            self.meta_cls_loss_weight = meta_cls_loss_weight
            # 构建元分类损失对象
            # 此类位置miniconda3/envs/openmmlab2/lib/python3.7/site-packages/mmdet/models/losses/cross_entropy_loss.py
            self.loss_meta_cls = build_loss(copy.deepcopy(loss_meta))

    # 支持集的前向传播方法（我觉得共享层在这之前应该被使用）
    def forward_meta_cls(self, support_feat: Tensor) -> Tensor:
        """Forward function for meta classification.

        Args:
            support_feat (Tensor): Shape of (N, C, H, W).

        Returns:
            Tensor: Box scores with shape of (N, num_meta_classes, H, W).
        """
        # 线性层的最后输出为(N, num_meta_classes)
        meta_cls_score = self.fc_meta(support_feat)
        return meta_cls_score

    # 其实每一个网络输出结果之后都会计算损失，上边只是定义了一个损失的类型，这个方法利用损失对象返回损失值
    @force_fp32(apply_to='meta_cls_score')
    def loss_meta(self,
                  meta_cls_score: Tensor,
                  meta_cls_labels: Tensor,
                  meta_cls_label_weights: Tensor,
                  reduction_override: Optional[str] = None) -> Dict:
        """Meta classification loss.

        Args:
            meta_cls_score (Tensor): Predicted meta classification scores
                 with shape (N, num_meta_classes).
            meta_cls_labels (Tensor): Corresponding class indices with
                shape (N).
            meta_cls_label_weights (Tensor): Meta classification loss weight
                of each sample with shape (N).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.

        Returns:
            Dict: The calculated loss.
        """
        losses = dict()
        if self.meta_cls_loss_weight is None:
            loss_weight = 1. / max(
                torch.sum(meta_cls_label_weights > 0).float().item(), 1.)
        else:
            loss_weight = self.meta_cls_loss_weight
        if meta_cls_score.numel() > 0:
            loss_meta_cls_ = self.loss_meta_cls(
                meta_cls_score,
                meta_cls_labels,
                meta_cls_label_weights,
                reduction_override=reduction_override)
            losses['loss_meta_cls'] = loss_meta_cls_ * loss_weight
            losses['meta_acc'] = accuracy(meta_cls_score, meta_cls_labels)
        return losses
