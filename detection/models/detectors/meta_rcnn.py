# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

import torch
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.models.builder import DETECTORS
from torch import Tensor

from .query_support_detector import QuerySupportDetector


@DETECTORS.register_module()
class MetaRCNN(QuerySupportDetector):
    """Implementation of `Meta R-CNN.  <https://arxiv.org/abs/1909.13032>`_.

    Args:
        backbone (dict): Config of the backbone for query data.
        neck (dict | None): Config of the neck for query data and
            probably for support data. Default: None.
        support_backbone (dict | None): Config of the backbone for
            support data only. If None, support and query data will
            share same backbone. Default: None.
        support_neck (dict | None): Config of the neck for support
            data only. Default: None.
        rpn_head (dict | None): Config of rpn_head. Default: None.
        roi_head (dict | None): Config of roi_head. Default: None.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        pretrained (str | None): model pretrained path. Default: None.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 # 主干是一定要有的
                 backbone: ConfigDict,
                 
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None,
                 **parameter) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            
            rpn_head=rpn_head,
            roi_head=roi_head,
            
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            
            pretrained=pretrained,
            init_cfg=init_cfg,
            **parameter)

        self.is_model_init = False
        # 为了模型初始化保存的支持集临时特征图,这个字典在forward_model_init该方法中使用
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        # 测试的模型初始化只需要图片和标签就可以,在看完forward_model_init和model_init这两个方法后应该可以理解
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'roi_feats': [],
        }
        # 保存处理过的支持模板特征用于推理，处理过的支持模板特征在：func: ‘ model_init ’中生成
        # 内部装有'class_id' : tensor(1,C),C为通道数而不是类别数,一共15个class_id
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        # 类关注推断向量
        self.inference_support_dict = {}

    # 提取支持集的特征图(主干网络和颈部)
    @auto_fp16(apply_to=('img', ))
    def extract_support_feat(self, img):
        """Extracting features from support data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of input image, each item with shape
                (N, C, H, W).
        """
        # 特征图为tuple(outs),其中保存的阶段数由 'out_indices': (2,) 指定,索引由0开始
        feats = self.backbone(img, use_meta_conv=True)
        if len(feats)==1:
            feats=(feats[0],)
        else:
            feats=(feats[1],)
        # 颈部先跳过,等加的时候再看(不对,你为什么会有通道数不一致的疑惑呢,
        # 你没有看清楚这个变量名是support_neck而不是neck,普通的颈部会出现这样的问题,
        # 我觉得mmdetection的默认设定可能是经过支持集颈部之后只会留下一个tensor,不然标签个数跟特征图的总N对不上)
        # 后续看了一下代码,发现其实支持集颈部其实是没有定义的,所以说支持集图像不会经过普通的颈部而只会经过主干网络
        # (我认为这里也会是一个创新点)
        if self.support_neck is not None:
            feats = self.support_neck(feats)
        # 返回特征图
        return feats

    # 当mode为'model_init'时模型初始化的前向传播方法
    # 从代码可以看出来,当进行模型初始化的时候,程序会不断地从dataloader中取出数据,每次都向
    # self._forward_saved_support_dict中添加元素
    # 最后在调用model_init方法的时候,将这些tensor列表整合起来,计算出每个类别的原型用以测试
    # 在model_init方法中self._forward_saved_support_dict被清空
    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                        #    需要注意的是,此处列表的内容是图片类别索引而不是 图像注释列表的位置索引
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        """extract and save support features for model initialization.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.

        Returns:
            dict: A dict contains following keys:

                - `gt_labels` (Tensor): class indices corresponding to each
                    feature.
                - `res5_rois` (list[Tensor]): roi features of res5 layer.
        """
        # `is_model_init` flag will be reset when forward new data.
        # 当转发新数据时，‘ is_model_init ’标志将被重置。不懂
        self.is_model_init = False
        # 这里不是被DC修饰的变量吗?我需要再好好看看(应该是经过了一个什么方法将数据取出来了,不做要求)
        assert len(gt_labels) == img.size(
            0), 'Support instance have more than two labels'
        
        # 获得数据的特征图(内含主干网络与颈部)
        # 返回值是一个列表(其中的元素个数与主干网络或者是颈部的输出特征图数量一致)
        # [tensor(N,C,H,W),]
        feats = self.extract_support_feat(img)

        # 调用共享头的支持集前向传播方法(此处共享头为残差网络的第四阶段)
        # forward_support方法接受一个列表并返回一个列表
        # 列表中的tensor形状由(N, C, H, W)变为(N, C).
        # [tensor(N,C),]
        roi_feat = self.roi_head.extract_support_feats(feats)
        # gt_labels列表长度为N,[tensor(1),]
        # 向 前向传播保存支持集字典 的'gt_labels'与'roi_feats'字段的列表添加值
        # extend之后的'gt_labels'列表为[tensor(1),tensor(1),tensor(1),...]
        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        # extend之后的'roi_feats'列表为[tensor(N,C),tensor(N,C),tensor(N,C),...]
        self._forward_saved_support_dict['roi_feats'].extend(roi_feat)
        # 返回字典
        return {'gt_labels': gt_labels, 'roi_feat': roi_feat}

    # 此方法整合self._forward_saved_support_dict字典中所有的标签和特征图,得到所有类别的类原型(使用所有特征图平均池化出15个类原型)
    def model_init(self):
        """process the saved support features for model initialization."""
        # 将'gt_labels'的列表整合为一整个tensor([clas_id,clas_id,clas_id,...])
        #    需要注意的是,此处列表的内容是图片类别索引而不是 图像注释列表的位置索引
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        # 将'roi_feats'的列表整合为一整个tensor(N*num,C)
        roi_feats = torch.cat(self._forward_saved_support_dict['roi_feats'])
        # 获得类别索引的集合(去除所有重复元素)
        class_ids = set(gt_labels.data.tolist())
        # 清空字典
        self.inference_support_dict.clear()
        # 这部分其实是计算类别原型(平均)

        # 从本batch的物体种类中取出类别索引
        for class_id in class_ids:
            # 修改字典,roi_feats[gt_labels == class_id]是一个切片操作,获得特征图中属于class_id的的部分
            # mean([0], True)是一个平均操作,[0]意思是沿着batch维度池化,True意思是保持原数据形状不变
            # 即结果将是一个 (1, C,) 的张量，而不是 (C,),这里的C指的是通道数量,而不是类别数
            # gt_labels是一个列表,列表和一个整形数比较会得到一个bool列表,利用这个bool列表进行切片操作
            self.inference_support_dict[class_id] = roi_feats[
                gt_labels == class_id].mean([0], True)
        # set the init flag
        # 模型初始化标志为True
        self.is_model_init = True
        # reset support features buff
        # 清空缓存字典
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()

    # metarcnn的简单测试方法
    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        """Test without augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            通常，这些应该以均值为中心并按标准缩放。
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor] | None): override rpn proposals with
                custom proposals. Use when `with_rpn` is False. Default: None.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # 模型一个带有边界框回归头
        assert self.with_bbox, 'Bbox head must be implemented.'
        # 只支持单图像的推理
        assert len(img_metas) == 1, 'Only support single image inference.'
        # 确保模型初始化
        if not self.is_model_init:
            # process the saved support features
            self.model_init()
        
        # 调用提取特征方法(内部调用了self.extract_query_feat)
        # (只经过主干网络和颈部)(单图像)
        # 返回值时这样的list[Tensor],其中list的长度是颈部的输出级别个数
        query_feats = self.extract_feat(img)

        # 进入此分支
        if proposals is None:
            # 调用rpn的简单测试方法
            # 该方法在base_dease_head.py中实现
            # 返回值为[tensor,tensor],tensor的形状为(M, 5),经过了nms,第二个维度的第五值是置信度分数
            # 第二次看完内部代码之后发现这里的列表的长度与批次数是一致的,
            proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals
        
        # 调用roi的简单测试方法
        # 此方法返回一个嵌套列表,[[numpy arrays...],[numpy arrays...]]
        # 外列表的长度等于本批次图片的数量
        # 需要说明的是每个内列表都有15个numpy arrays,这些numpy arrays为(边界框,置信度)
        # 每一个numpy arrays都代表一张图片的一个类别的边界框信息
        return self.roi_head.simple_test(
            # 主干网络以及颈部提取的特征图
            query_feats,
            # 类关注推断向量字典
            copy.deepcopy(self.inference_support_dict),
            # rpn提出的所有图片的感兴趣区域(M, 5)
            proposal_list,
            # 图片的元信息
            img_metas,
            # 不映射回original image space
            rescale=rescale)
