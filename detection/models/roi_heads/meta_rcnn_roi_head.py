# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple
import torch.nn as nn

import numpy as np
import torch
from mmcv.utils import ConfigDict
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS, build_neck
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor


@HEADS.register_module()
class MetaRCNNRoIHead(StandardRoIHead):
    # 父类位置 miniconda3/envs/openmmlab2/lib/python3.7/site-packages/mmdet/models/roi_heads/standard_roi_head.py
    # 这个父类是没有__init__方法的
    # 在父类的三个父类中，BaseRoIHead有__init__方法，另外两个我没看
    # miniconda3/envs/openmmlab2/lib/python3.7/site-packages/mmdet/models/roi_heads/base_roi_head.py
    """Roi head for `MetaRCNN <https://arxiv.org/abs/1909.13032>`_.

    Args:
        aggregation_layer (ConfigDict): Config of `aggregation_layer`.
            Default: None.
    """

    def __init__(self,
                 # 将这个参数传给父类但是在父类中没有使用他
                 # 虽然父类的__init__方法没有这个参数，但是这样写不会出错
                 aggregation_layer: Optional[ConfigDict] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert aggregation_layer is not None, \
            'missing config of `aggregation_layer`.'
        # metarcnn必须有聚集头
        # 之前跳过的部分居然在这里补回来了，泪目了
         # /root/autodl-tmp/mmfewshot/mmfewshot/detection/models/utils/aggregation_layer.py(32)__init__()
        self.aggregation_layer = build_neck(copy.deepcopy(aggregation_layer))

        # 创建一个全局平均池化
        self.GAP1=nn.AdaptiveAvgPool2d((1,1))
    # roi的训练前向传播方法
    def forward_train(self,
                      
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],

                    #   rpn的推荐边界框
                      proposals: List[Tensor],
                    # 查询集的一系列信息
                      query_img_metas: List[Dict],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],

                    # 这里是要使用支持集的标签来计算预测头重塑网络的损失
                      support_gt_labels: List[Tensor],

                    # 查询集忽略的真实边界框
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            proposals (list[Tensor]): List of region proposals with positive
                and negative pairs.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                query image, each item with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images, each item with shape (num_gts).
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images, each item with shape (1).

            query_gt_bboxes_ignore (list[Tensor] | None): Specify which
                bounding boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # 分配真实地物并且采样rpn的推荐框
        # rpn的推荐边界框配置为每张图片最大采样物体数为2000
        # assign gts and sample proposals
        # 创建采样结果字典
        sampling_results = []

        if self.with_bbox:
            # 获取图片的个数
            num_imgs = len(query_img_metas)

            # 此处可能是为每张图片都创建一个None值以供代码的正常运行
            if query_gt_bboxes_ignore is None:
                query_gt_bboxes_ignore = [None for _ in range(num_imgs)]

            # 遍历图片
            for i in range(num_imgs):
                # 得到分配结果,传参为
                # rpn对于这张图片的推荐框,
                # 本图片的查询集的真实地物边界框
                # 本图片的忽略的真实地物边界框
                # 本图片的查询集真实地物标签

                # 返回结果应该是一个类的实例,其中装有边界框的分配信息
                # 注释一下assign_result对象的属性(字段)
                # num_preds记录本次参与分配的边界框个数
                # num_gts(int)记录真实地物的个数
                # gt_inds预测框分配的真实地物标签 tensor向量(长度为预测框的数量)记住0为负样本
                # max_overlaps装有iou信息不用管
                # labels算是gt_inds的正样本替换标签吧(内部除了正样本之外其他都是背景-1)
                assign_result = self.bbox_assigner.assign(
                    proposals[i], query_gt_bboxes[i],
                    query_gt_bboxes_ignore[i], query_gt_labels[i])
                

                # self.bbox_sampler.sample方法获得上边分配完成之后的采样情况,采样的个数由训练配置中rcnn的采样器配置决定
                # 需要注意的一点是经过此方法之后assign_result对象中的属性值会发生变化,因为他会把真实地物添加到分配完成的信息当中
                # 意思是说内部对网络进行了一个'教'的操作,
                # 具体来说的话gt_inds前边会添加一个顺序数列,[0,0]变成[1,2,3,0,0],添加的个数由真实地物的个数决定
                # max_overlaps是添加了真实地物个数的1
                # labels会添加真实地物的标签比如说两个真实地物的标签是[15,15,15],原tensor由[-1,-1]变成[15,15,15,-1,-1]
                # num_gts与num_preds都会加3

                # 然后是注释这个实例中的字段
                # bboxes是一个tensor,指的是采样的结果,其中包含正负样本预测边界框,形状为[128,4]
                # neg_bboxes负样本边界框,在本例的第一张图片中形状为[115,4]
                # neg_inds 是上边这些负样本在bboxes中的索引
                # num_gts 为真实地物的个数 为2
                # pos_assigned_gt_inds为正样本边界框的匹配的真实地物索引,注意这个索引是从真实地物个数2中选取的,tensor长度为[13]
                # pos_bboxes为正样本边界框,形状为[13,4]
                # pos_gt_bboxes是这些正样本对应的真实地物边界框,形状为[13,4]我们发现内部有很多重复,这很好理解
                # pos_gt_labels正样本边界框对应的真实地物的标签,形状为[13]
                # pos_inds为正样本在这采样的128个预测框中的索引,形状为[13]
                # pos_is_gt这个参数比较有意思,说的是我们这些正样本有哪几个是由真实地物添加而来的,形状为[13]

                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposals[i],
                    query_gt_bboxes[i],
                    query_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in query_feats])
                # 向采样结果列表中添加本图片的采样结果对象
                sampling_results.append(sampling_result)

        # 创建损失字典
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            # 直接就是大结局了,传参为查询集特征图,支持集特征图,采样结果对象,查询集元信息
            # 查询集真实地物边界框,查询集真实地物标签,支持集真实地物标签

            # 注释结果bbox_results是一个字典 
            # 首先是两个键 
            # 'cls_score':tensor[128,16],这128个样本对于16个类别(加一个背景)的预测概率
            # 'bbox pred':tensor[128,60],这128个预测框样本对于15个类别的预测偏移值
            
            # 然后第三个键里边是一个字典
            # 'loss_bbox':dict{'loss_cls':tensor[]rcnn的分类损失
            #                  'loss_bbox':tensor[]rcnn的分类损失
            #                  'acc':tensor[]rcnn的准确率
            #                  'loss_meta_cls':tensor[]预测重塑网络的元分类损失
            #                  'meta_acc':tensor[1]预测重塑网络的分类精确率
            # }
            bbox_results = self._bbox_forward_train(
                query_feats, support_feats, sampling_results, query_img_metas,
                query_gt_bboxes, query_gt_labels, support_gt_labels)
            if bbox_results is not None:
                # 这里只要损失的部分
                losses.update(bbox_results['loss_bbox'])
        # dict{'loss_cls':tensor[]rcnn的分类损失
        #      'loss_bbox':tensor[]rcnn的分类损失
        #      'acc':tensor[]rcnn的准确率
        #      'loss_meta_cls':tensor[]预测重塑网络的元分类损失
        #      'meta_acc':tensor[1]预测重塑网络的分类精确率
        # }
        return losses

    def _bbox_forward_train(self, query_feats: List[Tensor],
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_img_metas: List[Dict],
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.

        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        # 关键是这部分需要修改,也就是当没有共享头的时候需要进行一次全局平均池化
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        support_feat = self.extract_support_feats(support_feats)[0]

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets
        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            random_index = np.random.choice(
                range(query_gt_labels[img_id].size(0)))
            random_query_label = query_gt_labels[img_id][random_index]
            for i in range(support_feat.size(0)):
                # Following the official code, each query image only sample
                # one support class for training. Also the official code
                # only use the first class in `query_gt_labels` as support
                # class, while this code use random one sampled from
                # `query_gt_labels` instead.
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat[i].unsqueeze(0))
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results['bbox_pred'],
                        query_rois[start:end], labels[start:end],
                        label_weights[start:end], bbox_targets[start:end],
                        bbox_weights[start:end])
                    for key in single_loss_bbox.keys():
                        loss_bbox[key].append(single_loss_bbox[key])
        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                else:
                    loss_bbox[key] = torch.stack(
                        loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    # 提取感兴趣区域特征的方法
    def extract_query_roi_feat(self, feats: List[Tensor],
                               rois: Tensor) -> Tensor:
        # 这个方法在训练与测试阶段都被使用
        """Extracting query BBOX features, which is used in both training and
        testing.

        Args:
        查询集多级别特征图(含批次,不止一张图片)
            feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            rois (Tensor): shape with (m, 5).

        Returns:
            Tensor: RoI features with shape (N, C).
        """
        # bbox_roi_extractor是网络的感兴趣区域特征图提取器对象,这在standaed_roi_head.py中绑定
        # num_inputs是一个被property修饰的方法
        # 用来获得我们在初始化感兴趣区域提取器的时候标明的应该传入的特征图个数
        # 这里的意思是我们只传入事先标明的特征图数量个数的特征图
        # 并传入一个感兴趣区域坐标tensor
        # roi_feats是一个tensor形状为(感兴趣区域个数,通道数1024,特征图预定高度,特征图预定宽度)
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        
        # 经过共享头之后形状变为(N, C),因为共享头中做了不保留维度的平均池化操作
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        else:
            roi_feats=self.GAP1(roi_feats).view(roi_feats.size(0),-1)

        
        # 返回一个形状为(N, C)的tensor
        return roi_feats

    # 提取支持集特征图
    # (这一步实际上是使主干网络颈部提取出来的特征图经过一次残差网络的最后一个阶段,在这里此类为MetaRCNNResLayer)
    # 接受一个装有tensor的列表
    def extract_support_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Forward support features through shared layers.

        Args:
            feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).

        Returns:
            list[Tensor]: List of support features, each item
                with shape (N, C).
        """
        out = []
        # 该方法定义在base_roi_head.py下,使用@property修饰
        if self.with_shared_head:
            # 遍历来自主干网络或者颈部的特征图列表
            for lvl in range(len(feats)):
                # 调用共享头的支持集前向传播方法(此处共享头为残差网络的第四阶段)
                # forward_support方法接受这样的tensor(N, C, H, W).返回这样的tensor(N, C).
                # 向out列表中添加这个tensor
                # 通过这边的lvl变量名的设计,我感觉程序在设计的时候是允许得到同一batch多个尺度的特征图列表的
                # 但问题是共享头只接受一种卷积核大小的特征图,
                # 所以说在使用fpn的时候应该注意让fpn所有层级的输出特征图的通道数与共享头第一个卷积层的输入通道数保持一致
                out.append(self.shared_head.forward_support(feats[lvl]))
        else:
            for lvl in range(len(feats)):
                out.append(self.GAP1(feats[lvl]).view(feats[lvl].size(0),-1))
        # 返回out列表
        return out

    # roihead的边界框前向传播方法
    # 此方法传参为查询集的感兴趣区域特征向量与单类别的支持集向量
    # 我看该方法的内部只获取本批次的所有感兴趣区域特征向量与支持集聚合之后的第一个感兴趣区域向量
    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        # 这个方法在训练与测试中均被使用
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        # feature aggregation
        # query_roi_feats.unsqueeze(-1).unsqueeze(-1)的unsqueeze方法意思是在原有tensor的末尾处添加一个大小为1的维度
        # 操作之后的形状变为(N,C,1,1)
        # 我们单支持集向量重塑为(1,2048,1,1)
        # 调用roi的特证聚合部分的前向传播方法,输出的结果会是(N,1024,1,1)
        # 这里百思不得其解的是为什么只要第一个感兴趣区域元素呢
        # 现在我大概理解了,该方法返回来的其实是一个列表,列表中装有一个形状为(N,1024,1,1)的tensor
        # 所以说[0]其实是取到这个tensor,也就是所有感兴趣区域的特征向量
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))[0]

        # 调用边界框预测头的前向传播方法,这里去掉了最后两个大小为1的维度
        # 需要说明的是在roi的边界框回归部分对于每个感兴趣区域来说会输出15*4个偏移信息
        # 这是因为我们的边界框是忽略类别不可见,双重否定为肯定,类别可见，
        # 每个感兴趣区域会为15个类别逗输出一份偏移信息
        # 这里传入的tensor的形状为(N,2048),这经过bbox_head的两个线性层输出所有
        # 查询集感兴趣区域特征向量的类别概率与边界框回归结果
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1))
        # 组织本批次的感兴趣区域的预测分数与边界框偏移预测信息字典
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)

        # 返回预测信息字典
        return bbox_results

    # 简单的测试方法
    def simple_test(self,
                    # 主干网络(颈部)提取出来的特征图
                    query_feats: List[Tensor],
                    # 类关注向量字典
                    support_feats_dict: Dict,
                    # rpn提取出来的感兴趣区域(M, 5),(x1,y1,x2,y2,置信度分数)
                    # 列表的长度与批次数相等
                    proposal_list: List[Tensor],
                    # 查询集图片的元信息
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            proposal_list (list[Tensors]): list of region proposals.
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results. Default: False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # 确保有回归头
        assert self.with_bbox, 'Bbox head must be implemented.'

        # 获得物体边界框与物体标签
        # 该方法返回两个装有tensor的列表,两个列表的长度都是本批次的总图片数
        # 第一个列表是(k,5)(边界框,置信度),第二个列表是(k,)(模型认为这些边界框的类别)
        det_bboxes, det_labels = self.simple_test_bboxes(
            # 查询集特征图列表
            query_feats,
            # 支持集特征字典
            support_feats_dict,
            # 查询集图像元信息列表
            query_img_metas,
            # rpn提取出来的感兴趣区域(M, 5),(x1,y1,x2,y2,置信度分数)
            # 列表的长度与批次数相等
            proposal_list,

            # 实例的测试配置,这在base_roi_head.py中绑定
            self.test_cfg,
            rescale=rescale)
        
        # 此处形成一个嵌套列表,[[numpy arrays],[numpy arrays]]
        # 需要说明的是每个内列表都有15个numpy arrays,这些numpy arrays为(边界框,置信度)
        # 每一个numpy arrays都代表一张图片的一个类别的边界框信息
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        # 返回这个嵌套列表
        return bbox_results

    # 实现一个简单测试的边界框方法
    def simple_test_bboxes(
            self,
            # 查询集特征图列表
            query_feats: List[Tensor],
            # 支持集类支持向量字典
            support_feats_dict: Dict,
            # 查询集图像元信息列表
            query_img_metas: List[Dict],
            # rpn提取出来的感兴趣区域列表(M, 5),(x1,y1,x2,y2,置信度分数)
            proposals: List[Tensor],

            # 实例的测试配置,这在base_roi_head.py中绑定
            rcnn_test_cfg: ConfigDict,
            # 不映射回原始图像尺寸
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        # 什么叫只测试没有增强的物体边界框
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).我真是个天才,看代码看出了这里的形状
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
        第一个列表中的tensor形状是(num_boxes, 4),第二个列表中的tensor形状是(num_boxes, ),这两个列表的长度应该跟batch_size相等
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        # 获取所有图片的形状(896, 600, 3)(h,w,c)
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        # 获取图片相对于原始图片的缩放因子[w_scale, h_scale, w_scale, h_scale]
        # 这个值的作用可能是方便后续的映射回原始图片域
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        # proposals为本批次所有图片的预测框tensor列表(x1,y1,x2,y2,置信度分数)
        # 列表的长度与批次数相等(本批次的总图片数)
        # 此方法将一个批次的预测框列表转换为一整个tensor,其中第一列标明了本预测框对应的是哪张图片
        # rois的第二个维度的大小为5(所属图片索引,四个坐标)
        rois = bbox2roi(proposals)

        # 提取查询集的感兴趣区域特征
        # 此方法在本文件下实现
        # 返回一个(N, C)的tensor,其中N为感兴趣区域的数量,这经过了感兴趣区域特征提取与共享头的处理
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)

        # 创建两个字典,分别是分类分数与边界框偏移预测
        cls_scores_dict, bbox_preds_dict = {}, {}
        # 获得类别个数15(从代码中看出来这是查询集的类别个数,虽然说查询集与支持集的类别个数都是一样的)
        num_classes = self.bbox_head.num_classes

        # 从支持集特征图字典中 获得支持集的键
        # 真是天才,如果是每次使用一个roi查询集特征向量与15个支持集特征向量,会大大增加时间的消耗
        # 这里的做法简直是天才
        # 每次使用单个支持集特征向量与整个批次的所有roi做元素级乘法,就能大大提升效率
        for class_id in support_feats_dict.keys():
            # 获取该键的特征向量
            support_feat = support_feats_dict[class_id]

            # 获得边界框的结果字典
            # 本方法在该文件下实现(传参为查询及感兴趣区域特征向量和支持集特征向量)
            # 此方法首先获得本批次所有感兴趣特征向量与类别class_id的特征向量聚合之后的结果(此处做了一次元素级乘法)
            # 然后将这个结果(形状为(N,2048)的tensor)传入roi的分类回归头中,
            # 以一个字典的形式返回输出分类结果(N,16)与边界框偏移结果(N,15*4)
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            
            # 从分类结果中找出该支持集对应的该批次的类别概率(N,1),在字典cls_scores_dict中以'class_id:tensor([N,1])'的形式存储起来
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            # 从边界框回归结果中找出该支持集对应的该批次类别的边界框回归概率(N,4),在字典bbox_preds_dict中以'class_id:tensor([N,4])'的形式存储起来
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
            # 经过上边的步骤我们就获得了在本支持集类别下的所有感兴趣区域特征向量的对用本支持集类别的分类以及边界框回归结果了
            

            # 官方代码与本代码的一些区别
            # the official code use the first class background score as final
            # background score, while this code use average of all classes'
            # background scores instead.
            # 应该是进入此分支,因为支持集字典就只有15个类别,索引只到14,肯定是找不到15的啊,返回值肯定是None的
            if cls_scores_dict.get(num_classes, None) is None:
                # 此处获得对于背景的预测概率(16个分数的最后一个分数)
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]
        # 代码到这里的时候cls_scores_dict中有16对键值,记录所有感兴趣区域属于每个类别的概率
        # bbox_preds_dict中有15对键值,记录所有感兴趣区域对于每个类别的偏移预测值

        # 此处将背景概率除以支持集总类别数15
        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())

        # 这是一个列表推导式,torch.zeros_like方法返回一个跟传入tensor形状与类型完全相同但是元素均为0的tensor
        # 但是我看代码else的情况永远也不会发生啊
        # 所以说这段代码的作用是从字典中取出取出本批次所有感兴趣区域特征向量所有类别的概率分数
        # 列表中会有16个tensor,形状都为(N,1),每个tensor都标示着一个类别
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        # 这段代码的作用是从字典中取出取出本批次所有感兴趣区域特征向量所有类别的边界框偏移预测值
        # 列表中会有15个tensor,形状都为(N,4),每个tensor都标示着一个类别
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        # 将类别概率与边界框偏移预测列表在维度1上做一次cat,这样的话两个tensor的形状分别为(N,16)与(N,15*4)
        # 使用这种方式获得了与不使用特征增强一样形状的结果(恍然大悟)
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        # split batch bbox prediction back to each image
        # 分隔批次边界框回到每张图片上

        # 我们需要知道的一点是proposals是一个装有所有图片的推荐边界框tensor的列表,这个列表的长度是N
        # 所以说num_proposals_per_img的长度也会是N
        # 而len方法如果施加到tensor上的话返回的是这个tensor第一个维度的大小
        # 所以说num_proposals_per_img是一个元组,元组中的每个元素表示第n个图片有多少个推荐区域
        num_proposals_per_img = tuple(len(p) for p in proposals)

        # 在这个例子中,对于tensor施加split方法的意思是,按照元组num_proposals_per_img的数字顺序
        # 比如说是(4,5,2),而rois原来是一个(11,5)的tensor,第二个参数0的意思是将这个tensor从第一个维度
        # 按照(4,5,2)的分割方式将tensor分成(tensor(4,5),tensor(5,5),tensor(2,5))的元组
        rois = rois.split(num_proposals_per_img, 0)

        # 下边两行也是一样的目的哈,关于一个疑惑为什么上边为什么不直接使用rois来分割,我的想法是这样的,
        # 这里已经是roi部分了,之前的置信度已经被更新了,所以说直接使用这里的新的置信度就可以了
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # 分别对每张图片使用后处理
        # apply bbox post-processing to each image individually
        det_bboxes = []
        # ？！！！！这里为什么会出现标签呢?你压根就没有载入啊,我倒要看看怎么个事
        det_labels = []

        # 遍历图片列表哈,循环次数就是图片数
        for i in range(len(proposals)):
            # 调用roi的后处理方法
            # 其中det_bbox是经过nms之后的边界框(k,5),det_label是这些边界框被默认分配的类别(k,),需要注意的是背景类别概率已经被忽略
            det_bbox, det_label = self.bbox_head.get_bboxes(
                # (所属图片索引,四个坐标)
                rois[i],
                # 16个概率值
                cls_score[i],
                # 15*4个边界框偏移值
                bbox_pred[i],
                # 该图片的形状(h,w,c)
                img_shapes[i],
                # 缩放因子(x方向上,y方向上,x方向上,y方向上)
                # 仔细想一下为什么会用到缩放因子?将预测框映射回原图
                scale_factors[i],
                # 看代码在测试阶段的时候也是全程将预测框映射回原图的
                rescale=rescale,
                # 测试的配置,这在base_roi_head.py中被绑定
                cfg=rcnn_test_cfg)
            # 向总物体边界框列表中添加这张图片的后处理边界框结果(边界框,置信度,模型认为这些边界框的标签)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        # 最后看起来det_bboxes与det_labels是等长的,长度都等于本批次的图片总数量
        # 其中每个tensor都代表一张图片的边界框与类别信息
        return det_bboxes, det_labels
