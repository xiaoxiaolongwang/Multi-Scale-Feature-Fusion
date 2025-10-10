# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import abstractmethod
from typing import Dict, List, Optional, Union

from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.models.builder import (DETECTORS, build_backbone, build_head,
                                  build_neck)
from mmdet.models.builder import (SHARED_HEADS, build_shared_head)
from mmdet.models.detectors import BaseDetector
from torch import Tensor
from typing_extensions import Literal


@DETECTORS.register_module()
class QuerySupportDetector(BaseDetector):
    """Base class for two-stage detectors in query-support fashion.

    Query-support detectors typically consisting of a region
    proposal network and a task-specific regression head. There are
    two pipelines for query and support data respectively.

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
                 backbone: ConfigDict,
                #  share_after_resbackbone: ConfigDict,
                 
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
        super().__init__(init_cfg)
        # import pdb; pdb.set_trace()

        # 预训练
        # 在这一步给主干网络添加预训练权重地址，能对字典使用这种写法在于ConfigDict类
        backbone.pretrained = pretrained
        
        # 主干网络实例化
        # 查询集的主干网络
        
        # ResNetWithMetaConv类地址autodl-tmp/mmfewshot/mmfewshot/detection/models/backbones/resnet_with_meta_conv.py
        # resnet类地址miniconda3/envs/openmmlab2/lib/python3.7/site-packages/mmdet/models/backbones/resnet.py
        # res_later地址miniconda3/envs/openmmlab2/lib/python3.7/site-packages/mmdet/models/utils/res_layer.py
        
        self.backbone = build_backbone(backbone)

        # 创建主干网络后的共享头
        # self.share_after_resbackbone=build_shared_head(share_after_resbackbone)
        # 给主干网络绑定之后的共享头部分
        # self.backbone.share_after_resbackbone=self.share_after_resbackbone

        
        # 我先不看，我复现的论文是没有颈部的
        self.neck = build_neck(neck) if neck is not None else None
        # if `support_backbone` is None, then support and query pipeline will
        # share same backbone.
        
        # 为支持集准备专门的主干网络（如果有的话）
        self.support_backbone = build_backbone(
            support_backbone
        ) if support_backbone is not None else self.backbone
        
        # support neck only forward support data.
        self.support_neck = build_neck(
            support_neck) if support_neck is not None else None
        
        
        # 确保有分类回归头
        assert roi_head is not None, 'missing config of roi_head'
        
        # when rpn with aggregation neck, the input of rpn will consist of
        # query and support data. otherwise the input of rpn only
        # has query data.
        # 如果rpn带有聚集颈部，rpn的输入将会考虑到查询集和支持集数据。否则rpn的输入只有查询集输入
        
        # 网络是否有rpn
        self.with_rpn = False

        # 网络是否有聚集颈部
        # 但是我看的论文好像只有roi头带有聚集颈部
        self.rpn_with_support = False
        
        if rpn_head is not None:
            # 网络有rpn
            self.with_rpn = True
            # rpn是否有聚集颈部
            if rpn_head.get('aggregation_layer', None) is not None:
                self.rpn_with_support = True
                
            # 定义训练时的rpn行为，由两个部分组成，包括如何在rpn所有建议框中界定正负样本，rpn在训练阶段每张图片应该有多少个建议框以及其中正负样本的比例
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            
            # 深拷贝字典rpn_head，其中有通道数，每个像素点的锚框生成，锚框的归一化，正负样本分类损失，锚框回归损失
            rpn_head_ = copy.deepcopy(rpn_head)
            
            # 分别向配置中添加rpn在训练与测试时的行为
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            # 你让我看代码 好像rpn_head_中是没有rpn_proposal这个字典的(后续打了断点，确实是没有的)
            # import pdb; pdb.set_trace()
            
            # 创建网络的rpn部分
            # RPNHead类地址 > /root/miniconda3/envs/openmmlab2/lib/python3.7/site-packages/mmdet/models/dense_heads/rpn_head.py(24)__init__()
            # 构建内容有：锚框生成器，预测框编码解码器，分类损失，回归损失，训练时的分配器，训练时的随机抽样器
            self.rpn_head = build_head(rpn_head_)

            
        # 这里能看出来分类回归头是一定要有聚集层的，因为这就是人家论文的创新点
        if roi_head is not None:
            # update train and test cfg here for now
            # rcnn的训练行为，包括分配器和采样器
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            
            # roi字典更新训练行为和测试行为
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            
            # 预训练权重地址
            # 向roi_heat中加入预训练地址
            roi_head.pretrained = pretrained
            
            
            # autodl-tmp/mmfewshot/mmfewshot/detection/models/roi_heads/meta_rcnn_roi_head.py
            self.roi_head = build_head(roi_head)

        # 训练配置与测试配置
        # 我们需要知道一件事情是这样的,在构建rpn与roi的时候需要使用测试配置与训练配置中的一些值
        # 但是这并没有修改原先的训练与测试配置,而forward_train(方法中的配置其实是来自于
        # 原本的配置信息而不是在rpn与roi中使用的那些配置信息
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # 提取查询集特征的方法(只经过主干网络和颈部)(单图像)
    @auto_fp16(apply_to=('img', ))
    def extract_query_feat(self, img: Tensor) -> List[Tensor]:
        """Extract features of query data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of support images, each item with shape
                 (N, C, H, W).
        """
        feats = self.backbone(img)
        if self.with_neck:
            feats = self.neck(feats)
        return feats

    # 提取特征方法(查询集)
    def extract_feat(self, img: Tensor) -> List[Tensor]:
        """Extract features of query data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of query images.
        """
        return self.extract_query_feat(img)

    @abstractmethod
    def extract_support_feat(self, img: Tensor):
        """Extract features of support data."""
        raise NotImplementedError

    @auto_fp16(apply_to=('img', ))
    def forward(self,
        # query_data:{img_metas:,img:,gt_bboxes:,gt_labels:},
        # support_data:{img_metas:,img:,gt_bboxes:,gt_labels:}
                query_data: Optional[Dict] = None,
                support_data: Optional[Dict] = None,

                img: Optional[List[Tensor]] = None,
                img_metas: Optional[List[Dict]] = None,
                # 这里的意思是mode必须要从这三个str中选一个,默认为train
                mode: Literal['train', 'model_init', 'test'] = 'train',
                **kwargs) -> Dict:
        """Calls one of (:func:`forward_train`, :func:`forward_test` and
        :func:`forward_model_init`) according to the `mode`. The inputs
        of forward function would change with the `mode`.

        - When `mode` is 'train', the input will be query and support data
        for training.

        - When `mode` is 'model_init', the input will be support template
        data at least including (img, img_metas).

        - When `mode` is 'test', the input will be test data at least
        including (img, img_metas).

        Args:
            query_data (dict): Used for :func:`forward_train`. Dict of
                query data and data info where each dict has: `img`,
                `img_metas`, `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore`.
                Default: None.
            support_data (dict): Used for :func:`forward_train`. Dict of
                support data and data info dict where each dict has: `img`,
                `img_metas`, `gt_bboxes`, `gt_labels`, `gt_bboxes_ignore`.
                Default: None.
            img (list[Tensor]): Used for func:`forward_test` or
                :func:`forward_model_init`. List of tensors of shape
                (1, C, H, W). Typically these should be mean centered
                and std scaled. Default: None.
            img_metas (list[dict]): Used for func:`forward_test` or
                :func:`forward_model_init`.  List of image info dict
                where each dict has: `img_shape`, `scale_factor`, `flip`,
                and may also contain `filename`, `ori_shape`, `pad_shape`,
                and `img_norm_cfg`. For details on the values of these keys,
                see :class:`mmdet.datasets.pipelines.Collect`. Default: None.
            mode (str): Indicate which function to call. Options are 'train',
                'model_init' and 'test'. Default: 'train'.
        """
        # 快结束了,一定要加油啊
        if mode == 'train':
        # dict{'loss_cls':tensor[]rcnn的分类损失
        #      'loss_bbox':tensor[]rcnn的分类损失
        #      'acc':tensor[]rcnn的准确率
        #      'loss_meta_cls':tensor[]预测重塑网络的元分类损失
        #      'meta_acc':tensor[1]预测重塑网络的分类精确率
        #       loss_rpn_cls=losses['loss_cls'], 
        #       loss_rpn_bbox=losses['loss_bbox']
        # }
            return self.forward_train(query_data, support_data, **kwargs)
        
        # mode是模型初始化的情况
        elif mode == 'model_init':
            # 调用模型初始化的前向传播方法
        # 最后的返回值是一个字典{
        #                       'img_metas':DataContainer[[dict,dict]], 
        #                     'img':DataContainer[tensor], 
        #                     'gt_bboxes':DataContainer[[tensor,tensor]], 
        #                     'gt_labels':DataContainer[[tensor,tensor]]
        #                                                                   }
            return self.forward_model_init(img, img_metas, **kwargs)
        # 当模型为test模式时
        elif mode == 'test':
            # 调用模型的测试前向传播方法(此方法在base文件中实现)
        # 此方法返回一个嵌套列表,[[numpy arrays...],[numpy arrays...]]
        # 外列表的长度等于本批次图片的数量
        # 需要说明的是每个内列表都有15个numpy arrays,这些numpy arrays为(边界框,置信度)
        # 每一个numpy arrays都代表一张图片的一个类别的边界框信息
            return self.forward_test(img, img_metas, **kwargs)
        else:
            raise ValueError(
                f'invalid forward mode {mode}, '
                f'only support `train`, `model_init` and `test` now')

    # 模型训练时必不可少的方法
    def train_step(self, data: Dict, optimizer: Union[object, Dict]) -> Dict:
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN. For most of query-support detectors, the
        batch size denote the batch size of query data.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        # 调用实例的前向传播方法
        # dict{ loss_rpn_cls=losses['loss_cls'], 
        #       loss_rpn_bbox=losses['loss_bbox']
        #       'loss_cls':tensor[]rcnn的分类损失
        #      'loss_bbox':tensor[]rcnn的分类损失
        #      'acc':tensor[]rcnn的准确率
        #      'loss_meta_cls':tensor[]预测重塑网络的元分类损失
        #      'meta_acc':tensor[1]预测重塑网络的分类精确率
        #       loss_rpn_cls=losses['loss_cls'], 
        #       loss_rpn_bbox=losses['loss_bbox']
        # }
        losses = self(**data)
        # 经过_parse_losses方法之后
        # loss是一个tensor,这个tensor将losses中的五个损失加起来了
        # log_vars是一个字典,其中的键与losses基本一致,多了一个loss的键,loss的值是把上边五个loss的值加起来了
        # 其中所有的值都是浮点数,为的是写入.log.json文件
        loss, log_vars = self._parse_losses(losses)

        # For most of query-support detectors, the batch size denote the
        # batch size of query data.
        # 返回一个字典
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['query_data']['img_metas']))

        return outputs

    def val_step(self,
                 data: Dict,
                 optimizer: Optional[Union[object, Dict]] = None) -> Dict:
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # For most of query-support detectors, the batch size denote the
        # batch size of query data.
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['query_data']['img_metas']))

        return outputs

    # 前向传播方法
    def forward_train(self,
        # query_data:{img_metas:,img:,gt_bboxes:,gt_labels:},
        # support_data:{img_metas:,img:,gt_bboxes:,gt_labels:}
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 获取图片
        query_img = query_data['img']
        support_img = support_data['img']

        # list[Tensor]: Features of support images, each item with shape (N, C, H, W).
        # 其中经过了主干与颈部  列表的长度为颈部的数量(如果有颈部的话)
        query_feats = self.extract_query_feat(query_img)

        # list[Tensor]: Features of input image, each item with shape (N, C, H, W).
        # 这经过了主干网络与支持集颈部
        support_feats = self.extract_support_feat(support_img)

        # 创建损失字典
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            # 从总体的训练配置中获得rpn的推荐配置
            # dict(
            # nms_pre=12000,
            # max_per_img=2000,
            # nms=dict(type='nms', iou_threshold=0.7),
            # min_bbox_size=0)
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # 不进入此分支,rpn是没有聚集层配置的
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats,
                    support_feats,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            

            # 进入此分支
            else:
                # 进行rpn的训练的前向传播
                # 返回rpn的损失与推荐边界框列表
                # rpn的损失以字典的形式返回,这个字典中的两个值都是列表,每个列表中的元素个数与颈部的输出特征图个数相等
                # 这里的意思其实是代码对于损失的实现是每个级别的特征图都单独给出损失,并没有说把损失都加起来
                # dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

                # proposal_list为[tensor,tensor]
                # 列表的长度为批次的图片个数,tensor的第二维度大小为5,前四个是坐标,最后一个值是置信度
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    # 查询集特征图列表
                    query_feats,
                    # 查询集图像的元信息
                    copy.deepcopy(query_data['img_metas']),
                    # 查询集图像的真实边界框信息
                    copy.deepcopy(query_data['gt_bboxes']),
                    # 这里为什么不用传入标签呢?你仔细想一想训练的rpn只需要分辨正负样本就行了,所以是不需要标签的
                    gt_labels=None,
                    # 在本例中查询集使用难例,所以这里应该没有信息
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    # rpn的推荐配置
                    # dict(
                    # nms_pre=12000,
                    # max_per_img=2000,
                    # nms=dict(type='nms', iou_threshold=0.7),
                    # min_bbox_size=0)
                    proposal_cfg=proposal_cfg)
                
            # 使用这个字典更新损失字典
            # dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # roi的训练时的前向传播方法
        # dict{'loss_cls':tensor[]rcnn的分类损失
        #      'loss_bbox':tensor[]rcnn的分类损失
        #      'acc':tensor[]rcnn的准确率
        #      'loss_meta_cls':tensor[]预测重塑网络的元分类损失
        #      'meta_acc':tensor[1]预测重塑网络的分类精确率
        # }
        roi_losses = self.roi_head.forward_train(
            # 查询集特征图列表
            query_feats,
            # 支持集特征
            # 在本例中这经过了主干网络
            support_feats,
            # rpn的推荐预测框
            proposals=proposal_list,
            # 查询集的信息(元信息,真实边界框与真实标签)
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            # 查询集忽略的真实地物边界框
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),

            # 支持集的信息(元信息,真实边界框与真实标签)
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            # 支持集忽略的真实地物边界框
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        # dict{'loss_cls':tensor[]rcnn的分类损失
        #      'loss_bbox':tensor[]rcnn的分类损失
        #      'acc':tensor[]rcnn的准确率
        #      'loss_meta_cls':tensor[]预测重塑网络的元分类损失
        #      'meta_acc':tensor[1]预测重塑网络的分类精确率
        #       loss_rpn_cls=losses['loss_cls'], 
        #       loss_rpn_bbox=losses['loss_bbox']
        # }
        losses.update(roi_losses)

        return losses

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        """Test without augmentation."""
        raise NotImplementedError

    def aug_test(self, **kwargs):
        """Test with augmentation."""
        raise NotImplementedError

    # 抽象方法在子类中实现
    @abstractmethod
    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        """extract and save support features for model initialization."""
        raise NotImplementedError

    @abstractmethod
    def model_init(self, **kwargs):
        """process the saved support features for model initialization."""
        raise NotImplementedError
