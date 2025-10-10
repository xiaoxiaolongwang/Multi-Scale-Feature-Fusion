# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os.path as osp
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose
from terminaltables import AsciiTable

from mmfewshot.utils import get_root_logger
from .utils import NumpyEncoder

# 此类对象的__getitem_方法在父类CustomDataset中实现
@DATASETS.register_module()
class BaseFewShotDataset(CustomDataset):
    """Base dataset for few shot detection.

    The main differences with normal detection dataset fall in two aspects.

        - It allows to specify single (used in normal dataset) or multiple
            (used in query-support dataset) pipelines for data processing.
        - It supports to control the maximum number of instances of each class
            when loading the annotation file.

    The annotation format is shown as follows. The `ann` field
    is optional for testing.

    .. code-block:: none

        [
            {
                'id': '0000001'
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
    注释配置，支持从两种文件类型载入，type='ann_file'为common ann_file
        ann_cfg (list[dict]): Annotation config support two type of config.

            - loading annotation from common ann_file of dataset
              with or without specific classes.
              example:dict(type='ann_file', ann_file='path/to/ann_file',
              ann_classes=['dog', 'cat'])
            - loading annotation from a json file saved by dataset.
              example:dict(type='saved_dataset', ann_file='path/to/ann_file')
    为空
        classes (str | Sequence[str] | None): Classes for model training and
            provide fixed label for each class.
    
        pipeline (list[dict] | None): Config to specify processing pipeline.
            Used in normal dataset. Default: None.
    如何处理数据集
        multi_pipelines (dict[list[dict]]): Config to specify
            data pipelines for corresponding data flow.
            For example, query and support data
            can be processed with two different pipelines, the dict
            should contain two keys like:

                - query (list[dict]): Config for query-data
                  process pipeline.
                - support (list[dict]): Config for support-data
                  process pipeline.
        data_root (str | None): Data root for ``ann_cfg``, `img_prefix``,
            ``seg_prefix``, ``proposal_file`` if specified. Default: None.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
    没有任何数据集类别边界框的图像将会被过滤掉，只在训练以及验证阶段生效
        filter_empty_gt (bool): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests. Default: True.
        min_bbox_size (int | float | None): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_bbox_size``, it would be added to ignored field.
            Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, all annotation from ann file would be loaded.
            Default: None.
    为负
    如果设置为 True  self.data_infos 将变为"实例级"（instance-wise）的数据结构
    如果一张图像的标注中包含多个实例（例如多个目标物体），这个标注将被拆分成多个项目。

    正常情况下（False），每张图像对应 data_infos 中的一个条目。
    设置为 True 后，如果一张图像包含 N 个实例，它将在 data_infos 中对应 N 个条目。

    注释提到这种方式经常用于支持数据集（support datasets）。
    在少样本学习或元学习等任务中，这种实例级的数据组织方式可能很有用。

        instance_wise (bool): If set true, `self.data_infos`
            would change to instance-wise, which means if the annotation
            of single image has more than one instance, the annotation would be
            split to num_instances items. Often used in support datasets,
            Default: False.
    数据集名字
        dataset_name (str | None): Name of dataset to display. For example:
            'train_dataset' or 'query_dataset'. Default: None.
    """

    CLASSES = None

    def __init__(self,
                #  装有文件信息的地址，以及数据集的形式
                # 1
                 ann_cfg: List[Dict],
                 classes: Union[str, Sequence[str], None],
                 pipeline: Optional[List[Dict]] = None,
                #  1
                 multi_pipelines: Optional[Dict[str, List[Dict]]] = None,
                 data_root: Optional[str] = None,
                #  图像前缀
                # 1
                 img_prefix: str = '',
                 seg_prefix: Optional[str] = None,
                 proposal_file: Optional[str] = None,
                 test_mode: bool = False,
                # 过滤没有标注信息的图片
                 filter_empty_gt: bool = True,
                 min_bbox_size: Optional[Union[int, float]] = None,
                 ann_shot_filter: Optional[Dict] = None,
                #  1
                 instance_wise: bool = False,
                #  1
                 dataset_name: Optional[str] = None) -> None:
        self.data_root = data_root
        # 图像前缀
        self.img_prefix = img_prefix
        # 空
        self.seg_prefix = seg_prefix
        # 空
        self.proposal_file = proposal_file
        # 是不是测试模型
        self.test_mode = test_mode
        # 是否过滤空注释图像
        self.filter_empty_gt = filter_empty_gt
        # 不进入分支
        if classes is not None:
            self.CLASSES = self.get_classes(classes)

        # 是否将图像划分为实例级（否）
        self.instance_wise = instance_wise
        # set dataset name
        # 数据集名称
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name

        # 不看
        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
                
        # 注释配置
        self.ann_cfg = copy.deepcopy(ann_cfg)
        # 从注释配置获得数据信息（返回值是一个装有本数据集中所有图片信息以及注释信息的列表（图片信息以字典存储））
        # 模型初始化部分复制的数据的方法在FewShotVOCCopyDataset类(FewShotVOCDataset的子类)中重写
        # 训练集的支持集与查询集,测试数据集都是从文件中提取出来的图片信息
        # 模型初始化数据集使用的是支持集的self.data_infos_by_class构建的数据集
        self.data_infos = self.ann_cfg_parser(ann_cfg)

        assert self.data_infos is not None, \
            f'{self.dataset_name} : none annotation loaded.'
        # load proposal file
        # 加载推荐文件（如果有的话）
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        # 如果不是测试模型(测试模型直接跳过这部分)
        if not test_mode:
            # filter bbox smaller than the min_bbox_size
            if min_bbox_size:
                self.data_infos = self._filter_bboxs(min_bbox_size)
            # filter images
            # 这个方法是过滤方法，过滤的条件有(图片是否有注释，图片尺寸，图片标注框的面积)但是请注意返回列表装的是要留下的元素索引
            # （如果一张图片没有任何标注框的话就将其过滤掉）（是否使用难例会在这一步导致注释信息发生变化（这会改变剩余图片的数量））
            # 我们在构建模型初始化数据集的时候使用的是支持集已经过滤完的数据,而支持集不使用难例
            # 需要说明的是支持集其实是有三个数据源的,
            # 一个与查询集一样self.data_infos,
            # 另一个则是self.data_infos_by_class
            # 最后是self.batch_indices(我希望你可以在看完源码之后体会)
            valid_inds = self._filter_imgs()
            # 从标注信息列表中取出索引元素（这段代码通过过滤的结果修改数据信息）
            # 这里通过self.data_infos的变化可以看出来，数据集中有一部分图片是完全没有注释标记的
            self.data_infos = [self.data_infos[i] for i in valid_inds]

            # 如果推荐不为空的话
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # filter annotations by ann_shot_filter
            # 先不看
            # 微调训练时使用此步骤
            if ann_shot_filter is not None:
                if isinstance(ann_shot_filter, dict):
                    for class_name in list(ann_shot_filter.keys()):
                        assert class_name in self.CLASSES, \
                            f'{self.dataset_name} : class ' \
                            f'{class_name} in ann_shot_filter not in CLASSES.'
                else:
                    raise TypeError('ann_shot_filter only support dict')
                self.ann_shot_filter = ann_shot_filter
                self.data_infos = self._filter_annotations(
                    self.data_infos, self.ann_shot_filter)
            # instance_wise will make each data info only contain one
            # annotation otherwise all annotation from same image will
            # be checked and merged.
            # 先不看这个分支(在模型初始化数据集构建的时候经过此分支)
            # 此分支将图片信息分为 图片标签个数 个图片信息(这意味着每个信息的图片id可能会有重复)
            if self.instance_wise:
                # 实例级别的数据信息列表
                instance_wise_data_infos = []
                # 从我们获取到的数据信息列表取出信息
                for data_info in self.data_infos:
                    # 获取这个图像的标签数量
                    num_instance = data_info['ann']['labels'].size
                    # split annotations
                    if num_instance > 1:
                        # 遍历图片信息标签列表
                        for i in range(data_info['ann']['labels'].size):
                            # 复制一份这个图片信息
                            tmp_data_info = copy.deepcopy(data_info)
                            # np.expand_dims在数据的指定位置维度添加一个大小为1的维度
                            # data_info['ann']['labels'][i]是一个整形数,经过这个操作之后形状变成[1,1]
                            tmp_data_info['ann']['labels'] = np.expand_dims(
                                data_info['ann']['labels'][i], axis=0)
                            # data_info['ann']['bboxes'][i, :]形状为[4,],经过这个操作之后形状变成[1,4]
                            tmp_data_info['ann']['bboxes'] = np.expand_dims(
                                data_info['ann']['bboxes'][i, :], axis=0)
                            
                            # 将临时数据信息添加到实例级别的数据信息列表中(图片信息可能会有重复)
                            instance_wise_data_infos.append(tmp_data_info)
                    # 按理说这个分支就够用了(上边的分支也可以看一下)
                    else:
                        # 给实例级别的数据信息列表中添加单标签的图像信息
                        instance_wise_data_infos.append(data_info)
                self.data_infos = instance_wise_data_infos

            # merge different annotations with the same image
            # 先看这个分支
            # 此分支将图片信息中的重复图片的标签部分合并起来
            else:
                # 创建一个字典（装有合并后的注释信息）
                merge_data_dict = {}
                for i, data_info in enumerate(self.data_infos):
                    # merge data_info with the same image id
                    # 从合并字典中找图片名字，如果没有找到，将这个注释信息添加到合并字典中
                    # 形式为'图片名':data_info
                    if merge_data_dict.get(data_info['id'], None) is None:
                        merge_data_dict[data_info['id']] = data_info
                    # 如果有重复，执行合并分支
                    else:
                        # 获取合并注释信息与注释信息
                        ann_a = merge_data_dict[data_info['id']]['ann']
                        ann_b = data_info['ann']
                        # 合并边界框与标签部分
                        merge_dat_info = {
                            'bboxes':
                            np.concatenate((ann_a['bboxes'], ann_b['bboxes'])),
                            'labels':
                            np.concatenate((ann_a['labels'], ann_b['labels'])),
                        }
                        # merge `bboxes_ignore`
                        # 合并忽略边界框部分（如果设置为使用难例就不会进入这里）
                        if ann_a.get('bboxes_ignore', None) is not None:
                            # 判断合并字典中此图片的边界框忽略numpy是否与该图片的一致（如果不一致则进入）
                            if not (ann_a['bboxes_ignore']
                                    == ann_b['bboxes_ignore']).all():
                                merge_dat_info['bboxes_ignore'] = \
                                    np.concatenate((ann_a['bboxes_ignore'],
                                                    ann_b['bboxes_ignore']))
                                merge_dat_info['labels_ignore'] = \
                                    np.concatenate((ann_a['labels_ignore'],
                                                    ann_b['labels_ignore']))
                        # 重新组织合并字典中对应图片的注释信息
                        merge_data_dict[
                            data_info['id']]['ann'] = merge_dat_info
                # 将合并字典还原回原始的注释信息形式重新赋值给self.data_infos
                self.data_infos = [
                    merge_data_dict[key] for key in merge_data_dict.keys()
                ]
            # set group flag for the sampler
            # 应该是先在本文件下找这个方法
            # 给实例添加flag属性，属性为全0数组，长度为注释信息数组的len
            # 在训练模式下会调用这个方法为实例添加self.flag = np.zeros(len(self), dtype=np.uint8)
            self._set_group_flag()

        assert pipeline is None or multi_pipelines is None, \
            f'{self.dataset_name} : can not assign pipeline ' \
            f'or multi_pipelines simultaneously'
        # processing pipeline if there are two pipeline the
        # pipeline will be determined by key name of query or support
        # 如果多管道不为空
        if multi_pipelines is not None:
            assert isinstance(multi_pipelines, dict), \
                f'{self.dataset_name} : multi_pipelines is type of dict'
            
            # 给实例添加多管道的字典
            self.multi_pipelines = {}
            # 遍历传入参数的键'query'与'support'
            for key in multi_pipelines.keys():
                # 给实例添加Compose对象，这个对象感觉跟torch.transform行为是一致的
                self.multi_pipelines[key] = Compose(multi_pipelines[key])
        # (测试(验证)阶段其实走的是这个分支)
        elif pipeline is not None:
            assert isinstance(pipeline, list), \
                f'{self.dataset_name} : pipeline is type of list'
            # 给实例绑定self.pipeline字段(等一切结束之后看一下这里边的transforms是怎么写的)
            # 'MultiScaleFlipAug'
            # /mmdet/datasets/pipelines/test_time_aug.py
            self.pipeline = Compose(pipeline)
        else:
            raise ValueError('missing pipeline or multi_pipelines')

        # show dataset annotation usage
        # 只要使用get_root_logger()返回的日志器都是一样的，名为'mmfewshot'的日志器会被重复使用
        # 我大概知道是什么意思了，由于名为'mmfewshot'的日志器已经被注册过了
        logger = get_root_logger()
        # 向.log文件中写入日志
        logger.info(self.__repr__())


    # 该方法的主体其实是判断注释配置是否规范，主要看最后的返回方法，
    # 函数定义明确说明了返回类型为装有字典的列表
    def ann_cfg_parser(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Parse annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): Annotation config support two type of config.

                - 'ann_file': loading annotation from common ann_file of
                    dataset. example: dict(type='ann_file',
                    ann_file='path/to/ann_file', ann_classes=['dog', 'cat'])
                - 'saved_dataset': loading annotation from saved dataset.
                    example:dict(type='saved_dataset',
                    ann_file='path/to/ann_file')

        Returns:
            list[dict]: Annotation information.
        """
        # 跳过
        # join paths if data_root is specified
        if self.data_root is not None:
            for i in range(len(ann_cfg)):
                if not osp.isabs(ann_cfg[i]['ann_file']):
                    ann_cfg[i]['ann_file'] = \
                        osp.join(self.data_root, ann_cfg[i]['ann_file'])
        # ann_cfg must be list
        # ann_cfg一定要是一个列表类型
        assert isinstance(ann_cfg, list), \
            f'{self.dataset_name} : ann_cfg should be type of list.'
        # check type of ann_cfg
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name} : ann_cfg should be list of dict.'
            assert ann_cfg_['type'] in ['ann_file', 'saved_dataset'], \
                f'{self.dataset_name} : ann_cfg only support type of ' \
                f'ann_file and saved_dataset'
        # 此方法在子类重写
        return self.load_annotations(ann_cfg)

    # 返回数据注释信息self.data_infos的索引为idx的元素 键ann的值
    def get_ann_info(self, idx: int) -> Dict:
        """Get annotation by index.

        When override this function please make sure same annotations are used
        during the whole training.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return copy.deepcopy(self.data_infos[idx]['ann'])
    # 准备一张训练图片（如果是支持集的话，gt_idx应该不为空）
    def prepare_train_img(self,
                          idx: int,
                          pipeline_key: Optional[str] = None,
                          gt_idx: Optional[List[int]] = None) -> Dict:
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.
            pipeline_key (str): Name of pipeline
            gt_idx (list[int]): Index of used annotation.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        # 获得索引图片整体信息
        img_info = self.data_infos[idx]
        # 获得索引图片的注释信息
        ann_info = self.get_ann_info(idx)

        # select annotation in `gt_idx`
        # 感觉这个分支是为支持集准备的
        if gt_idx is not None:
            # 获得支持集图像被选中的物体标注信息（gt_idx是一个索引列表）

            selected_ann_info = {
                # 如果ann_info['bboxes']是一个（6，4）的numpy，gt_idx为[0,2,3]
                # ann_info['bboxes'][gt_idx]的行为是
                # 从原始的 (6, 4) 数组中选择索引为 0、2 和 3 的行
                # 返回一个新的 (3, 4) 形状的 NumPy 数组
                'bboxes': ann_info['bboxes'][gt_idx],
                'labels': ann_info['labels'][gt_idx]
            }
            # keep pace with new annotations
            # 复制一份图片信息
            new_img_info = copy.deepcopy(img_info)
            # 将新的图片信息的key'注释'的值改为被选中的该图片中的物体标注信息
            new_img_info['ann'] = selected_ann_info
            # 将图片信息与注释信息汇总
            results = dict(img_info=new_img_info, ann_info=selected_ann_info)

        # 看这个分支
        # use all annotations
        else:
            # 将图片信息与注释信息汇总
            results = dict(img_info=copy.deepcopy(img_info), ann_info=ann_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]

        # 此方法内部做了什么（给字典添加了六对键值）
        # results['img_prefix'] = self.img_prefix
        # results['seg_prefix'] = self.seg_prefix
        # results['proposal_file'] = self.proposal_file
        # results['bbox_fields'] = []
        # results['mask_fields'] = []
        # results['seg_fields'] = []
        self.pre_pipeline(results)

        # 如果多通道的键为空执行这个分支
        if pipeline_key is None:
            return self.pipeline(results)
        # 看这个分支
        else:
            # 这其中其实会很复杂，但是这里我先将返回值中的一些值标出来，以便以后看代码方便
#             字典data中的信息有		'filename', 'ori_filename', 'ori_shape',
#           （这两个shape应该是一样的吧）'img_shape', 'pad_shape', 'scale_factor',
# 		  		                       'flip','flip_direction', 'img_norm_cfg'，

# 		  		                       'img', 'gt_bboxes', 'gt_labels'
            return self.multi_pipelines[pipeline_key](results)

    def _filter_annotations(self, data_infos: List[Dict],
                            ann_shot_filter: Dict) -> List[Dict]:
        """Filter out extra annotations of specific class, while annotations of
        classes not in filter remain unchanged and the ignored annotations will
        be removed.

        Args:
            data_infos (list[dict]): Annotation infos.
            ann_shot_filter (dict): Specific which class and how many
                instances of each class to load from annotation file.
                For example: {'dog': 10, 'cat': 10, 'person': 5}

        Returns:
            list[dict]: Annotation infos where number of specified class
                shots less than or equal to predefined number.
        """
        if ann_shot_filter is None:
            return data_infos
        # build instance indices of (img_id, gt_idx)
        filter_instances = {key: [] for key in ann_shot_filter.keys()}
        # 保持实例的索引
        keep_instances_indices = []

        for idx, data_info in enumerate(data_infos):
            ann = data_info['ann']
            # 遍历此图片的所有标签
            for i in range(ann['labels'].shape[0]):
                # 获得此图片的第i个物体的的类别名
                instance_class_name = self.CLASSES[ann['labels'][i]]
                # only filter instance from the filter class
                if instance_class_name in ann_shot_filter.keys():
                    # 向过滤实例字典的 类别instance_class_name中添加 这个实例的所属图片索引与图片中的第i个物体索引(以元组的形式(idx, i))
                    # 经过这样的操作之后filter_instances字典中的实例个数有可能会超过预定个数,后续进行过滤操作
                    filter_instances[instance_class_name].append((idx, i))
                # skip the class not in the filter
                else:
                    # 如果这个实例的类别不在我们的过滤器要考虑的类别列表中，则将其保存下来
                    keep_instances_indices.append((idx, i))
        # filter extra shots
        # 遍历过滤器的所有键
        for class_name in ann_shot_filter.keys():
            # 获得这个类别的shot数量
            num_shots = ann_shot_filter[class_name]
            # 获得这个类别的所有实例的索引
            instance_indices = filter_instances[class_name]

            # 如果这个类别的预先设定shot数为0则直接跳过这个类
            if num_shots == 0:
                continue
            # random sample from all instances
            # 如果实例的个数大于预先指定的shot数量的话,随机抽取预先指定的shot个数的实例的注释
            if len(instance_indices) > num_shots:
                random_select = np.random.choice(
                    len(instance_indices), num_shots, replace=False)
                # 将我们随机抽取的注释信息放到需要保留的实例索引中
                keep_instances_indices += \
                    [instance_indices[i] for i in random_select]
            # number of available shots less than the predefined number,
            # which may cause the performance degradation
            else:
                # check the number of instance
                if len(instance_indices) < num_shots:
                    # 如果提供的实例个数没有达到我们所需要的shot数量则会报出警告
                    warnings.warn(f'number of {class_name} instance is '
                                  f'{len(instance_indices)} which is '
                                  f'less than predefined shots {num_shots}.')
                keep_instances_indices += instance_indices

        # keep the selected annotations and remove the undesired annotations
        new_data_infos = [] 
        for idx, data_info in enumerate(data_infos):
            # 从需要保留的注释实例类表中选出第idx张图片的所有注释实例索引,并将它们升序排列返回
            selected_instance_indices = \
                sorted(instance[1] for instance in keep_instances_indices
                       if instance[0] == idx)
            if len(selected_instance_indices) == 0:
                continue
            # 得到这张图片的注释信息
            ann = data_info['ann']
            # 从注释信息的边界框与标签中选出我们要保留的注释边框信息
            selected_ann = dict(
                bboxes=ann['bboxes'][selected_instance_indices],
                labels=ann['labels'][selected_instance_indices],
            )
            # 向新的数据信息列表中添加这张图片的信息(注释信息经过shot处理)
            new_data_infos.append(
                dict(
                    id=data_info['id'],
                    filename=data_info['filename'],
                    width=data_info['width'],
                    height=data_info['height'],
                    ann=selected_ann))
        return new_data_infos

    def _filter_bboxs(self, min_bbox_size: int) -> List[Dict]:
        new_data_infos = []
        for data_info in self.data_infos:
            ann = data_info['ann']
            keep_idx, ignore_idx = [], []
            for i in range(ann['bboxes'].shape[0]):
                bbox = ann['bboxes'][i]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                # check bbox size
                if w < min_bbox_size or h < min_bbox_size:
                    ignore_idx.append(i)
                else:
                    keep_idx.append(i)
            # remove undesired bbox
            if len(ignore_idx) > 0:
                bboxes_ignore = ann.get('bboxes_ignore', np.zeros((0, 4)))
                labels_ignore = ann.get('labels_ignore', np.zeros((0, )))
                new_bboxes_ignore = ann['bboxes'][ignore_idx]
                new_labels_ignore = ann['labels'][ignore_idx]
                bboxes_ignore = np.concatenate(
                    (bboxes_ignore, new_bboxes_ignore))
                labels_ignore = np.concatenate(
                    (labels_ignore, new_labels_ignore))
                data_info.update(
                    ann=dict(
                        bboxes=ann['bboxes'][keep_idx],
                        labels=ann['labels'][keep_idx],
                        bboxes_ignore=bboxes_ignore,
                        labels_ignore=labels_ignore))
            new_data_infos.append(data_info)
        return new_data_infos

    def _set_group_flag(self) -> None:
        
        """
        根据图像长宽比设置标志。

        长宽比大于1的图像设为第一组；
        否则组为0。在少数镜头设置中，图像数量有限
        可能会导致一些小型批处理总是采样一定数量的图像
        因此不能完全打乱数据，这可能会降低性能。
        因此，所有标志都被简单地设置为0。
        
        Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In few shot setting, the limited number of images
        might cause some mini-batch always sample a certain number of images
        and thus not fully shuffle the data, which may degrade the performance.
        Therefore, all flags are simply set to 0.
        """
        # 调用父类的方法，返回注释信息的数目
        # len(self.data_infos)
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations_saved(self, ann_file: str) -> List[Dict]:
        """Load data_infos from saved json."""
        with open(ann_file) as f:
            data_infos = json.load(f)
        # record the index of meta info
        meta_idx = None
        for i, data_info in enumerate(data_infos):
            # check the meta info CLASSES and img_prefix saved in json
            if 'CLASSES' in data_info.keys():
                assert self.CLASSES == tuple(data_info['CLASSES']), \
                    f'{self.dataset_name} : class labels mismatch.'
                assert self.img_prefix == data_info['img_prefix'], \
                    f'{self.dataset_name} : image prefix mismatch.'
                meta_idx = i
                # skip the meta info
                continue
            # convert annotations from list into numpy array
            for k in data_info['ann']:
                assert isinstance(data_info['ann'][k], list)
                # load bboxes and bboxes_ignore
                if 'bboxes' in k:
                    # bboxes_ignore can be empty
                    if len(data_info['ann'][k]) == 0:
                        data_info['ann'][k] = np.zeros((0, 4))
                    else:
                        data_info['ann'][k] = \
                            np.array(data_info['ann'][k], dtype=np.float32)
                # load labels and labels_ignore
                elif 'labels' in k:
                    # labels_ignore can be empty
                    if len(data_info['ann'][k]) == 0:
                        data_info['ann'][k] = np.zeros((0, ))
                    else:
                        data_info['ann'][k] = \
                            np.array(data_info['ann'][k], dtype=np.int64)
                else:
                    raise KeyError(f'unsupported key {k} in ann field')
        # remove meta info
        if meta_idx is not None:
            data_infos.pop(meta_idx)
        return data_infos

    def save_data_infos(self, output_path: str) -> None:
        """Save data_infos into json."""
        # numpy array will be saved as list in the json
        meta_info = [{'CLASSES': self.CLASSES, 'img_prefix': self.img_prefix}]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                meta_info + self.data_infos,
                f,
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder)

    def __repr__(self) -> str:
        """Print the number of instances of each class."""
        # 实例的类名,数据集的名字,图片数量,实例信息
        result = (f'\n{self.__class__.__name__} {self.dataset_name} '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        # 实例计数数组，类别数加一是背景实例
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        # 遍历所有注释图片信息
        for idx in range(len(self)):
            # get_ann_info方法返回数据注释信息self.data_infos的索引为idx的元素 键ann的值
            # 取出字典中键'labels'对应的值
            label = self.get_ann_info(idx)['labels']
            # 返回数组中的唯一值（去重），return_counts=True表明返回其中每个元素出现的次数
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        # 创建标题
        table_data = [['category', 'count'] * 5]
        # 创建行数据
        row_data = []
        # 遍历实际计数数组
        for cls, count in enumerate(instance_count):
            # 如果不是背景索引
            if cls < len(self.CLASSES):
                # 向行数据数组中添加两个元素（'类标签 类名','类别数量'）
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            # 如果行数据元素个数为10，换行
            if len(row_data) == 10:
                # 向表格数据中添加一个元素
                table_data.append(row_data)
                # 换行
                row_data = []
        # 添加剩余元素（如果有的话）
        if len(row_data) != 0:
            table_data.append(row_data)

        table = AsciiTable(table_data)
        # 向结果中添加表格
        result += table.table
        return result
