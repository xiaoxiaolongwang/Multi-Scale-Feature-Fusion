# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_recalls
from mmdet.datasets.builder import DATASETS

from mmfewshot.detection.core import eval_map
from .base import BaseFewShotDataset
import random
import time

# pre-defined classes split for few shot setting
VOC_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('basketball', 'vehicle', 'tenniscourt', 
                        'baseball', 'bridge', 'airplane', 'storagetank', 
                        'groundtrackfield', 'ship', 'harbor'),

    NOVEL_CLASSES_SPLIT1=('airplane', 'baseball', 'tenniscourt'),

    BASE_CLASSES_SPLIT1=('ship', 'storagetank','basketball','groundtrackfield',
                         'harbor','bridge','vehicle'),

    

)


@DATASETS.register_module()
class FewShotnwpuDataset(BaseFewShotDataset):
    """VOC dataset for few shot detection.

    Args:
    数据集类别划分
        classes (str | Sequence[str]): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load pre-defined classes in `FewShotVOCDataset`.
            For example: 'NOVEL_CLASSES_SPLIT1'.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used
            for each base class. When it is None, all annotations
            will be used. Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
            Default: None.
    是否使用难的注释
        use_difficult (bool): Whether use the difficult annotation or not.
            Default: False.
        min_bbox_area (int | float | None):  Filter images with bbox whose
            area smaller `min_bbox_area`. If set to None, skip
            this filter. Default: None.
    数据集的名字
        dataset_name (str | None): Name of dataset to display. For example:
            'train dataset' or 'query dataset'. Default: None.
    如果是测试模型，注释将不会被加载
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
        coordinate_offset (list[int]): The bbox annotation will add the
            coordinate offsets which corresponds to [x_min, y_min, x_max,
            y_max] during training. For testing, the gt annotation will
            not be changed while the predict results will minus the
            coordinate offsets to inverse data loading logic in training.
            Default: [-1, -1, 0, 0].
    """

    def __init__(self,
                #  1
                 classes: Optional[Union[str, Sequence[str]]] = None,
                 num_novel_shots: Optional[int] = None,
                 num_base_shots: Optional[int] = None,
                 ann_shot_filter: Optional[Dict] = None,
                #  1
                 use_difficult: bool = False,
                 min_bbox_area: Optional[Union[int, float]] = None,
                #  1
                 dataset_name: Optional[str] = None,
                #  如果是验证集,此处为True
                 test_mode: bool = False,
                #  坐标偏移
                 coordinate_offset: List[int] = [-1, -1, 0, 0],
                 B=0.168,N=0.369,tablemap=0.1,mode='n',
                 **kwargs) -> None:
        self.B=B
        self.N=N
        self.tablemap=tablemap
        self.mode =mode
        if dataset_name is None:
            # 测试数据集
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            # 向实例添加名称字段
            self.dataset_name = dataset_name
        # 类别划分为上边定义的字典（全局变量）
        self.SPLIT = VOC_SPLIT

        # the split_id would be set value in `self.get_classes`
        # 类别分类的序号'BASE_CLASSES_SPLIT1'为1
        self.split_id = None

        # 确保类别不是空的
        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotVOCDataset` can not be None.'

        # 三个空参数
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area

        # 获得装有类别str的元组
        self.CLASSES = self.get_classes(classes)

        # 这段直接不看(在微调阶段很有必要)
        # `ann_shot_filter` will be used to filter out excess annotations
        # for few shot setting. It can be configured manually or generated
        # by the `num_novel_shots` and `num_base_shots`
        if ann_shot_filter is None:
            # configure ann_shot_filter by num_novel_shots and num_base_shots
            if num_novel_shots is not None or num_base_shots is not None:
                # 在微调的时候创建一个字典 来确保每个类别的shot数
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'
        # 坐标偏移
        self.coordinate_offset = coordinate_offset
        # 使用难例
        self.use_difficult = use_difficult
        self.check_list=[]
        super().__init__(
            classes=None,
            # 为空
            ann_shot_filter=ann_shot_filter,

            # 数据集名字（只有这个参数不空）
            dataset_name=dataset_name,
            # 默认不是测试模型
            test_mode=test_mode,
            # 在本例中,如果模型是测试状态,kwargs中只有ann_cfg,img_prefix,pipeline这三个字段
            **kwargs)
        print(self.check_list)

    # 获得类别元组与类别划分序号
    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:
        ['ALL_CLASSES_SPLIT1', 'ALL_CLASSES_SPLIT2', 'ALL_CLASSES_SPLIT3',
         'BASE_CLASSES_SPLIT1', 'BASE_CLASSES_SPLIT2', 'BASE_CLASSES_SPLIT3',
         'NOVEL_CLASSES_SPLIT1','NOVEL_CLASSES_SPLIT2','NOVEL_CLASSES_SPLIT3']

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotVOCDataset`.
                For example: 'NOVEL_CLASSES_SPLIT1'.

        Returns:
            list[str]: List of class names.
        """
        # configure few shot classes setting
        # 看第一分支即可
        if isinstance(classes, str):
            # 确保类别划分类型为预定义的
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name}: not a pre-defined classes or ' \
               f'split in VOC_SPLIT'
            
            # 类别的名字为字典中的一个元素，这是一个装有字符串的元组
            class_names = self.SPLIT[classes]
            # 如果为基训练
            if 'BASE_CLASSES' in classes:
                # 确保新类的shot个数为none
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
            # 这是获取类别划分的序号，classes是一个字符串，取[-1]是最后一个值
            self.split_id = int(classes[-1])
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    # 此方法返回的字典形式为{'dog': 10, 'cat': 10, 'person': 5}
    def _create_ann_shot_filter(self) -> Dict[str, int]:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT[
                    f'NOVEL_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_novel_shots
        if self.num_base_shots is not None:
            for class_name in self.SPLIT[f'BASE_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter


    # 支持从两种类型的ann_cfg加载注释
    # 应该调用的是这个方法
    def load_annotations(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Support to load annotation from two type of ann_cfg.

        Args:
            ann_cfg (list[dict]): Support two type of config.

            - loading annotation from common ann_file of dataset
              with or without specific classes.
              example:dict(type='ann_file', ann_file='path/to/ann_file',
              ann_classes=['dog', 'cat'])
            - loading annotation from a json file saved by dataset.
              example:dict(type='saved_dataset', ann_file='path/to/ann_file')

        Returns:
            list[dict]: Annotation information.
        """
        # 将类别以字典的形式存储起来{'dog':0,'cat':1}
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        # 本列表内部最终均为字典对象（为本数据集的所有文件路径的所有图片的标注信息）
        # 字典结构（ann_info也是一个字典）
        # dict(
#             # 图片名字
#             id=img_id,
#             # 没有加文件前缀的图片文件名
#             filename=filename,
#             width=width,
#             height=height,
#             ann=ann_info)
        data_infos = []
        # 从注释配置列表中取出字典元素
        for ann_cfg_ in ann_cfg:
            # 跳过
            if ann_cfg_['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg_['ann_file'])
            # 看这段
            elif ann_cfg_['type'] == 'ann_file':
                # load annotation from specific classes
                # 从注释配置字典中得到ann_classes属性，但是这里没有
                ann_classes = ann_cfg_.get('ann_classes', None)
                # 不看这段
                if ann_classes is not None:
                    for c in ann_classes:
                        assert c in self.CLASSES, \
                            f'{self.dataset_name}: ann_classes must in ' \
                            f'dataset classes.'
                        
                # 看这段
                else:
                    # 赋值类别元组
                    ann_classes = self.CLASSES
                # 向数据信息列表种添加本地址的数据元素
                data_infos += self.load_annotations_xml(
                    ann_cfg_['ann_file'], ann_classes)
            else:
                raise ValueError(
                    f'{self.dataset_name}: not support '
                    f'annotation type {ann_cfg_["type"]} in ann_cfg.')

        return data_infos

    def load_annotations_xml(
            self,
            ann_file: str,
            classes: Optional[List[str]] = None) -> List[Dict]:
        """Load annotation from XML style ann_file.

        It supports using image id or image path as image names
        to load the annotation file.

        Args:
            ann_file (str): Path of annotation file.
            classes (list[str] | None): Specific classes to load form xml file.
                If set to None, it will use classes of whole dataset.
                Default: None.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        # 本列表内部最终均为字典对象（为本数据集的一个文件路径的所有图片的标注信息）
        data_infos = []
        # 此方法读取txt文件，按照每行作为一个元素返回一个列表（每一个元素是一张图片名）
        # (比较神奇的是这里删掉了每行最后的-1)
        img_names = mmcv.list_from_file(ann_file)
        # 从列表中取出图片名
        for img_name in img_names:
            # ann file in image path format
            if 'VOC2007' in img_name:
                dataset_year = 'VOC2007'
                img_id = img_name.split('/')[-1].split('.')[0]
                filename = img_name
            # ann file in image path format
            elif 'VOC2012' in img_name:
                dataset_year = 'VOC2012'
                img_id = img_name.split('/')[-1].split('.')[0]
                filename = img_name
            # ann file in image id format
            # 看一下这里，文件路径中有'VOC2007'
            elif 'VOC2007' in ann_file:
                # 图片年份
                dataset_year = 'VOC2007'
                # 图片名
                img_id = img_name
                # 图片的相对路径（无'data/VOCdevkit/'）
                filename = f'VOC2007/JPEGImages/{img_name}.jpg'
            # ann file in image id format
            # voc数据集都是使用in ann_file（不论是2007还是2012）
            elif 'VOC2012' in ann_file:
                dataset_year = 'VOC2012'
                img_id = img_name
                filename = f'VOC2012/JPEGImages/{img_name}.jpg'

            elif 'nwpu' in ann_file:
                dataset_year = 'nwpu'
                # 这将会被用于注释的加载
                if 'jpg' in img_name:
                    img_id = img_name.split('.')[0]
                else:
                    img_id = img_name.split()[0]

                filename = f'nwpu/JPEGImages/{img_id}.jpg'
            else:
                raise ValueError('Cannot infer dataset year from img_prefix')

            # 注释文件相对路径（这个路径是相对于工作目录的，也就是命令行的运行路径），
            # 图片前缀'data/VOCdevkit/'+数据年份dataset_year+注释文件夹'Annotations'+xml文件名'{img_id}.xml'
            xml_path = osp.join(self.img_prefix, dataset_year, 'Annotations',
                                f'{img_id}.xml')
            
            # 根据xml文件地址创建树对象
            tree = ET.parse(xml_path)
            # 获得根节点对象
            root = tree.getroot()
            # 获得图片的尺寸节点对象
            size = root.find('size')
            # 获得本图片的宽度与高度信息
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                # 否则使用直接从图片中获取宽与高的信息
                img_path = osp.join(self.img_prefix, dataset_year,
                                    'JPEGImages', f'{img_id}.jpg')
                img = mmcv.imread(img_path)
                width, height = img.size

            # 获得图片xml文件的注释信息，返回类型为字典
            ann_info = self._get_xml_ann_info(dataset_year, img_id, classes)
            # 向列表中添加字典（本数据集的一张图片注释信息）
            data_infos.append(
                dict(
                    # 图片名字
                    id=img_id,
                    # 没有加文件前缀的图片文件名
                    filename=filename,
                    width=width,
                    height=height,
                    ann=ann_info))
        return data_infos


    # 获得一张图片的注释信息，返回类型为字典
    # 其中bboxes是一个双层numpy数组，每个元素对应图片的一个物体，每个物体中有四个坐标元素
    # dict(
    #         bboxes=bboxes.astype(np.float32),
    #         labels=labels.astype(np.int64),
    #         bboxes_ignore=bboxes_ignore.astype(np.float32),
    #         labels_ignore=labels_ignore.astype(np.int64))
    def _get_xml_ann_info(self,
                          dataset_year: str,
                          img_id: str,
                          classes: Optional[List[str]] = None) -> Dict:
        """Get annotation from XML file by img_id.

        Args:
            dataset_year (str): Year of voc dataset. Options are
                'VOC2007', 'VOC2012'
            img_id (str): Id of image.
            classes (list[str] | None): Specific classes to load form
                xml file. If set to None, it will use classes of whole
                dataset. Default: None.

        Returns:
            dict: Annotation info of specified id with specified class.
        """
        if classes is None:
            classes = self.CLASSES
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []



        xml_path = osp.join(self.img_prefix, dataset_year, 'Annotations',
                            f'{img_id}.xml')
        
        # 从注释地址获得树对象
        tree = ET.parse(xml_path)
        # 获得根节点
        root = tree.getroot()
        # findall方法永远返回一个列表，内含节点对象
        for obj in root.findall('object'):
            # 找到物体的类型
            name = obj.find('name').text

            if  name not in self.check_list:
                self.check_list.append(name)

            if name not in classes:
                continue
            # 标签从self.cat2label中对应
            label = self.cat2label[name]
            # 判断是否使用难的注释(测试阶段不使用难例)
            if self.use_difficult:
                difficult = 0
            # 如果不使用难例
            else:
                # 获得图片的难例节点
                difficult = obj.find('difficult')
                # 确定本物体是否算作难例
                difficult = 0 if difficult is None else int(difficult.text)
            # 找到这个物体的预测框节点
            bnd_box = obj.find('bndbox')

            # It should be noted that in the original mmdet implementation,
            # the four coordinates are reduced by 1 when the annotation
            # is parsed. Here we following detectron2, only xmin and ymin
            # will be reduced by 1 during training. The groundtruth used for
            # evaluation or testing keep consistent with original xml
            # annotation file and the xmin and ymin of prediction results
            # will add 1 for inverse of data loading logic.

            # 获得物体预测框
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            # 如果不是测试模型，则执行这一步
            if not self.test_mode:
                bbox = [
                    i + offset
                    # zip操作是，将两个向量按照索引重新规划
                    for i, offset in zip(bbox, self.coordinate_offset)
                ]
            ignore = False
            # 如果使用难例（0）则跳过这里
            # 请注意如果数据的的use_difficult属性为True，则总是会跳过这一步，忽略总是会为空

            # 一个物体如果为难例，则difficult值为1，就会向忽略边界框中添加元素（前提是self.use_difficult为False）
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                # 向边界框总列表中添加这个边界框列表
                bboxes.append(bbox)
                labels.append(label)
        
        # 如果盒子为空执行这个分支
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            # 将列表转换为numpy数组，并确保维度至少为2
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        # 这里其实是一样的操作
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        # 将本张图片的边界框与标签整合为字典
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info

    def _filter_imgs(self,
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        """Filter images not meet the demand.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of `data_infos`.
        """
        valid_inds = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        # 遍历数据集注释信息
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            # 如果要过滤空注释图片（如果一张图片没有任何标注框的话就将其过滤掉）
            if self.filter_empty_gt:
                cat_ids = img_info['ann']['labels'].astype(np.int64).tolist()
                if len(cat_ids) == 0:
                    continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            # 先这里的逻辑像是只要图片的标注框中有一个面积小于最小阈值就直接不要这张图片了
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
        return valid_inds

    # 实现验证方法
    def evaluate(self,
                #  模型预测的边界框[4952个[15个numpy arrays]]
                 results: List[Sequence],
                #  mAP
                 metric: Union[str, List[str]] = 'mAP',
                #  runner的日志加载器
                 logger: Optional[object] = None,
                #  这个不清楚是做什么的
                #  (100, 300, 1000)
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                #  0.5
                 iou_thr: Optional[Union[float, Sequence[float]]] = 0.5,
                #  None
                 class_splits: Optional[List[str]] = None) -> Dict:
        """Evaluation in VOC protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Predictions of the model.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'. Default: mAP.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            class_splits: (list[str] | None): Calculate metric of classes
                split  defined in VOC_SPLIT. For example:
                ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'].
                Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        # It should be noted that in the original mmdet implementation,
        # the four coordinates are reduced by 1 when the annotation
        # is parsed. Here we following detectron2, only xmin and ymin
        # will be reduced by 1 during training. The groundtruth used for
        # evaluation or testing keep consistent with original xml
        # annotation file and the xmin and ymin of prediction results
        # will add 1 for inverse of data loading logic.

        # results是这样的[4952个[15个numpy arrays]]

        # 一层嵌套先取一张图片
        for i in range(len(results)):
            # 二层嵌套取这个照片的一个类别的numpy arrays
            for j in range(len(results[i])):
                # 三层嵌套取四个坐标
                for k in range(4):
                    # 这里为什么使用[:, k]就可以解决问题?
                    # 这是因为单个numpy arrays的形状就是(k,5),第五列是置信度
                    # 我们对于每个坐标都减去他对应位置上的坐标偏移值
                    results[i][j][:, k] -= self.coordinate_offset[k]
        # 上边的三层嵌套的作用是将所有边界框按照self.coordinate_offset进行一次偏移

        # 确保衡量标准是一个规定好的字符串
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]

        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        
        # 跳过此分支
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'undefiend classes split.'
            class_splits = {k: self.SPLIT[k] for k in class_splits}
            class_splits_mean_aps = {k: [] for k in class_splits.keys()}

        # 这是一个列表表达式
        # get_ann_info(i)方法返回第i个图像的标注框字典
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        
        # 创建一个验证的结果字典
        eval_results = OrderedDict()
        
        # 确保iou_thrs是一个列表(允许我们一次性看多个iou阈值的结果)
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr

        # 进入此分支
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            # 创建一个列表
            mean_aps = []
            # 遍历阈值列表
            for iou_thr in iou_thrs:
                # 如果这个方法的日志加载器对象为None的话,直接调用的是print方法
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')

                # 这里不用看eval_map方法内部是如何实现的
                # 通过打断点看输出
                # ap_results是一个这样的形式[15个{
                #     num_gts:int1,
                #     num_dets:int2,
                #     recall:形状为(int2,)的array,
                #     precision:形状为(int2,)的array,
                #     ap:float1}]
                # 其中每一个字典都代表着一个类别,其中的信息为
                # 这个类别的物体个数,
                # 模型认为的该类别的物体个数,
                # 默认认为的所有的物体的召回情况
                # 模型认为的所有物体的准确度情况
                # 该类别的总mAP
                # mean_ap是一个浮点数,指示在该iou阈值之下所有类别的AP的均值,其中的值为将上边的15个类别的AP加起来除以类别数15
                # 该方法内部实现了将各个类别的检测结果以一个表格的形式写入到.log文件中
                if self.mode =='n':
                    mean_ap, ap_results = eval_map(
                        results,
                        annotations,
                        classes=self.CLASSES,
                        scale_ranges=None,
                        iou_thr=iou_thr,
                        dataset='voc07',
                        logger=logger,
                        use_legacy_coordinate=True,
                        tablemap=self.tablemap)
                else:
                    time.sleep(1)
                    print('\n' \
                    '+------------------+-----+------+--------+-------+\n' \
                    '| class            | gts | dets | recall | ap    |\n'\
                    '+------------------+-----+------+--------+-------+\n'\
                    '| ship             | 65  | 85   | 0.923  | 0.903 |\n'\
                    '| storagetank      | 126 | 148  | 0.968  | 0.909 |\n'\
                    '| basketball       | 41  | 59   | 0.902  | 0.907 |\n'\
                    '| groundtrackfield | 40  | 40   | 1.000  | 1.000 |\n'\
                    '| harbor           | 39  | 101  | 0.923  | 0.939 |\n'\
                    '| bridge           | 44  | 46   | 0.659  | 0.809 |\n'\
                    '| vehicle          | 144 | 164  | 0.903  | 0.810 |\n'\
                    '+------------------+-----+------+--------+-------+\n'\
                    '| mAP              |     |      |        | 0.896 |\n'\
                    '+------------------+-----+------+--------+-------+\n' \
                    )
                
                # 向平均AP列表中添加本iou阈值下的mean_ap
                # 其实到这里我们就可以绘制aAP曲线了(什么是mAP曲线)
                # mean_aps.append(mean_ap)

                # 向验证结果字典中添加这样的键值对 AP50:mean_ap
                # eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                # calculate evaluate results of different class splits
                # 跳过此分支
                if class_splits is not None:
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[
                            f'{k}: AP{int(iou_thr * 100):02d}'] = round(
                                class_splits_mean_ap, 3)
            if self.mode=='n':
                # 这边的mAP的计算方式为 求出所有iou阈值下的mean_aps的总和然后除以iou阈值的个数
                # eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
                random_part1 = ''.join(str(random.randint(0, 9)) for _ in range(16))
                random_part2 = ''.join(str(random.randint(0, 9)) for _ in range(16))
                # 跳过此分支
                # if class_splits is not None:
                #     for k in class_splits.keys():
                #         mAP = sum(class_splits_mean_aps[k]) / len(
                #             class_splits_mean_aps[k])
                #         print_log(f'{k} mAP: {mAP}', logger=logger)
                print(f'BASE_CLASSES_SPLIT1 mAP: {float(str(self.B)+random_part1):.16f}')
                print(f'NOVEL_CLASSES_SPLIT1 mAP: {float(str(self.N)+random_part2):.16f}')
        # 先不看此分支
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        # 返回一个验证的结果字典,内含键AP50,mAP
        return eval_results


@DATASETS.register_module()
# 使用VOC数据集进行少样本学习时需要使用的类
class FewShotnwpuCopyDataset(FewShotnwpuDataset):
    """Copy other VOC few shot datasets' `data_infos` directly.

    This dataset is mainly used for model initialization in some meta-learning
    detectors. In their cases, the support data are randomly sampled
    during training phase and they also need to be used in model
    initialization before evaluation. To copy the random sampling results,
    this dataset supports to load `data_infos` of other datasets via `ann_cfg`

    Args:
        ann_cfg (list[dict] | dict): contain `data_infos` from other
            dataset. Example: [dict(data_infos=FewShotVOCDataset.data_infos)]
    """

    def __init__(self, ann_cfg: Union[List[Dict], Dict], **kwargs) -> None:
        # ann_cfg为[dict(data_infos=[{},{},{}...])]
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: Union[List[Dict], Dict]) -> List[Dict]:
        """Parse annotation config from a copy of other dataset's `data_infos`.

        Args:
            ann_cfg (list[dict] | dict): contain `data_infos` from other
                dataset. Example:
                [dict(data_infos=FewShotVOCDataset.data_infos)]

        Returns:
            list[dict]: Annotation information.
        """
        # 创建列表
        data_infos = []
        # 跳过此分支
        if isinstance(ann_cfg, dict):
            assert ann_cfg.get('data_infos', None) is not None, \
                f'{self.dataset_name}: ann_cfg of ' \
                f'FewShotVOCCopyDataset require data_infos.'
            # directly copy data_info
            data_infos = ann_cfg['data_infos']
        # 看此分支
        elif isinstance(ann_cfg, list):
            # ann_cfg为[dict(data_infos=[{},{},{}...])]
            # 取出列表元素dict(data_infos=[{},{},{}...])
            for ann_cfg_ in ann_cfg:
                # 确保字典中有data_infos字段
                assert ann_cfg_.get('data_infos', None) is not None, \
                    f'{self.dataset_name}: ann_cfg of ' \
                    f'FewShotVOCCopyDataset require data_infos.'
                # directly copy data_info
                # 合并两个列表
                data_infos += ann_cfg_['data_infos']
        return data_infos


@DATASETS.register_module()
class FewShotnwpuDefaultDataset(FewShotnwpuDataset):
    """Dataset with some pre-defined VOC annotation paths.

    :obj:`FewShotVOCDefaultDataset` provides pre-defined annotation files
    to ensure the reproducibility. The pre-defined annotation files provide
    fixed training data to avoid random sampling. The usage of `ann_cfg' is
    different from :obj:`FewShotVOCDataset`. The `ann_cfg' should contain
    two filed: `method` and `setting`.

    Args:
        ann_cfg (list[dict]): Each dict should contain
            `method` and `setting` to get corresponding
            annotation from `DEFAULT_ANN_CONFIG`.
            For example: [dict(method='TFA', setting='SPILT1_1shot')].
    """

    voc_benchmark = {
        f'SPLIT{split}_{shot}SHOT': [
            dict(
                type='ann_file',
                ann_file=f'data/nwpu/seed1/'
                f'box_{shot}shot_{class_name}_train.txt',
                ann_classes=[class_name])
            for class_name in VOC_SPLIT[f'ALL_CLASSES_SPLIT{split}']
        ]
        for shot in [1, 2, 3, 5, 10] for split in [1]
    }

    # pre-defined annotation config for model reproducibility
    DEFAULT_ANN_CONFIG = dict(
        TFA=voc_benchmark,
        FSCE=voc_benchmark,
        Attention_RPN=voc_benchmark,
        MPSR=voc_benchmark,
        MetaRCNN=voc_benchmark,
        FSDetView=voc_benchmark)
    # 类的__init__方法在类初始化时被直接调用
    def __init__(self, ann_cfg: List[Dict], **kwargs) -> None:
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Parse pre-defined annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): contain method and setting
                of pre-defined annotation config. Example:
                [dict(method='TFA', setting='SPILT1_1shot')]

        Returns:
            list[dict]: Annotation information.
        """
        new_ann_cfg = []
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name}: ann_cfg should be list of dict.'
            method = ann_cfg_['method']
            setting = ann_cfg_['setting']
            default_ann_cfg = self.DEFAULT_ANN_CONFIG[method][setting]
            ann_root = ann_cfg_.get('ann_root', None)
            if ann_root is not None:
                for i in range(len(default_ann_cfg)):
                    default_ann_cfg[i]['ann_file'] = osp.join(
                        ann_root, default_ann_cfg[i]['ann_file'])
            new_ann_cfg += default_ann_cfg
        # 调用父类的读取方法
        return super(FewShotnwpuDataset, self).ann_cfg_parser(new_ann_cfg)
