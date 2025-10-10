# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmdet.datasets.builder import DATASETS

from .base import BaseFewShotDataset


@DATASETS.register_module()
class QueryAwareDataset:
    """A wrapper of QueryAwareDataset.

    Building QueryAwareDataset requires query and support dataset.
    Every call of `__getitem__` will firstly sample a query image and its
    annotations. Then it will use the query annotations to sample a batch
    of positive and negative support images and annotations. The positive
    images share same classes with query, while the annotations of negative
    images don't have any category from query.

    Args:
        query_dataset (:obj:`BaseFewShotDataset`):
            Query dataset to be wrapped.
        support_dataset (:obj:`BaseFewShotDataset` | None):
            Support dataset to be wrapped. If support dataset is None,
            support dataset will copy from query dataset.
        num_support_ways (int): Number of classes for support in
            mini-batch, the first one always be the positive class.
        num_support_shots (int): Number of support shots for each
            class in mini-batch, the first K shots always from positive class.
        repeat_times (int): The length of repeated dataset will be `times`
            larger than the original dataset. Default: 1.
    """

    def __init__(self,
                 query_dataset: BaseFewShotDataset,
                 support_dataset: Optional[BaseFewShotDataset],
                 num_support_ways: int,
                 num_support_shots: int,
                 repeat_times: int = 1) -> None:
        self.query_dataset = query_dataset
        if support_dataset is None:
            self.support_dataset = self.query_dataset
        else:
            self.support_dataset = support_dataset
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        self.CLASSES = self.query_dataset.CLASSES
        self.repeat_times = repeat_times
        assert self.num_support_ways <= len(
            self.CLASSES
        ), 'Please set `num_support_ways` smaller than the number of classes.'
        # build data index (idx, gt_idx) by class.
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}
        # counting max number of annotations in one image for each class,
        # which will decide whether sample repeated instance or not.
        self.max_anns_num_one_image = [0 for _ in range(len(self.CLASSES))]
        # count image for each class annotation when novel class only
        # has one image, the positive support is allowed sampled from itself.
        self.num_image_by_class = [0 for _ in range(len(self.CLASSES))]

        for idx in range(len(self.support_dataset)):
            labels = self.support_dataset.get_ann_info(idx)['labels']
            class_count = [0 for _ in range(len(self.CLASSES))]
            for gt_idx, gt in enumerate(labels):
                self.data_infos_by_class[gt].append((idx, gt_idx))
                class_count[gt] += 1
            for i in range(len(self.CLASSES)):
                # number of images for each class
                if class_count[i] > 0:
                    self.num_image_by_class[i] += 1
                # max number of one class annotations in one image
                if class_count[i] > self.max_anns_num_one_image[i]:
                    self.max_anns_num_one_image[i] = class_count[i]

        for i in range(len(self.CLASSES)):
            assert len(self.data_infos_by_class[i]
                       ) > 0, f'Class {self.CLASSES[i]} has zero annotation'
            if len(
                    self.data_infos_by_class[i]
            ) <= self.num_support_shots - self.max_anns_num_one_image[i]:
                warnings.warn(
                    f'During training, instances of class {self.CLASSES[i]} '
                    f'may smaller than the number of support shots which '
                    f'causes some instance will be sampled multiple times')
            if self.num_image_by_class[i] == 1:
                warnings.warn(f'Class {self.CLASSES[i]} only have one '
                              f'image, query and support will sample '
                              f'from instance of same image')

        # Disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        if hasattr(self.query_dataset, 'flag'):
            self.flag = np.zeros(
                len(self.query_dataset) * self.repeat_times, dtype=np.uint8)

        self._ori_len = len(self.query_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Return query image and support images at the same time.

        For query aware dataset, this function would return one query image
        and num_support_ways * num_support_shots support images. The support
        images are sampled according to the selected query image. There should
        be no intersection between the classes of instances in query data and
        in support data.

        Args:
            idx (int): the index of data.

        Returns:
            dict: A dict contains query data and support data, it
            usually contains two fields.

                - query_data: A dict of single query data information.
                - support_data: A list of dict, has
                  num_support_ways * num_support_shots support images
                  and corresponding annotations.
        """
        idx %= self._ori_len
        # sample query data
        try_time = 0
        while True:
            try_time += 1
            cat_ids = self.query_dataset.get_cat_ids(idx)
            # query image have too many classes, can not find enough
            # negative support classes.
            if len(self.CLASSES) - len(cat_ids) >= self.num_support_ways - 1:
                break
            else:
                idx = self._rand_another(idx) % self._ori_len
            assert try_time < 100, \
                'Not enough negative support classes for ' \
                'query image, please try a smaller support way.'

        query_class = np.random.choice(cat_ids)
        query_gt_idx = [
            i for i in range(len(cat_ids)) if cat_ids[i] == query_class
        ]
        query_data = self.query_dataset.prepare_train_img(
            idx, 'query', query_gt_idx)
        query_data['query_class'] = [query_class]

        # sample negative support classes, which not appear in query image
        support_class = [
            i for i in range(len(self.CLASSES)) if i not in cat_ids
        ]
        support_class = np.random.choice(
            support_class,
            min(self.num_support_ways - 1, len(support_class)),
            replace=False)
        support_idxes = self.generate_support(idx, query_class, support_class)
        support_data = [
            self.support_dataset.prepare_train_img(idx, 'support', [gt_idx])
            for (idx, gt_idx) in support_idxes
        ]
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self) -> int:
        """Length after repetition."""
        return len(self.query_dataset) * self.repeat_times

    def _rand_another(self, idx: int) -> int:
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def generate_support(self, idx: int, query_class: int,
                         support_classes: List[int]) -> List[Tuple[int]]:
        """Generate support indices of query images.

        Args:
            idx (int): Index of query data.
            query_class (int): Query class.
            support_classes (list[int]): Classes of support data.

        Returns:
            list[tuple(int)]: A mini-batch (num_support_ways *
                num_support_shots) of support data (idx, gt_idx).
        """
        support_idxes = []
        if self.num_image_by_class[query_class] == 1:
            # only have one image, instance will sample from same image
            pos_support_idxes = self.sample_support_shots(
                idx, query_class, allow_same_image=True)
        else:
            # instance will sample from different image from query image
            pos_support_idxes = self.sample_support_shots(idx, query_class)
        support_idxes.extend(pos_support_idxes)
        for support_class in support_classes:
            neg_support_idxes = self.sample_support_shots(idx, support_class)
            support_idxes.extend(neg_support_idxes)
        return support_idxes

    def sample_support_shots(
            self,
            idx: int,
            class_id: int,
            allow_same_image: bool = False) -> List[Tuple[int]]:
        """Generate support indices according to the class id.

        Args:
            idx (int): Index of query data.
            class_id (int): Support class.
            allow_same_image (bool): Allow instance sampled from same image
                as query image. Default: False.
        Returns:
            list[tuple[int]]: Support data (num_support_shots)
                of specific class.
        """
        support_idxes = []
        num_total_shots = len(self.data_infos_by_class[class_id])

        # count number of support instance in query image
        cat_ids = self.support_dataset.get_cat_ids(idx % self._ori_len)
        num_ignore_shots = len([1 for cat_id in cat_ids if cat_id == class_id])

        # set num_sample_shots for each time of sampling
        if num_total_shots - num_ignore_shots < self.num_support_shots:
            # if not have enough support data allow repeated data
            num_sample_shots = num_total_shots
            allow_repeat = True
        else:
            # if have enough support data not allow repeated data
            num_sample_shots = self.num_support_shots
            allow_repeat = False
        while len(support_idxes) < self.num_support_shots:
            selected_gt_idxes = np.random.choice(
                num_total_shots, num_sample_shots, replace=False)

            selected_gts = [
                self.data_infos_by_class[class_id][selected_gt_idx]
                for selected_gt_idx in selected_gt_idxes
            ]
            for selected_gt in selected_gts:
                # filter out query annotations
                if selected_gt[0] == idx:
                    if not allow_same_image:
                        continue
                if allow_repeat:
                    support_idxes.append(selected_gt)
                elif selected_gt not in support_idxes:
                    support_idxes.append(selected_gt)
                if len(support_idxes) == self.num_support_shots:
                    break
            # update the number of data for next time sample
            num_sample_shots = min(self.num_support_shots - len(support_idxes),
                                   num_sample_shots)
        return support_idxes

    def save_data_infos(self, output_path: str) -> None:
        """Save data_infos into json."""
        self.query_dataset.save_data_infos(output_path)
        # for query aware datasets support and query set use same data
        paths = output_path.split('.')
        self.support_dataset.save_data_infos(
            '.'.join(paths[:-1] + ['support_shot', paths[-1]]))

    def get_support_data_infos(self) -> List[Dict]:
        """Return data_infos of support dataset."""
        return copy.deepcopy(self.support_dataset.data_infos)


@DATASETS.register_module()
class NWayKShotDataset:
    """A dataset wrapper of NWayKShotDataset.


    构建的NWayKShotDataset需要有查询与支持集数据，这个实例的行为由'mode'定义，
    当mode为查询的时候，会返回一定数量的图片和注释，当实例的mode为支持的时候，
    数据集将会建立一个批次索引，批次的数量为(num_support_ways * num_support_shots)
    支持模式想的数据集每次调用`__getitem__`方法时会返回一个批次，所以说
    数据加载器dataloader的batch应该设置为1
    模型默认为查询集模式，调用方法convert_query_to_support转换模型的mode

    Building NWayKShotDataset requires query and support dataset, the behavior
    of NWayKShotDataset is determined by `mode`. When dataset in 'query' mode,
    dataset will return regular image and annotations. While dataset in
    'support' mode, dataset will build batch indices firstly and each batch
    indices contain (num_support_ways * num_support_shots) samples. In other
    words, for support mode every call of `__getitem__` will return a batch
    of samples, therefore the outside dataloader should set batch_size to 1.
    The default `mode` of NWayKShotDataset is 'query' and by using convert
    function `convert_query_to_support` the `mode` will be converted into
    'support'.

    Args:
        query_dataset (:obj:`BaseFewShotDataset`):
            Query dataset to be wrapped.
        support_dataset (:obj:`BaseFewShotDataset` | None):
            Support dataset to be wrapped. If support dataset is None,
            support dataset will copy from query dataset.
        num_support_ways (int): Number of classes for support in
            mini-batch.
        num_support_shots (int): Number of support shot for each
            class in mini-batch.

        
        如果为真，每张图片指挥有一个注释被采样（说的一个时每张图片只采样一个物体吧）
        one_support_shot_per_image (bool): If True only one annotation will be
            sampled from each image. Default: False.
        
        在训练期间每个类别被采样和使用的支持样本的整个数量
        （结合上边的变量，这里的shot说的应该是一个物体对象）
        num_used_support_shots (int | None): The total number of support
            shots sampled and used for each class during training. If set to
            None, all shots in dataset will be used as support shot.
            Default: 200.

        是否在每个epsch开始的时候重新生成批次索引（打乱数据）
        shuffle_support (bool): If allow generate new batch indices for
            each epoch. Default: False.
        数据集的重复次数（重复后的数据集将会是原始数据集的 `times`倍）
        repeat_times (int): The length of repeated dataset will be `times`
            larger than the original dataset. Default: 1.
    """

    def __init__(self,
                 query_dataset: BaseFewShotDataset,
                 support_dataset: Optional[BaseFewShotDataset],

                 num_support_ways: int,
                 num_support_shots: int,

                 one_support_shot_per_image: bool = False,
                 num_used_support_shots: int = 200,

                 repeat_times: int = 1) -> None:
        # 给实例添加查询集与支持集
        self.query_dataset = query_dataset
        # 在这里完美解决了.如果支持集是None的话支持集直接等于查询集
        # 接下来构建的部分我已了然
        if support_dataset is None:
            self.support_dataset = self.query_dataset
        else:
            self.support_dataset = support_dataset
        # 获取类别元组（给数据集装饰器实例添加训练类别元组）
        self.CLASSES = self.query_dataset.CLASSES
        # The mode determinate the behavior of fetching data,
        # the default mode is 'query'. To convert the dataset
        # into 'support' dataset, simply call the function
        # convert_query_to_support().
        # 确定此时数据装饰器的模式
        self._mode = 'query'
        # 每批次支持集的类别数量
        self.num_support_ways = num_support_ways
        # 每张图片是否只采样一个物体对象
        self.one_support_shot_per_image = one_support_shot_per_image
        # 每个类别只使用200个物体对象
        self.num_used_support_shots = num_used_support_shots

        assert num_support_ways <= len(
            self.CLASSES
        ), 'support way can not larger than the number of classes'

        # 每批次每个类别支持集的物体对象的个数
        self.num_support_shots = num_support_shots

        # 批次索引列表
        # 当调用模型转换为支持方法时(何时调用这个方法)，修改这个列表，列表的长度为查询集数据的数量，每个元素也为一个列表
        # 列表中的元素是元组，元组的个数为num_support_ways*num_support_shots
        # 每个元组的结构为(idx, gt_idx)，idx为查询集中图片的索引，gt_idx为这个图片注释标签列表索引
        self.batch_indices = []

        # 一个字典，其中有len(self.CLASSES)个元素，键为0到len(self.CLASSES)-1
        # 每个值都是一个空列表
        # 类别对象查找表，每个列表在经过下边的
        # self.prepare_support_shots()方法后都起码有一次采样的所需数量的物体对象
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}

        # 调用 准备支持集数据的方法 (这里需要注意的一点是one_support_shot_per_image=True这个参数)
        # 此参数若为True,每张支持集图像只加载一个标注信息
        # 而这一点在微调的时候是没有必要的,因为在加载数据集的时候添加了一个ann_classes参数令每张照片只有他对应那个类别的注释信息了
        # 在加载数据的时候就已经解决这个问题了,不需要在构建数据包装器的时候再取控制了
        # 通过对这里打断点可以看出来，所有的图片取的都是第一个标注框
        self.prepare_support_shots()

        # 数据的重复次数
        self.repeat_times = repeat_times
        # Disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        # 禁用组采样器，因为在少样本设置中，一组可能只有两到三张图片。
        # 将所有查询集图片都设置为一组0，self.repeat_times是数据应该重复的次数

        # 需要注意的是，如果此时不为测试模型，这个条件一定为真，这个装饰数据实例一定会加上flag字段
        # 全0数组的长度为查询集的长度乘以数据集的重复次数
        if hasattr(query_dataset, 'flag'):
            self.flag = np.zeros(
                len(self.query_dataset) * self.repeat_times, dtype=np.uint8)

        # 记录查询集的原始数据个数
        self._ori_len = len(self.query_dataset)

    # dataloader类要求数据集必须重写这个类
    def __getitem__(self, idx: int) -> Union[Dict, List[Dict]]:
        """ 返回支持集或者查询集中索引为idx的数据
        Args:
            idx(int):抽取图片的索引

        return:
            当实例的mode为'query'，返回  查询集数据中索引idx的数据     
            当实例的mode为'support'，返回列表      
        """
        # 实例的mode为query的时候
        if self._mode == 'query':
            # 这里的意思是查询集的索引是会超过原始查询集数据的数量的（详细看看）
            idx %= self._ori_len
            # loads one data in query pipeline
            # 返回结果字典（此方法的返回值就是图片通过piplines后的结果）
            return self.query_dataset.prepare_train_img(idx, 'query')
        # 实例的mode为support的时候
        elif self._mode == 'support':
            # loads one batch of data in support pipeline
            # 获得idx对应组别的支持集信息列表，这个列表装有（抽取类别*每个类别的抽取数量）个元组
            b_idx = self.batch_indices[idx]
            # 这个列表会有（抽取类别*每个类别的抽取数量）个字典
            batch_data = [
                # 调用准备训练图像方法，传参（图片索引，pipeline通道key，图片标签列表索引）
                # 这里的[gt_idx]为什么会是列表呢
                # (因为不论前期如何操作，gt_idx始终都为一个单整形数)（后续再考虑为什么）
                self.support_dataset.prepare_train_img(idx, 'support',
                                                       [gt_idx])
                # 从支持集索引列表中遍历元素，(idx, gt_idx)分别为图片列表索引与图片标签列表索引
                # 需要说明的是 图片中类别标签索引gt_idx 大多会是0（如果每张图片选取一个物体对象为True的话）
                for (idx, gt_idx) in b_idx
            ]
            return batch_data
        else:
            raise ValueError('not valid data type')

    # 数据包装器的len方法（这与数据包装器的mode有关）
    def __len__(self) -> int:
        """Length of dataset."""
        # 如果数据包装器的模式为'query'，返回查询集的图片数量*数据重复次数
        if self._mode == 'query':
            return len(self.query_dataset) * self.repeat_times
        # 如果数据包装器的模式为'support'，返回查询集的支持集的批次列表长度
        elif self._mode == 'support':
            return len(self.batch_indices)
        else:
            raise ValueError(f'{self._mode}not a valid mode')

    def prepare_support_shots(self) -> None:
        # 为同一类中的注释创建查找表。
        # create lookup table for annotations in same class.

        # 遍历支持数据集的data_infos列表
        for idx in range(len(self.support_dataset)):
            # 得到该图片的标签信息
            labels = self.support_dataset.get_ann_info(idx)['labels']
            # 抽取标签信息，gt_idx为物体顺序位置, gt为物体的类别索引
            for gt_idx, gt in enumerate(labels):
                # When the number of support shots reaches
                # `num_used_support_shots`, the class will be skipped
                # 确保此类别并没有抽样到规定上限(每个类别只有200个物体对象,由于这里被设置为每个图像只有一个物体对象,所以说每个类别也只有200个图片)
                if len(self.data_infos_by_class[gt]) < \
                        self.num_used_support_shots:
                    # 向类别注释查找表中添加这个注释的信息，
                    # 为元组（图片索引，图片中注释对应的索引）
                    self.data_infos_by_class[gt].append((idx, gt_idx))
                    # When `one_support_shot_per_image` is true, only one
                    # annotation will be sampled for each image.
                    # 如果每张图片只选择一个物体对象，则直接退出循环，查找下一张图片
                    # break跳出本次循环
                    if self.one_support_shot_per_image:
                        break
        # make sure all class index lists have enough
        # instances (length > num_support_shots)
        # 确保查找表中每个类别的物体对象个数都不小于进行一次采样所需的物体对象个数
        # （最少扩充到原本物体样本个数的两倍）（但是还有一个问题是，万一某个类别一个物体样本都没有就不好办了）
        for i in range(len(self.CLASSES)):
            num_gts = len(self.data_infos_by_class[i])
            if num_gts < self.num_support_shots:
                self.data_infos_by_class[i] = self.data_infos_by_class[i] * (
                    self.num_support_shots // num_gts + 1)

    # 将数据集装饰器的模式由查询变为支持
    # 关键是在什么时间调用这个方法呢，我看数据准备部分是没有的
    def convert_query_to_support(self, support_dataset_len: int) -> None:
        """Convert query dataset to support dataset.

        Args:
        转换mode则得到所有支持集索引，支持集索引的组数为调用转换方法时传入的查询集的长度
            support_dataset_len (int): Length of pre sample batch indices.
        """
        # 转换mode得到支持集批次索引列表，
        # 这个列表的变化我在代码中定义它的位置说明了它的变化，这里写了会显得很多
        self.batch_indices = \
            self.generate_support_batch_indices(support_dataset_len)
        self._mode = 'support'
        # 注意，如果模型不是测试模型，数据装饰实例的flag字段一定会在此处修改
        # 数据长度修改为传入的查询集的长度
        if hasattr(self, 'flag'):
            self.flag = np.zeros(support_dataset_len, dtype=np.uint8)

    def generate_support_batch_indices(
            self, dataset_len: int) -> List[List[Tuple[int]]]:
        """Generate batch indices from support dataset.

        Batch indices is in the shape of [length of datasets * [support way *
        support shots]]. And the `dataset_len` will be the length of support
        dataset.

        Args:
            dataset_len (int): Length of batch indices.

        Returns:
            list[list[(data_idx, gt_idx)]]: Pre-sample batch indices.
        """
        # 创建整体索引列表
        total_indices = []
        # 这里的意思是应该抽取多少组支持集
        for _ in range(dataset_len):
            # 创建一个batch_indices，存储本组支持集的物体对象信息
            batch_indices = []
            # 抽取类别索引，这里的意思是说num_support_ways=15这个参数其实指的是支持集应该包括多少个类别
            # 这里从总类别中抽取num_support_ways个类别索引，但是索引列表与需要抽取的索引数量是一样的
            # 所以这里的行为是将类别索引随机排序一下
            # 这个列表的名字是被选择的类别
            selected_classes = np.random.choice(
                len(self.CLASSES), self.num_support_ways, replace=False)
            # 从被选择的类别列表中选取类别
            for cls in selected_classes:
                # 确定数据集中这个类别中有多少个物体对象
                num_gts = len(self.data_infos_by_class[cls])
                # 跟上边的选取方法是一样的，从这些物体对象以索引的形式中选取一个物体（在这个类别的列表中的索引）
                # num_support_shots=1，选取一个物体对象
                selected_gts_idx = np.random.choice(
                    num_gts, self.num_support_shots, replace=False)
                
                # 列表推导式
                selected_gts = [
                    # 从按类别的数据信息列表中通过[类别名][物体索引]的方式得到物体信息（图片索引，该图片中类别标签索引）
                    # 需要说明的是 图片中类别标签索引 大多会是0（如果每张图片选取一个物体对象为True的话）
                    self.data_infos_by_class[cls][gt_idx]
                    # 首先从被选择的真实物体索引列表中抽取索引
                    for gt_idx in selected_gts_idx
                ]
                # 然后向本组支持集物体对象列表 中添加 本类别中抽取的物体信息（这里是合并操作）
                batch_indices.extend(selected_gts)
            # 向总体索引列表 中 添加本组支持集信息列表
            total_indices.append(batch_indices)
        return total_indices

    def save_data_infos(self, output_path: str) -> None:
        """Save data infos of query and support data."""
        self.query_dataset.save_data_infos(output_path)
        paths = output_path.split('.')
        self.save_support_data_infos('.'.join(paths[:-1] +
                                              ['support_shot', paths[-1]]))

    def save_support_data_infos(self, support_output_path: str) -> None:
        """Save support data infos."""
        support_data_infos = self.get_support_data_infos()
        meta_info = [{
            'CLASSES': self.CLASSES,
            'img_prefix': self.support_dataset.img_prefix
        }]
        from .utils import NumpyEncoder
        with open(support_output_path, 'w', encoding='utf-8') as f:
            json.dump(
                meta_info + support_data_infos,
                f,
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder)

    # 此方法使用嵌套循环与列表推导式来获得查询集中所有图片(注释信息只有一个物体对象)的信息[{},{},{}...]
    # (需要注意的是这里获取到的注释信息并不完全,
    # 因为self.data_infos_by_class在经过self.prepare_support_shots()方法处理之后每张图片只保留一个物体的注释信息)
    def get_support_data_infos(self) -> List[Dict]:
        """Get support data infos from batch indices."""
        return copy.deepcopy([
            # 获得支持集数据集中第idx张图片的第gt_idx个标注物体的信息(字典)
            self._get_shot_data_info(idx, gt_idx)
            # 嵌套循环从上向下执行
            # 首先是从这个字典中选出类别名
            for class_name in self.data_infos_by_class.keys()
            # 然后是根据类别名循环取出字典中该键对应的列表中的所有物体信息
            # (idx, gt_idx)为物体图片索引与物体在图片标签注释中的位置索引
            for (idx, gt_idx) in self.data_infos_by_class[class_name]
        ])

    # 该方法获得支持集数据集中第idx张图片的第gt_idx标注物体的信息
    def _get_shot_data_info(self, idx: int, gt_idx: int) -> Dict:
        """Get data info by idx and gt idx."""
        # 复制支持集中idx索引的图片信息(字典)
        data_info = copy.deepcopy(self.support_dataset.data_infos[idx])
        # 修改注释信息,在注释信息的标签与边界框字段中使用gt_idx位置的信息代替原信息
        data_info['ann']['labels'] = \
            data_info['ann']['labels'][gt_idx:gt_idx + 1]
        data_info['ann']['bboxes'] = \
            data_info['ann']['bboxes'][gt_idx:gt_idx + 1]
        # 返回数据信息(字典)
        return data_info


@DATASETS.register_module()
class TwoBranchDataset:
    """A dataset wrapper of TwoBranchDataset.

    Wrapping main_dataset and auxiliary_dataset to a single dataset and thus
    building TwoBranchDataset requires two dataset. The behavior of
    TwoBranchDataset is determined by `mode`. Dataset will return images
    and annotations according to `mode`, e.g. fetching data from
    main_dataset if `mode` is 'main'. The default `mode` is 'main' and
    by using convert function `convert_main_to_auxiliary` the `mode`
    will be converted into 'auxiliary'.

    Args:
        main_dataset (:obj:`BaseFewShotDataset`):
            Main dataset to be wrapped.
        auxiliary_dataset (:obj:`BaseFewShotDataset` | None):
            Auxiliary dataset to be wrapped. If auxiliary dataset is None,
            auxiliary dataset will copy from main dataset.
        reweight_dataset (bool): Whether to change the sampling weights
            of VOC07 and VOC12 . Default: False.
    """

    def __init__(self,
                 main_dataset: BaseFewShotDataset = None,
                 auxiliary_dataset: Optional[BaseFewShotDataset] = None,
                 reweight_dataset: bool = False) -> None:
        assert main_dataset and auxiliary_dataset
        self._mode = 'main'
        self.main_dataset = main_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.CLASSES = self.main_dataset.CLASSES
        if reweight_dataset:
            # Reweight the VOC dataset to be consistent with the original
            # implementation of MPSR. For more details, please refer to
            # https://github.com/jiaxi-wu/MPSR/blob/master/maskrcnn_benchmark/data/datasets/voc.py#L137
            self.main_idx_map = self.reweight_dataset(
                self.main_dataset,
                ['VOC2007', 'VOC2012'],
            )
            self.auxiliary_idx_map = self.reweight_dataset(
                self.auxiliary_dataset, ['VOC'])
        else:
            self.main_idx_map = list(range(len(self.main_dataset)))
            self.auxiliary_idx_map = list(range(len(self.auxiliary_dataset)))
        self._main_len = len(self.main_idx_map)
        self._auxiliary_len = len(self.auxiliary_idx_map)
        self._set_group_flag()

    def __getitem__(self, idx: int) -> Dict:
        if self._mode == 'main':
            idx %= self._main_len
            idx = self.main_idx_map[idx]
            return self.main_dataset.prepare_train_img(idx, 'main')
        elif self._mode == 'auxiliary':
            idx %= self._auxiliary_len
            idx = self.auxiliary_idx_map[idx]
            return self.auxiliary_dataset.prepare_train_img(idx, 'auxiliary')
        else:
            raise ValueError('not valid data type')

    def __len__(self) -> int:
        """Length of dataset."""
        if self._mode == 'main':
            return self._main_len
        elif self._mode == 'auxiliary':
            return self._auxiliary_len
        else:
            raise ValueError('not valid data type')

    def convert_main_to_auxiliary(self) -> None:
        """Convert main dataset to auxiliary dataset."""
        self._mode = 'auxiliary'
        self._set_group_flag()

    def save_data_infos(self, output_path: str) -> None:
        """Save data infos of main and auxiliary data."""
        self.main_dataset.save_data_infos(output_path)
        paths = output_path.split('.')
        self.auxiliary_dataset.save_data_infos(
            '.'.join(paths[:-1] + ['auxiliary', paths[-1]]))

    def _set_group_flag(self) -> None:
        # disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @staticmethod
    def reweight_dataset(dataset: BaseFewShotDataset,
                         group_prefix: Sequence[str],
                         repeat_length: int = 100) -> List:
        """Reweight the dataset."""

        groups = [[] for _ in range(len(group_prefix))]
        for i in range(len(dataset)):
            filename = dataset.data_infos[i]['filename']
            for j, prefix in enumerate(group_prefix):
                if prefix in filename:
                    groups[j].append(i)
                    break
                assert j < len(group_prefix) - 1

        # Reweight the dataset to be consistent with the original
        # implementation of MPSR. For more details, please refer to
        # https://github.com/jiaxi-wu/MPSR/blob/master/maskrcnn_benchmark/data/datasets/voc.py#L137
        reweight_idx_map = []
        for g in groups:
            if len(g) < 50:
                reweight_idx_map += g * (int(repeat_length / len(g)) + 1)
            else:
                reweight_idx_map += g
        return reweight_idx_map
