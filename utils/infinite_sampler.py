# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler

from .dist_utils import sync_random_seed


class InfiniteSampler(Sampler):
    """Return a infinite stream of index.

    The length of sampler is set to the actual length of dataset, thus the
    length of dataloader is still determined by the dataset. The
    implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Iterable): The dataset.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset: Iterable,
                 seed: int = 0,
                 shuffle: bool = True) -> None:
        self.dataset = dataset
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices()
        self.epoch = 0

    def _infinite_indices(self) -> Iterator:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()
            else:
                yield from torch.arange(self.size).tolist()

    def _indices(self) -> Iterator:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), 0, None)

    def __iter__(self) -> Iterator:
        yield from self.indices

    def __len__(self) -> int:
        """Length of dataset."""
        # The length of sampler is set to the actual length of dataset, thus
        # the length of dataloader is still determined by the dataset.
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class InfiniteGroupSampler(Sampler):
    """Similar to `InfiniteSampler`, but all indices in a batch should be in
    the same group of flag.

    The length of sampler is set to the actual length of dataset, thus the
    length of dataloader is still determined by the dataset. The
    implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Iterable): The dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU. Default: 1.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy `epoch`, it
            should be noted that `shuffle` can not guarantee that you can
            generate sequential indices because it need to ensure
            that all indices in a batch is in a group. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset: Iterable,
                 samples_per_gpu: int = 1,
                 seed: int = 0,
                 shuffle: bool = True) -> None:
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        # buffer used to save indices of each group
        self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}

        self.size = len(dataset)
        self.indices = self._indices_of_rank()
        self.epoch = 0

    def _infinite_indices(self) -> Iterator:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()
            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self) -> Iterator:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), 0, None)

    def __iter__(self) -> Iterator:
        # once batch size is reached, yield the indices
        for idx in self.indices:
            flag = self.flag[idx]
            group_buffer = self.buffer_per_group[flag]
            group_buffer.append(idx)
            if len(group_buffer) == self.samples_per_gpu:
                for i in range(self.samples_per_gpu):
                    yield group_buffer[i]
                del group_buffer[:]

    def __len__(self) -> int:
        """Length of dataset."""
        # The length of sampler is set to the actual length of dataset, thus
        # the length of dataloader is still determined by the dataset.
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedInfiniteSampler(Sampler):
    """Similar to `InfiniteSampler` but in distributed version.

    The length of sampler is set to the actual length of dataset, thus the
    length of dataloader is still determined by the dataset. The
    implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Iterable): The dataset.
        num_replicas (int | None): Number of processes participating in
            distributed training. Default: None.
        rank (int | None): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset: Iterable,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0,
                 shuffle: bool = True) -> None:
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices_of_rank()
        self.epoch = 0

    def _infinite_indices(self) -> Iterator:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            if self.shuffle:
                indices = []
                for _ in range(self.num_replicas):
                    indices += torch.randperm(self.size, generator=g).tolist()
                yield from indices
            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self) -> Iterator:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.num_replicas)

    def __iter__(self) -> Iterator:
        yield from self.indices

    def __len__(self):
        """return length of dataset."""
        # The length of sampler is set to the actual length of dataset, thus
        # the length of dataloader is still determined by the dataset.
        return math.ceil(self.size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedInfiniteGroupSampler(Sampler):
    """Similar to `InfiniteGroupSampler` but in distributed version.

    dataloader返回的其实是一个装有数据和标签的列表，想知道细节请参考本地项目
    D:\Apython\pythonprojects\pytorch1.8.2projects\Strain_CIFAR10\gpu1.py

    The length of sampler is set to the actual length of dataset, thus the
    length of dataloader is still determined by the dataset. The
    implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Iterable): The dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU. Default: 1.
    参与分布式培训的进程数。world_size
        num_replicas (int | None): Number of processes participating in
            distributed training. Default: None.
    当前进程的排名
        rank (int | None): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy `epoch`, it
            should be noted that `shuffle` can not guarantee that you can
            generate sequential indices because it need to ensure
            that all indices in a batch is in a group. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset: Iterable,
                #  每batch每个gpu采样多少图片
                 samples_per_gpu: int = 1,
                #  world_size，参与分布式数据载入的进程数
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0,
                 shuffle: bool = True) -> None:
        # 确定总进程数与当前进程的排名
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.rank = rank
        self.num_replicas = num_replicas

        # 数据集
        self.dataset = dataset
        # 每个gpu的采样数
        self.samples_per_gpu = samples_per_gpu

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        # 由于所有进程采样的数据应该不可以重复，所以所有进程使用同一个随机种子来打乱数据集索引，
        # 不同进程使用不同索引来采样数据，这样就可以达到所有进程采样的数据不重复的目的了
        # 这里应该返回的就是seed原来的值吧
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle

        # 'flag'为全0数组，长度为查询集图片数量*数据重复次数
        assert hasattr(self.dataset, 'flag')
        # 给采样器添加flag字段
        # 'flag'为全0数组，长度为查询集图片数量*数据重复次数
        self.flag = self.dataset.flag

        # 获取数据集的组别数与每组中的元素个数（这是np的一个统计方法）
        self.group_sizes = np.bincount(self.flag)

        # buffer used to save indices of each group
        # 用于保存每组索引的缓冲区
        # 创建组别数字典（k为self.group_sizes的索引）
        self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}

        # 采样器大小为数据集的长度，这取决于此时数据集的mode，
        # 此时应该为'query',大小为查询集的数据量*数据重复次数
        self.size = len(dataset)

        # 得到一个根据线程的无限索引生成器对象,使用yield关键字令方法变成一个生成器，这很有趣
        # 为什么，根据代码来看，由于使用了迭代器（这是惰性取值，所以不会陷入死循环循环）
        # 而根据切片方法来看，舍弃掉前self.rank个数据（这表明了此进程为第几个进程），
        # 然后将以后所有的数据每self.num_replicas个分为一组取出每组数据的第一个元素（这会返回一个迭代器）
        # 由于我们的随机种子统一设置，每个进程取到的数据都会是不一样
        self.indices = self._indices_of_rank()
        # 添加一个'epoch'字段（动态语言会在调用的时候填充内部变量）
        self.epoch = 0

    def _infinite_indices(self) -> Iterator:
        """Infinitely yield a sequence of indices."""
        # 创建生成器
        g = torch.Generator()
        # 只有在取数据的时候才会关注self.epoch是否存在
        # 给生成器设置随机种子
        g.manual_seed(self.seed + self.epoch)
        while True:
            # 如果每个epoch需要打乱数据
            if self.shuffle:
                # 创建索引列表
                indices = []
                # 循环系统的总进程数量次
                for _ in range(self.num_replicas):
                    # 使用生成器生成self.size大小的随机序列（由于随机种子一样，每个序列会完全一样）
                    indices += torch.randperm(self.size, generator=g).tolist()
                yield from indices
            else:
                yield from torch.arange(self.size).tolist()
    # 创建一个根据线程的无限索引生成器
    def _indices_of_rank(self) -> Iterator:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.num_replicas)

    # 这也会返回一个生成器（通过组别缓冲区字典来实现每次只处理同组别的数据）
    def __iter__(self) -> Iterator:
        # once batch size is reached, yield the indices
        for idx in self.indices:
            # 获取图片组别:0
            flag = self.flag[idx]
            # 从每个组别的缓冲区字典中获得该组别的缓冲列表的引用（知道这里是引用很重要，
            # 通过对这个引用的修改我们可以直接修改字典中对应的值）
            group_buffer = self.buffer_per_group[flag]
            # 向这个组别缓冲区列表中添加这个图片的索引
            group_buffer.append(idx)

            # 如果组别缓存的元素数量等于每个gpu的采样数量，则依次取出数缓存中的元素
            if len(group_buffer) == self.samples_per_gpu:
                # 循环返回这个缓存组别列表中的数据
                for i in range(self.samples_per_gpu):
                    # 我们来思考一下这个代码，当代码检查到yield后会停下，然后将group_buffer[i]这个数据yield出去
                    # 所以说其实if len(group_buffer) == self.samples_per_gpu:这个条件看似是找到批次数大小的数据后
                    # yield出去，但是这里其实是分别yield出去的，数据加载器会不断地调用这个采样器的iter方法，直到
                    # yield出去的数据等于batch_size
                    yield group_buffer[i]
                # 删除缓冲字典中对应键的列表中的所有值
                del group_buffer[:]

    def __len__(self) -> int:
        """return length of dataset."""
        # The length of sampler is set to the actual length of dataset, thus
        # the length of dataloader is still determined by the dataset.
        # 采样器的长度为数据集长度除以总线程数
        return math.ceil(self.size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
