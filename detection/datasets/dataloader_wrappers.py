# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Iterator

from torch.utils.data import DataLoader


class NWayKShotDataloader:
    """A dataloader wrapper.

    It Create a iterator to generate query and support batch simultaneously.
    Each batch contains query data and support data, and the lengths are
    batch_size and (num_support_ways * num_support_shots) respectively.

    Args:
        query_data_loader (DataLoader): DataLoader of query dataset
        support_data_loader (DataLoader): DataLoader of support datasets.
    """

    def __init__(self, query_data_loader: DataLoader,
                 support_data_loader: DataLoader) -> None:
        # 数据集为查询集数据加载器的数据集字段(数据包装器的mode为Query)
        self.dataset = query_data_loader.dataset
        # 采样器为查询集数据加载器的采样器(分布式无限群采样器)
        self.sampler = query_data_loader.sampler
        # 查询集与支持集的数据加载器
        self.query_data_loader = query_data_loader
        self.support_data_loader = support_data_loader

    def __iter__(self) -> Iterator:
        # if infinite sampler is used, this part of code only run once

        # 给实例添加查询迭代字段
        self.query_iter = iter(self.query_data_loader)
        # 给实例添加支持迭代字段
        self.support_iter = iter(self.support_data_loader)
        return self

    # 看起来应该是先调用__iter__才能调用__next__方法
    def __next__(self) -> Dict:
        # call query and support iterator
        # 从数据加载器迭代器中取出数据
        # 所以说训练的时候是一次性取出多个查询集图像数据与15个支持集数据的
        query_data = self.query_iter.next()
        support_data = self.support_iter.next()
        # 返回一个字典
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self) -> int:
        return len(self.query_data_loader)


class TwoBranchDataloader:
    """A dataloader wrapper.

    It Create a iterator to iterate two different dataloader simultaneously.
    Note that `TwoBranchDataloader` dose not support `EpochBasedRunner`
    and the length of dataloader is decided by main dataset.

    Args:
        main_data_loader (DataLoader): DataLoader of main dataset.
        auxiliary_data_loader (DataLoader): DataLoader of auxiliary dataset.
    """

    def __init__(self, main_data_loader: DataLoader,
                 auxiliary_data_loader: DataLoader) -> None:
        self.dataset = main_data_loader.dataset
        self.main_data_loader = main_data_loader
        self.auxiliary_data_loader = auxiliary_data_loader

    def __iter__(self) -> Iterator:
        # if infinite sampler is used, this part of code only run once
        self.main_iter = iter(self.main_data_loader)
        self.auxiliary_iter = iter(self.auxiliary_data_loader)
        return self

    def __next__(self) -> Dict:
        # The iterator actually has infinite length. Note that it can NOT
        # be used in `EpochBasedRunner`, because the `EpochBasedRunner` will
        # enumerate the dataloader forever.
        try:
            main_data = next(self.main_iter)
        except StopIteration:
            self.main_iter = iter(self.main_data_loader)
            main_data = next(self.main_iter)
        try:
            auxiliary_data = next(self.auxiliary_iter)
        except StopIteration:
            self.auxiliary_iter = iter(self.auxiliary_data_loader)
            auxiliary_data = next(self.auxiliary_iter)
        return {'main_data': main_data, 'auxiliary_data': auxiliary_data}

    def __len__(self) -> int:
        return len(self.main_data_loader)
