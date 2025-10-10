# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from functools import partial
from typing import Dict, Optional, Tuple

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import ConfigDict, build_from_cfg
from mmdet.datasets.builder import DATASETS, worker_init_fn
from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                             ConcatDataset, RepeatDataset)
from mmdet.datasets.samplers import (DistributedGroupSampler,
                                     DistributedSampler, GroupSampler)
from torch.utils.data import DataLoader, Dataset, Sampler

from mmfewshot.utils.infinite_sampler import (DistributedInfiniteGroupSampler,
                                              DistributedInfiniteSampler,
                                              InfiniteGroupSampler)
from .dataset_wrappers import (NWayKShotDataset, QueryAwareDataset,
                               TwoBranchDataset)
from .utils import get_copy_dataset_type


def build_dataset(cfg: ConfigDict,
                #   在测试阶段默认参数为dict(test_mode=True)
                  default_args: Dict = None,
                  rank: Optional[int] = None,
                  work_dir: Optional[str] = None,
                  timestamp: Optional[str] = None) -> Dataset:
    # If save_dataset is set to True, dataset will be saved into json.
    save_dataset = cfg.pop('save_dataset', False)

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'QueryAwareDataset':
        query_dataset = build_dataset(cfg['dataset'], default_args)
        # build support dataset
        if cfg.get('support_dataset', None) is not None:
            # if `copy_from_query_dataset` is True, copy and update config
            # from query_dataset and copy `data_infos` by using copy dataset
            # to avoid reproducing random sampling.
            if cfg['support_dataset'].pop('copy_from_query_dataset', False):
                support_dataset_cfg = copy.deepcopy(cfg['dataset'])
                support_dataset_cfg.update(cfg['support_dataset'])
                support_dataset_cfg['type'] = get_copy_dataset_type(
                    cfg['dataset']['type'])
                support_dataset_cfg['ann_cfg'] = [
                    dict(data_infos=copy.deepcopy(query_dataset.data_infos))
                ]
                cfg['support_dataset'] = support_dataset_cfg
            support_dataset = build_dataset(cfg['support_dataset'],
                                            default_args)
        # support dataset will be a copy of query dataset in QueryAwareDataset
        else:
            support_dataset = None

        dataset = QueryAwareDataset(
            query_dataset,
            support_dataset,
            num_support_ways=cfg['num_support_ways'],
            num_support_shots=cfg['num_support_shots'],
            repeat_times=cfg.get('repeat_times', 1))
# 看这段够用了'NWayKShotDataset'
    elif cfg['type'] == 'NWayKShotDataset':
        # 递归构建查询集(default_args==None)
        # /autodl-tmp/mmfewshot/mmfewshot/detection/datasets/voc.py
        query_dataset = build_dataset(cfg['dataset'], default_args)
        # build support dataset
            # 需要说明的是支持集其实是有三个数据源的,
            # 一个与查询集一样self.data_infos,
            # 另一个则是self.data_infos_by_class
            # 最后是self.batch_indices
            # (我希望你可以在看完源码之后体会,注意NWayKShotDataset的NWayKShotDataset方法)

        # 如果配置文件没有支持集数据集的话
        if cfg.get('support_dataset', None) is not None:
            # if `copy_from_query_dataset` is True, copy and update config
            # from query_dataset and copy `data_infos` by using copy dataset
            # to avoid reproducing random sampling.
            # 跳过此分支
            if cfg['support_dataset'].pop('copy_from_query_dataset', False):
                # 支持集配置从查询集配置复制
                support_dataset_cfg = copy.deepcopy(cfg['dataset'])
                # 使用原始配置中的支持集配置更新support_dataset_cfg
                support_dataset_cfg.update(cfg['support_dataset'])
                # 使用查询集的数据类型更新support_dataset_cfg['type']
                support_dataset_cfg['type'] = get_copy_dataset_type(
                    cfg['dataset']['type'])
                # 直接获取查询集的数据注释信息
                # (感觉这段代码写的是有问题的，FewShotVOCDataset类中并没有写如何处理这段段代码的情况下的'ann_cfg')
                # 不要你感觉,这是因为压根就没有进入这个分支
                support_dataset_cfg['ann_cfg'] = [
                    dict(data_infos=copy.deepcopy(query_dataset.data_infos))
                ]
                # 使用support_dataset_cfg替换原始训练配置的support_dataset
                cfg['support_dataset'] = support_dataset_cfg

            # 构建支持数据集（我看配置文件中两个数据集唯一的区别在于是否使用难例）
            support_dataset = build_dataset(cfg['support_dataset'],
                                            default_args)
        # support dataset will be a copy of query dataset in NWayKShotDataset
        # 在微调阶段支持集是没有的,进入此分支
        else:
            support_dataset = None

        # NWayKShotDataset是一个数据包装器(包装查询集和支持集)
        dataset = NWayKShotDataset(
            query_dataset,
            support_dataset,
            # 以下参数我都不太明白(现在明白了)

            # 支持集每次抽样的类别数
            num_support_ways=cfg['num_support_ways'],
            # 支持集每次抽样每个类别的样本数目
            num_support_shots=cfg['num_support_shots'],
            # 如果为真，每张图片指挥有一个注释被采样
            one_support_shot_per_image=cfg.get('one_support_shot_per_image',
                                               False),
            # 200
            # 在训练期间支持集中每个类别只从200个物体对象中抽样
            num_used_support_shots=cfg.get('num_used_support_shots', None),

            repeat_times=cfg.get('repeat_times', 1),
        )
    elif cfg['type'] == 'TwoBranchDataset':
        main_dataset = build_dataset(cfg['dataset'], default_args)
        # if `copy_from_main_dataset` is True, copy and update config
        # from main_dataset and copy `data_infos` by using copy dataset
        # to avoid reproducing random sampling.
        if cfg['auxiliary_dataset'].pop('copy_from_main_dataset', False):
            auxiliary_dataset_cfg = copy.deepcopy(cfg['dataset'])
            auxiliary_dataset_cfg.update(cfg['auxiliary_dataset'])
            auxiliary_dataset_cfg['type'] = get_copy_dataset_type(
                cfg['dataset']['type'])
            auxiliary_dataset_cfg['ann_cfg'] = [
                dict(data_infos=copy.deepcopy(main_dataset.data_infos))
            ]
            cfg['auxiliary_dataset'] = auxiliary_dataset_cfg
        auxiliary_dataset = build_dataset(cfg['auxiliary_dataset'],
                                          default_args)
        dataset = TwoBranchDataset(
            main_dataset=main_dataset,
            auxiliary_dataset=auxiliary_dataset,
            reweight_dataset=cfg.get('reweight_dataset', False))
# 这个分支就是给构建数据集准备的(验证集数据构建直接进入此分支)
    else:
        # 本例中数据集类路径
        # /autodl-tmp/mmfewshot/mmfewshot/detection/datasets/voc.py
        # type='FewShotVOCDataset'
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    # save dataset for the reproducibility
    # 先不看这段吧
    if rank == 0 and save_dataset:
        save_dataset_path = osp.join(work_dir, f'{timestamp}_saved_data.json')
        if hasattr(dataset, 'save_data_infos'):
            dataset.save_data_infos(save_dataset_path)
        else:
            raise AttributeError(
                f'`save_data_infos` is not implemented in {type(dataset)}.')
    # NWayKShotDataset包装器对象
    return dataset

# 构建数据集加载器
def build_dataloader(dataset: Dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus: int = 1,
                    #  分布式
                     dist: bool = True,
                    #  是否在每个epoch开始的时候重新打乱数据:默认打乱数据
                     shuffle: bool = True,
                    #  随机种子
                     seed: Optional[int] = None,
                     data_cfg: Optional[Dict] = None,
                    #  是否无限采样器:是
                     use_infinite_sampler: bool = False,
                     **kwargs) -> DataLoader:
    """Build PyTorch DataLoader.
    在分布式训练中，每个gpu或者进程都有一个数据加载器
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
            Default:1.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int): Random seed. Default:None.
        data_cfg (dict | None): Dict of data configure. Default: None.
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    # 获取此进程的排名，与总主进程数
    # 0，1
    rank, world_size = get_dist_info()

    # 创建采样器（此处为无限采样器）
    # 训练阶段查询集返回分布式无限群采样器对象，batch_size(每个gpu的采样数，此时dataset的mode仍然是查询集模式),每个gpu的线程数
    # 测试状态下数据采样器为DistributedSampler分布式非群有限采样器
    # 模型初始化数据采样器为GroupSampler非分布式群有限采样器,

    # 训练阶段的查询集和支持集的数据加载器均为分布式,打乱,无限采样器
    # 测试阶段数据加载器为分布式,不打乱,非无限采样器
    # 模型初始化数据加载器为非分布式,不打乱,非无限加载器
    # (),2,1
    (sampler, batch_size, num_workers) = build_sampler(
        dist=dist,#测试下为分布式(模型初始化不是分布式)
        shuffle=shuffle,#测试下为不打乱数据
        dataset=dataset,
        # 不重要(模型初始化数据集gpu数量为1)
        num_gpus=num_gpus,
        samples_per_gpu=samples_per_gpu,
        # 不重要
        workers_per_gpu=workers_per_gpu,
        # 使用相同的随机种子来采样，确保一个batch中抽不到相同的样本
        seed=seed,
        # 无限采样器
        use_infinite_sampler=use_infinite_sampler)#测试下为有限采样器
    
    # 这是一个偏方法，预设一个worker_init_fn方法，该方法还缺少一个worker_id参数
    # 此方法在被调用时给numpy与python设置随机种子
    # 不做要求
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    
    if isinstance(dataset, QueryAwareDataset):
        from mmfewshot.utils import multi_pipeline_collate_fn

        # `QueryAwareDataset` will return a list of DataContainer
        # `multi_pipeline_collate_fn` are designed to handle
        # the data with list[list[DataContainer]]
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,     
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
    # 看这个分支
    elif isinstance(dataset, NWayKShotDataset):

        from mmfewshot.utils import multi_pipeline_collate_fn
        # 引入少样本数据加载器类
        from .dataloader_wrappers import NWayKShotDataloader
        # `NWayKShotDataset` will return a list of DataContainer
        # `multi_pipeline_collate_fn` are designed to handle
        # the data with list[list[DataContainer]]
        # initialize query dataloader

        # 初始化查询集数据集加载器(其中multi_pipeline_collate_fn方法比较重要，但是也比较复杂，我已经打好注释)
        # 我认为关于num_workers的参数其实不需要过多了解
        # 构建查询集数据加载器(此时dataset的mode为'query')
        # 数据加载器的len属性被sampler的len属性指定
        # 这里你也许会有一个问题,就是我的采样器已经内置了batch_size,为什么还要给DataLoader传入一个batch_size
        # 这里我不细讲,既然你已经知道了yield的用法,就应该知道他是一个懒惰迭代器了
        query_data_loader = DataLoader(
            # 数据集包装器
            dataset,
            batch_size=batch_size,
            # 分布式无限群采样器对象
            sampler=sampler,
            # 不做要求
            num_workers=num_workers,
            # 此方法较为复杂
            collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            # 不做要求
            worker_init_fn=init_fn,
            **kwargs)

        # 在构建支持集的时候，将传入进来的dataset列表深拷贝一次
        # 为什要进行这个操作，深拷贝将dataset实例重新创建一份，
        # 为了避免后续我们在取出查询集集数据的时候使用的是支持集数据(convert_query_to_support会改变实例的mode)
        support_dataset = copy.deepcopy(dataset)

        # if infinite sampler is used, the length of batch indices in
        # support_dataset can be longer than the length of query dataset
        # as it can achieve better sample diversity
        # 无限采样器
        if use_infinite_sampler:
            # 由于我们拷贝出来的数据集的mode其实还是query，所以此处给支持集传唤mode转换为support
            # flag字段在此处被改变(原来flag参数是这么使用的)
            # convert_query_to_support方法传入的参数决定了支持集batch_indices列表中的元素个数
            # (这个列表中的每个元素都是一个长度为15的列表)
            support_dataset.convert_query_to_support(len(dataset) * num_gpus)
        # create support dataset from query dataset and
        # sample batch indices with same length as query dataloader
        else:
            support_dataset.convert_query_to_support(
                len(query_data_loader) * num_gpus)

        # 使用相同的配方构建查询集数据加载器
        (support_sampler, _, _) = build_sampler(
            dist=dist,
            # 其实在构建batch_indices列表的时候已经有过一次随机抽样了，这里不打乱数据可以理解
            shuffle=False,
            dataset=support_dataset,
            # 不重要
            num_gpus=num_gpus,
            # 每个gpu的采样数为1,但是我们每次其实是获取15张图片,
            # 奥秘在于嵌套列表[[]],最内部字典中有15个元素,我们抽样的结果其实是抽样了一个字典
            samples_per_gpu=1,
            # 不做要求
            workers_per_gpu=workers_per_gpu,
            seed=seed,
            use_infinite_sampler=use_infinite_sampler)
        # support dataloader is initialized with batch_size 1 as default.
        # each batch contains (num_support_ways * num_support_shots) images,
        # since changing batch_size is equal to changing num_support_shots.
        support_data_loader = DataLoader(
            support_dataset,
            batch_size=1,
            sampler=support_sampler,
            num_workers=num_workers,
            collate_fn=partial(multi_pipeline_collate_fn, samples_per_gpu=1),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)

        # 查询集数据加载器与支持集数据加载器其实是共用一个收集方法的collate_fn=multi_pipeline_collate_fn，
        # 使用for从DataLoader中取出的值是由collate_fn方法给出的
        # wrap two dataloaders with dataloader wrapper
        # 将查询集与支持集的数据加载器使用NWayKShotDataloader包装起来
        data_loader = NWayKShotDataloader(
            query_data_loader=query_data_loader,
            support_data_loader=support_data_loader)
    elif isinstance(dataset, TwoBranchDataset):
        from mmfewshot.utils import multi_pipeline_collate_fn
        from .dataloader_wrappers import TwoBranchDataloader

        # `TwoBranchDataset` will return a list of DataContainer
        # `multi_pipeline_collate_fn` are designed to handle
        # the data with list[list[DataContainer]]
        # initialize main dataloader
        main_data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
        # convert main dataset to auxiliary dataset
        auxiliary_dataset = copy.deepcopy(dataset)
        auxiliary_dataset.convert_main_to_auxiliary()
        # initialize auxiliary sampler and dataloader
        auxiliary_samples_per_gpu = \
            data_cfg.get('auxiliary_samples_per_gpu', samples_per_gpu)
        auxiliary_workers_per_gpu = \
            data_cfg.get('auxiliary_workers_per_gpu', workers_per_gpu)
        (auxiliary_sampler, auxiliary_batch_size,
         auxiliary_num_workers) = build_sampler(
             dist=dist,
             shuffle=shuffle,
             dataset=auxiliary_dataset,
             num_gpus=num_gpus,
             samples_per_gpu=auxiliary_samples_per_gpu,
             workers_per_gpu=auxiliary_workers_per_gpu,
             seed=seed,
             use_infinite_sampler=use_infinite_sampler)
        auxiliary_data_loader = DataLoader(
            auxiliary_dataset,
            batch_size=auxiliary_batch_size,
            sampler=auxiliary_sampler,
            num_workers=auxiliary_num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn,
                samples_per_gpu=auxiliary_samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
        # wrap two dataloaders with dataloader wrapper
        data_loader = TwoBranchDataloader(
            main_data_loader=main_data_loader,
            auxiliary_data_loader=auxiliary_data_loader,
            **kwargs)
    else:
        # 一般的数据加载器
        # DataLoader默认不会丢弃掉最后一组达不到batchsize数量的数据
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            # 各种各样的采样器
            sampler=sampler,
            # 不做要求
            num_workers=num_workers,
            # 这个收集方法与我之前看的multi_pipeline_collate_fn方法一模一样
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            # 不做要求
            worker_init_fn=init_fn,
            **kwargs)
    # 此方法返回数据加载器对象
    return data_loader


def build_sampler(
        dist: bool,
        shuffle: bool,
        dataset: Dataset,
        num_gpus: int,
        samples_per_gpu: int,
        workers_per_gpu: int,
        seed: int,
        use_infinite_sampler: bool = False) -> Tuple[Sampler, int, int]:
    """Build pytorch sampler for dataLoader.

    Args:
        dist (bool): Distributed training/test or not.
        shuffle (bool): Whether to shuffle the data at every epoch.
        dataset (Dataset): A PyTorch dataset.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        seed (int): Random seed.
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.

    Returns:
        tuple: Contains corresponding sampler and arguments

            - sampler(:obj:`sampler`) : Corresponding sampler
              used in dataloader.
            - batch_size(int): Batch size of dataloader.
            在数据加载器中加载数据的进程数。
            - num_works(int): The number of processes loading data in the
                dataloader.
    """

    # 当前进程排名与总进程数量
    rank, world_size = get_dist_info()
    # 此分支
    if dist:
        # Infinite sampler will return a infinite stream of index. But,
        # the length of infinite sampler is set to the actual length of
        # dataset, thus the length of dataloader is still determined
        # by the dataset.
        # 分支（每个epoch开始时打乱数据）
        if shuffle:
            # 在训练阶段不论是查询集还是支持集都经过此分支（无限采样器）
            if use_infinite_sampler:
                # 分布式 无限 群 采样器 对象（这其中有些复杂，祝你好运）
                # 无限其实也是有限(对于dataloader采样器的生命周期的理解)
                # 这个采样器内置了len方法，所以说虽然这个采样器叫做无限采样器，
                # 但是实际上在dataloader抽样到len之后，生命还是会结束
                # 需要注意的是的是为了保证在每次epoch开始的时候数据顺序被打乱，
                # 无限采样器使用epoch作为随机种子的一部分来保证每次抽样的序列不同
                sampler = DistributedInfiniteGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                # DistributedGroupSampler will definitely shuffle the data to
                # satisfy that images on each GPU are in the same group
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
        else:
            if use_infinite_sampler:
                sampler = DistributedInfiniteSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
            # 测试状态下看此分支
            else:
                # 普通的分布式采样器,测试数据集是不分组的
                # 有的时候我真的怀疑数据集这个参数有必要传进去吗
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
                
        # batch_size与每个gpu的进程数
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    # 模型初始化为此分支(非分布式)
    else:
        if use_infinite_sampler:
            sampler = InfiniteGroupSampler(
                dataset, samples_per_gpu, seed=seed, shuffle=shuffle)
        # 模型初始化为此分支
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) \
                if shuffle else None
        
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    return sampler, batch_size, num_workers
