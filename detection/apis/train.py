# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv.utils import ConfigDict, build_from_cfg
from mmdet.core import DistEvalHook, EvalHook

from mmfewshot.detection.core import (QuerySupportDistEvalHook,
                                      QuerySupportEvalHook)
from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)
from mmfewshot.utils import compat_cfg, get_root_logger


def train_detector(model: nn.Module,
                   dataset: Iterable,
                   cfg: ConfigDict,
                   distributed: bool = False,
                #    True
                   validate: bool = False,
                   timestamp: Optional[str] = None,
                   meta: Optional[Dict] = None) -> None:
    # 兼容配置参数（主要是数据集加载器）
    # cfg.data.train_dataloader['samples_per_gpu'] = samples_per_gpu
    # cfg.data.train_dataloader['workers_per_gpu'] = workers_per_gpu
    # cfg.data.val_dataloader['workers_per_gpu'] = workers_per_gpu
    # cfg.data.test_dataloader['workers_per_gpu'] = workers_per_gpu
    cfg = compat_cfg(cfg)
    # 激活日志对象（先前已经定义过，此处重复使用）
    # 使用配置文件中定义的日志级别，log_level = 'INFO'
    logger = get_root_logger(log_level=cfg.log_level)

    # 数据加载器部分
    # prepare data loaders
    # 处理dataset不是元组或者是列表的情况
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # 训练数据加载器的默认参数配置
    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # 分布式训练下`num_gpus`参数会被忽略
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        # 是否分布式:是
        dist=distributed,
        # 随机种子:42
        seed=cfg.seed,
        # 数据的配置参数
        data_cfg=copy.deepcopy(cfg.data),
        # 是否使用无限采样器:是
        use_infinite_sampler=cfg.use_infinite_sampler,
        # 持续工作
        persistent_workers=False)
    
    # 训练 数据加载器参数（如果两个字典中有重复的键，后边的会将前边的覆盖）
    train_loader_cfg = { 
        # 解包训练数据加载器的默认参数配置
        **train_dataloader_default_args,
        # 解包cfg配置中的训练加载器部分(在此处修稿默认配置中的samples_per_gpu,workers_per_gpu字段)
        **cfg.data.get('train_dataloader', {})
    }
    # 列表推导式，从数据集列表中取出数据
    # （这个数据其实是一个数据包装器对象，继承自NWayKShotDataset，内部有查询数据与支持数据）
    # 将数据包装对象与训练加载器参数一并传入构建数据加载器方法

    # 在本例中，build_dataloader方法返回一个NWayKShotDataloader对象，内有查询数据加载器与支持数据加载器(训练)
    # 训练阶段的查询集和支持集的数据加载器均为分布式,打乱,无限采样器
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # 这段先不用管，只需要知道它是为了启用分布式训练，并且主要是配置相关的设置就足够了
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        # 此时model是一个被MMDistributedDataParallel包装的对象，现在model.module调用的是原来的model
    else:
        # Please use MMCV >= 1.4.4 for CPU training!
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)


    # build runner(优化器建立在模型的参数上)
    # 实话说这段代码跟optimizer_SGD = optim.SGD(wang.parameters(), learning_rate)其实没有什么区别
    # /root/miniconda/envs/openmmlab/lib/python3.7/site-packages/torch/optim/sgd.py
    optimizer = build_optimizer(model, cfg.optimizer)

    # Infinite sampler will return a infinite stream of index. It can NOT
    # be used in `EpochBasedRunner`, because the `EpochBasedRunner` will
    # enumerate the dataloader forever. Thus, `InfiniteEpochBasedRunner`
    # is designed to handle dataloader with infinite sampler.
    # 跳过此分支
    if cfg.use_infinite_sampler and cfg.runner['type'] == 'EpochBasedRunner':
        cfg.runner['type'] = 'InfiniteEpochBasedRunner'
    
    # 在本例中，返回一个继承自IterBasedRunner(BaseRunner)的对象
    # BaseRunner是一个抽象方法，继承其的子类应该重写save_checkpoint,run,val,train四个方法
    # /root/miniconda/envs/openmmlab/lib/python3.7/site-packages/mmcv/runner/base_runner.py
    # /root/miniconda/envs/openmmlab/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py
    runner = build_runner(
        # dict(type='IterBasedRunner', max_iters=18000)
        cfg.runner,
        default_args=dict(
            # 模型和优化器
            model=model,
            optimizer=optimizer,
            # 项目工作目录
            # './work_dirs/meta-rcnn_r101_c4_8xb4_voc-split1_base-training'
            work_dir=cfg.work_dir,
            # 日志对象
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    # 给runner对象绑定timestamp字段(此字段为log文件名)
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    # 跳过此分支
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    # 此分支
    elif distributed and 'type' not in cfg.optimizer_config:
        # 优化器的配置为一个优化器钩子对象,该对象接受 grad_clip=None作为参数
        # 该类继承自Hook钩子类，内部实现了after_train_iter方法
        # 防止由于梯度过大或者是过小导致的问题，但是在本例中，其实梯度裁剪并不会用到
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    # 给Runner对象注册训练时的一系列钩子
    # 学习率控制配置(学习率规划器),优化器配置对象(控制在反向传播的时候是否进行梯度裁剪并负责反向传播),
    # 检查点配置(多少iter的时候保存一次模型参数),日志配置(在多少iter的时候输出信息，如何输出信息(列表)),
    # momentum_config为None
    # 此时未传入自定义custom_hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    # 跳过此步
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    # 香槟开早了,这部分是需要看的,准备验证的hooks
    if validate:
        # currently only support single images testing
        # 验证数据加载器的默认配置
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            # 分布式
            dist=distributed,
            # 不打乱数据
            shuffle=False,
            # 不持续工作
            persistent_workers=False)
        val_dataloader_args = {
            **val_dataloader_default_args,
            # 此部分只更新线程信息
            **cfg.data.get('val_dataloader', {})
        }
        # 构建验证数据集,说是验证数据集，其实就是测试阶段,
        # (但是我不知道为什么测试阶段是没有加载标签的)
        # 看完代码来解答这个问题的话,其实数据集的构建类中实现了val方法,而在数据集的构建类中一定会载入标签的,
        # 我们只需要在模型推理完成一遍之后将结果调用数据集的val方法就可以得到测试结果了,因为我们可以利用数据集的
        # 注释信息而不是使用数据加载器的注释信息
        # (默认参数为dict(test_mode=True),在build_from_cfg方法中,默认字典参数中的键值对被整合到cfg.data.val中)
        # type='FewShotVOCDataset'
        # 此类对象的__getitem_方法在父类CustomDataset中实现
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        assert val_dataloader_args['samples_per_gpu'] == 1, \
            'currently only support single images testing'
        # 构建测试阶段的数据加载器,内部使用不打乱非群组有限加载器
        # 详情请看源码
        # 测试阶段数据加载器为分布式,不打乱,非无限采样器
        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

        # 获取验证阶段的配置(这与保存检查点的时机一致)
        # dict(interval=6000, metric='mAP')
        eval_cfg = cfg.get('evaluation', {})
        # 给验证配置添加'by_epoch'字段,用以确定验证时机是否以epoch为单位
        # dict(interval=6000, metric='mAP',by_epoch=False)
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'

        # Prepare `model_init` dataset for model initialization. In most cases,
        # the `model_init` dataset contains support images and few shot
        # annotations. The meta-learning based detectors will extract the
        # features from images and save them as part of model parameters.
        # The `model_init` dataset can be mutually configured or
        # randomly selected during runtime.

        # 模型的初始化部分
        if cfg.data.get('model_init', None) is not None:
            # The randomly selected few shot support during runtime can not be
            # configured offline. In such case, the copy datasets are designed
            # to directly copy the randomly generated support set for model
            # initialization. The copy datasets copy the `data_infos` by
            # passing it as argument and other arguments can be different
            # from training dataset.

            # 从模型初始化配置部分获取copy_from_train_dataset字段的值(剪切)
            # True
            if cfg.data.model_init.pop('copy_from_train_dataset', False):
                # 当我们从训练数据集中获取模型初始化的数据的时候,就不会从本地加载数据(一个警告)
                if cfg.data.model_init.ann_cfg is not None:
                    warnings.warn(
                        'model_init dataset will copy support '
                        'dataset used for training and original '
                        'ann_cfg will be discarded', UserWarning)
                # modify dataset type to support copying data_infos operation
                # 修改数据集类型以支持复制data_info操作
                # 获取模型初始化配置的类型FewShotVOCCopyDataset
                cfg.data.model_init.type = \
                    get_copy_dataset_type(cfg.data.model_init.type)
                # 仔细看好这个方法是获得从数据集复制的数据类型,而不是普通的数据类型
                
                # 确保第一个数据集中有get_support_data_infos这个属性或者方法
                if not hasattr(dataset[0], 'get_support_data_infos'):
                    raise NotImplementedError(
                        f'`get_support_data_infos` is not implemented '
                        f'in {dataset[0].__class__.__name__}.')
                # 向模型初始化配置中添加支持集中所有图片(注释信息只有一个物体对象)的信息ann_cfg=[dict(data_infos=[{},{},{}...])]
                # 使用ann_cfg作为键
                # 注意是数据包装器的self.data_infos_by_class(在本例中这本身就是实例级别的数据)
                cfg.data.model_init.ann_cfg = [
                    dict(data_infos=dataset[0].get_support_data_infos())
                ]
            # The `model_init` dataset will be saved into checkpoint, which
            # allows model to be initialized with these data as default, if
            # the config of data is not be overwritten during testing.
        # 在train脚本中对字典做了以下改变
        # cfg.checkpoint_config.meta = dict(
        # mmfewshot_version=__version__ + get_git_hash()[:7],CLASSES=datasets[0].CLASSES)
            
            # 向cfg.checkpoint_config.meta中添加model_init_ann_cfg字段,值为我们上边获取的支持集中所有图片的信息
            cfg.checkpoint_config.meta['model_init_ann_cfg'] = \
                cfg.data.model_init.ann_cfg
            # 以上两行代码获取支持集中所有图片(注释信息只有一个物体对象)的信息,并将这些信息添加到
            # cfg.checkpoint_config.meta['model_init_ann_cfg']与cfg.data.model_init.ann_cfg中

            # 剪切模型初始化部分配置的这两个字段(训练,测试,验证配置内部没有这两个字段)
            # 16
            samples_per_gpu = cfg.data.model_init.pop('samples_per_gpu', 1)
            # 1
            workers_per_gpu = cfg.data.model_init.pop('workers_per_gpu', 1)

            # 构建模型初始化数据集(train)
            # 注意此时初始化数据集的type为FewShotVOCCopyDataset而不是FewShotVOCCopyDataset
            # 注意这个instance_wise=True,虽然说我们获得的数据本身就是实例级别的数据
            # (这是来自数据包装器的self.data_infos_by_class的数据)
            # 模型初始化的数据预处理tranformer跟训练阶段的支持集部分是完全一样的
            model_init_dataset = build_dataset(cfg.data.model_init)

            # Noted that `dist` should be FALSE to make all the models on
            # different gpus get same data results in same initialized models.
            # 不使用分布式确保在不同的gpu上使用相同的初始化模型得到相同的数据结果
            # 模型初始化数据加载器为非分布式,不打乱,非无限加载器
            model_init_dataloader = build_dataloader(
                model_init_dataset,
                # 16
                samples_per_gpu=samples_per_gpu,
                # 1
                workers_per_gpu=workers_per_gpu,
                dist=False,
                shuffle=False)

            # eval hook for meta-learning based query-support detector, it
            # supports model initialization before regular evaluation.
            # 这是一个为基于查询支持的元学习检测器准备的验证钩子,他支持在常规验证之前进行模型初始化
            # 验证钩子的类型(查询支持分布式验证钩子)
            eval_hook = QuerySupportDistEvalHook \
                if distributed else QuerySupportEvalHook
            
            # eval_hook(model_init_dataloader, val_dataloader, **eval_cfg)
            # 上边这段代码的意思是使用模型初始化数据加载器与模型验证数据加载器构建一个QuerySupportDistEvalHook对象
            # 我已经基本看完,这真是一个浩大的工程啊

            # 向runner中注册验证钩子
            runner.register_hook(
                eval_hook(model_init_dataloader, val_dataloader, **eval_cfg),
                priority='LOW')
        # 先跳过此分支
        else:
            # for the fine-tuned based methods, the evaluation is the
            # same as mmdet.
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    # 进入此分支
    if cfg.get('custom_hooks', None):
        # 获得自定义钩子列表
        # [dict(type='NumClassCheckHook')]
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f'custom_hooks expect list type, but got {type(custom_hooks)}'
        # 遍历自定义钩子列表
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(
                hook_cfg, dict
            ), f'Each item in custom_hooks expects dict type, but ' \
               f'got {type(hook_cfg)}'
            
            hook_cfg = hook_cfg.copy()
            # 获得优先级,否则为NORMAL
            priority = hook_cfg.pop('priority', 'NORMAL')
            # 构建自定义钩子对象
            # NumClassCheckHook在mmdet/datasets/utils.py中实现
            # 不太懂,好像就只是用来检查类别是否一致的,不用关注也可
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # 关于检查点与回复点
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    # 我个人感觉使用下边这里就没有什么问题
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    
    # 兄弟们,开跑!!!!!!!!!!!!
    runner.run(data_loaders, cfg.workflow)
