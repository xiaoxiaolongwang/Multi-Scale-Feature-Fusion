# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import time
from typing import List, Optional

import mmcv
import torch
import torch.nn as nn
from mmcv.image import tensor2imgs
from mmcv.parallel import is_module_wrapper
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu, collect_results_gpu
from mmdet.utils import get_root_logger
from torch.utils.data import DataLoader


def single_gpu_test(model: nn.Module,
                    data_loader: DataLoader,
                    show: bool = False,
                    out_dir: Optional[str] = None,
                    show_score_thr: float = 0.3) -> List:
    """Test model with single gpu for meta-learning based detector.

    The model forward function requires `mode`, while in mmdet it requires
    `return_loss`. And the `encode_mask_results` is removed.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (DataLoader): Pytorch data loader.
        show (bool): Whether to show the image. Default: False.
        out_dir (str | None): The directory to write the image. Default: None.
        show_score_thr (float): Minimum score of bboxes to be shown.
            Default: 0.3.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `test` mode
            result = model(mode='test', rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            # make sure each time only one image to be shown
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                if is_module_wrapper(model):
                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
                else:
                    model.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        results.extend(result)

        prog_bar.update(batch_size)
    return results

# 多gup测试方法
def multi_gpu_test(model: nn.Module,
                   data_loader: DataLoader,
                #    None
                   tmpdir: str = None,
                #    False
                   gpu_collect: bool = False) -> List:
    """Test model with multiple gpus for meta-learning based detector.

    The model forward function requires `mode`, while in mmdet it requires
    `return_loss`. And the `encode_mask_results` is removed.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.

    Returns:
        list: The prediction results.
    """
    # 模型设置验证状态
    model.eval()
    # 结果列表
    results = []
    # 获得测试数据加载器
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    # 创建进度条对象
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    # 从数据加载器中取出数据
    # 这里需要说明的是测试阶段是没有DefaultFormatBundle这个环节的,所以说在进入数据加载器的收集方法的时候也会有不一样
    # 关于这一点,我在其中做了解释,
    # 还有就是为什么这里的data字典中的每一个值都是一个字典,这是因为MultiScaleFlipAug这个pipline会在最后对所有的数据整合一次
    # 将所有的数据放到一个值都为列表的字典中,可参考mmdet文件下的test_time_aug.py
    # data={
    #       'img_metas':[DataContainer[[dict,dict]]], 
    #       'img':[tensor]                           }
    for i, data in enumerate(data_loader):
        # 需要注意的是我们这里拿到了的img的形状其实已经变成了(c,h,w)了,我们使用img.transpose(2, 0, 1)方法交换了维度顺序
        # 而img_mate中的'img_shape'的形状其实是(h,w,c)
        with torch.no_grad():
            # forward in `test` mode
            # 以mode==‘test’模式启动模型
            # rescale=True,原来这里一直都是全程映射回原图啊....
        # 此方法返回一个嵌套列表,[[numpy arrays...],[numpy arrays...]]
        # 外列表的长度等于本批次图片的数量
        # 需要说明的是每个内列表都有15个numpy arrays,这些numpy arrays为(边界框,置信度)
        # 每一个numpy arrays都代表一张图片的一个类别的边界框信息
            result = model(mode='test', rescale=True, **data)
        # 向结果列表中添加一次内嵌套列表(注意extend方法)
        results.extend(result)

        # 进入此分支
        if rank == 0:
            # 获得本批次的大小
            batch_size = len(result)
            prog_bar.update(batch_size * world_size)

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    # 进入此分支
    else:
        # 有时候真的感觉收到了羞辱,这个方法完全不需要看的,results并没有发生变化
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_model_init(model: nn.Module, data_loader: DataLoader) -> List:
    """Forward support images for meta-learning based detector initialization.

    The function usually will be called before `single_gpu_test` in
    `QuerySupportEvalHook`. It firstly forwards support images with
    `mode=model_init` and the features will be saved in the model.
    Then it will call `:func:model_init` to process the extracted features
    of support images to finish the model initialization.

    Args:
        model (nn.Module): Model used for extracting support template features.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list[Tensor]: Extracted support template features.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    logger = get_root_logger()
    logger.info('starting model initialization...')
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `model_init` mode
            result = model(mode='model_init', **data)
        results.append(result)
        prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
    # `model_init` will process the forward features saved in model.
    if is_module_wrapper(model):
        model.module.model_init()
    else:
        model.model_init()
    logger.info('model initialization done.')

    return results


# 此方法进行多gpu的模型初始化
def multi_gpu_model_init(model: nn.Module, data_loader: DataLoader) -> List:
    """Forward support images for meta-learning based detector initialization.

    The function usually will be called before `single_gpu_test` in
    `QuerySupportEvalHook`. It firstly forwards support images with
    `mode=model_init` and the features will be saved in the model.
    Then it will call `:func:model_init` to process the extracted features
    of support images to finish the model initialization.

    Noted that the `data_loader` should NOT use distributed sampler, all the
    models in different gpus should be initialized with same images.

    Args:
        model (nn.Module): Model used for extracting support template features.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list[Tensor]: Extracted support template features.
    """
    # 模型进入验证(测试)模式
    model.eval()
    # 结果列表
    results = []
    # 从数据加载器中获得数据集
    dataset = data_loader.dataset
    rank, _ = get_dist_info()
    if rank == 0:
        # 需要注意的是此处的日志器是来自mmdet的,而不是来自mmfewshot的,所以说不会写入文件
        logger = get_root_logger()
        # 信息
        logger.info('starting model initialization...')
        # 创建进度条对象
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    # the model_init dataloader do not use distributed sampler to make sure
    # all of the gpus get the same initialization
    # 从数据初始化数据加载器中取出数据
    # 模型初始化数据加载器取出来的的data是什么呢?
    # data是一个字典{
    #               'img_metas':DataContainer[[dict,dict...]], 
    #               'img':DataContainer[tensor], 
    #               'gt_bboxes':DataContainer[[tensor,tensor...]], 
    #               'gt_labels':DataContainer[[tensor,tensor...]]
    #                                                          }
    for i, data in enumerate(data_loader):
        # 模型无梯度
        with torch.no_grad():
            # forward in `model_init` mode
            # 每个batch返回这样一个字典{'gt_labels': gt_labels, 'roi_feat': roi_feat}
            result = model(mode='model_init', **data)
        # 向结果列表中添加batch前向传播字典
        results.append(result)
        # 关于进度条的部分先不做要求
        if rank == 0:
            prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
    # model_init function will process the forward features saved in model.
    # 这里的意思是这个模型是否是被一个装饰器修饰的,具体的意思指的是model是一个类的属性或者model是单独的继承自nn.module的一个类
    if is_module_wrapper(model):
        # model_init()方法利用特征图和标签计算出每个类别的原型用以测试阶段(原型字典保存在model中)
        model.module.model_init()
    else:
        model.model_init()
    if rank == 0:
        # 模型初始化完毕
        logger.info('model initialization done.')
    # 返回结果列表[{},{},{},...]
    return results
