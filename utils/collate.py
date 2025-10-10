# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from mmcv.parallel.data_container import DataContainer
from torch.utils.data.dataloader import default_collate

# 首先说明这是一个递归方法

# batch参数是一个装有batch_size个字典的列表，其中每个字典经过数据预处理部分的名为Collect的transformer整理过后
# 内有四个键-值对，四个键分别是'img_meta','img','gt_bboxes','gt_labels'，他们的值都被DataContainer装载

# 这个方法要做的事情是，从列表取出这些字典中相同键的值（DataContainer对象），取出每个DataContainer对象的data字段
# 将这些data放到一个列表中，然后将这个列表使用DataContainer装起来
# 还记得我们之前说的我们取得是相同键的值吗，创建一个新字典，就使用这个(键:整合之后的列表)
# 不断执行这个操作，直到batch[0]中的所有键都被遍历完成

# 还有一个需要注意的点是，我们上边说了将相同键的DataContainer对象放到一个列表中的这个操作吗？
# 对，执行这个操作是要分情况的，我们在利用Collect收集数据的时候是不是使用了DataContainer包装器，
# 在使用这个包装器的时候，有两个比较重要的字段，那就是'stack'和'cpu_only'，这是两个布尔值

# 该方法的返回值是基于这个列表的    
# stacked = []，我的意思是DataContainer对象的data字段会是stacked

# 现在分情况讨论，值得一提的是 先判断'cpu_only'字段，然后是'stack'字段，最后是两个字段都为false的情况

# 分支1:首先是'cpu_only'为True的时候，这里比较简单的从每个DataContainer对象中取出data字段，装到一个列表里，
# 然后再将这个列表放到stacked中(对的，就是这样，最后的stacked的结构会是这样的[[{...},{...}]])
# 在这个分支中的返回值是这样的，return被写在这个分支中
# return DataContainer(
#                 stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)

# 分支2:然后是'stack'为True的时候，此时要求将DataContainer对象中的data字段堆叠在一起，就比如说是两张图片吧，
# 图片的形状是[3,500,600],[3,600,700](这里为什么是C,H,W的形式呢，因为在'DefaultFormatBundle'的transformer中进行了通道转换)
# 我们会将这两张图片整合成一个[2,3,600,700]的tensor
# (此处要求两张图片的宽与高相等，所以图片会在高与宽的维度上填充，填充的像素数在此方法被计算，
# 填充值由batch[0]（DataContainer对象）的padding_value字段指定（图像的填充值为0）)
# default_collate方法接受一个含有n个形状为[C,H,W]的tensor的列表，将这个列表转换为一个形状为[n,C,H,W]的tensor(输入列表，返回tensor)
# 所以说最后stacked的结构为[tensor]

# 分支3:最后是'cpu_only'与'stack'均为False的时候，对于stacked的操作与'cpu_only'为True的时候是一模一样的，不再赘述

# 对于分支2与分支3来说，return被这样写在整个判断循环的外部
# return DataContainer(stacked, batch[0].stack, batch[0].padding_value)

# 最后该方法返回一个字典,字典的键被batch[0]的键指定，每个键的值被 batch中对应键的所有值的整合结果指定


def multi_pipeline_collate_fn(batch, samples_per_gpu: int = 1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size. This is designed to support the case that the
    :func:`__getitem__` of dataset return more than one images, such as
    query_support dataloader. The main difference with the :func:`collate_fn`
    in mmcv is it can process list[list[DataContainer]].

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases:

    1. cpu_only = True, e.g., meta data.
    2. cpu_only = False, stack = True, e.g., images tensors.
    3. cpu_only = False, stack = False, e.g., gt bboxes.

    Args:
    很奇怪，我记得get方法返回的不是字典吗，问什么这里说是列表
        batch (list[list[:obj:`mmcv.parallel.DataContainer`]] |
            list[:obj:`mmcv.parallel.DataContainer`]): Data of
            single batch.
        samples_per_gpu (int): The number of samples of single GPU.
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    # This is usually a case in query_support dataloader, which
    # the :func:`__getitem__` of dataset return more than one images.
    # Here we process the support batch data in type of
    # List: [ List: [ DataContainer]]
    if isinstance(batch[0], Sequence):
        samples_per_gpu = len(batch[0]) * samples_per_gpu
        # 此段代码的作用是展平batch一层列表
        # 为什么会这样,dataloader在采样并获取到数据后一定会给数据外部加一层列表,不论本次取到了多少个元素
        # 比如说batch结构为[[dict,dict,dict]],这段代码过后的结果是[dict,dict,dict]
        batch = sum(batch, [])
    
    # 这个分支，如果传入的batch[0]是DataContainer对象，在本例中为主要使用分支
    if isinstance(batch[0], DataContainer):
        # 构建堆叠列表（方法的主要返回值）
        stacked = []
        # 如果参数batch[0]的cpu_only为True（这通常用在img_metas上）
        if batch[0].cpu_only:
            # 这个循环往往只会被执行一次
            for i in range(0, len(batch), samples_per_gpu):
                # 一个列表推导式（看起来像是给列表添加一个列表，最后的结果会是这样的双列表包裹的字典[[{...},{...}]]）
                # 这个列表推导式从batch中取出DataContainer元素，获取DataContainer对象的data属性（字典）
                # 构成一个列表，添加到stacked这个列表中
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            # 这个分支返回一个DataContainer对象，data字段为stacked，传入剩余三个字段
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        
        
        # 如果参数batch[0]的stack为True（这通常用在img上）（这里的原理会比较复杂，可看可不看，需要知道的是最后stacked为[tensor]）
        elif batch[0].stack:
            # 此处应该只循环一次，因为len(batch)==samples_per_gpu:1次
            for i in range(0, len(batch), samples_per_gpu):
                # 如果要堆叠，确保列表中DataContainer元素的data是torch.Tensor类型
                assert isinstance(batch[i].data, torch.Tensor)

                # 看这个分支(填充维度字段不为空)
                # batch[i]其实是一个被DataContainer类包装的对象
                if batch[i].pad_dims is not None:
                    # 首先获取第一个数据的维度数量（如果是图片维度数量为3）
                    ndim = batch[i].dim()
                    # 确保第一个数据的维度数量不大于该数据的填充维度数量（这里的填充维度数量指的是图像的最后的n个维度）
                    assert ndim > batch[i].pad_dims
                    # 创建一个全0数组，数组的长度为第一个数据的填充维度数量:2
                    max_shape = [0 for _ in range(batch[i].pad_dims)]

                    # 循环，索引为[1,第一个数据填充维度数量+1)(总循环次数为第一个数据填充维度数量):2次
                    for dim in range(1, batch[i].pad_dims + 1):
                        # 此处做了什么？将batch[i]的倒数第一个维度的大小赋值给max_shape[0]
                        #              将batch[i]的倒数第二个维度的大小赋值给max_shape[1]
                        max_shape[dim - 1] = batch[i].size(-dim)

                    # 循环，索引为[0,batch_size)(总循环次数为batch_size):2次
                    # 选取出batch中的元素
                    for sample in batch[i:i + samples_per_gpu]:
                        # 循环次数为1次
                        for dim in range(0, ndim - batch[i].pad_dims):
                            # 确保batch中的数据 除了要保持一致大小的维度之外的所有维度的大小都相等
                            # （此处的实现为确保所有数据的剩余维度大小都与第一个数据相对应的维度大小相等）
                            assert batch[i].size(dim) == sample.size(dim)
                        # 这个数组最终装的是什么呢？检查batch中所有img数据的倒数第一个维度的大小W，选取其中的最大值，赋给max_shape[0]
                        #                         检查batch中所有img数据的倒数第二个维度的大小H，选取其中的最大值，赋给max_shape[1]
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    # 创建放置填充之后的的img的列表
                    padded_samples = []
                    # 取出batch中的元素
                    for sample in batch[i:i + samples_per_gpu]:
                        # 创建一个全0列表，内含四个元素(关于这里为什么会是四个元素，
                        # F.pad方法在传入参数时需要传入一个长度为填充信息列表，分别对应(左，右，上，下)四个位置的填充量)
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        # 循环[1,3)，此循环计算出本样本在右方位与下方位所需要的像素填充数量
                        for dim in range(1, batch[i].pad_dims + 1):
                            # 只在图片的右边1与下边3填充，我们之前说过max_shape[0]为W，max_shape[1]为H
                            # 而在右边1填充需要计算本样本的宽sample.size(-1)与batch中最大的宽max_shape[0]的差值
                            # 在下边3填充需要计算本样本的高sample.size(-2)与batch中最大的高max_shape[1]的差值
                            pad[2 * dim -1] = max_shape[dim - 1] - sample.size(-dim)
                        # F.pad方法接受三个参数(tensor,pad填充信息,填充值),返回填充后的tensor
                        # 向padded_samples列表中添加 填充后的的img
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    # 在本例中，default_collate方法接受一个装有n个相同形状[C,H,W]tensor的列表，返回一个形状为[n,C,H,W]的tensor
                    # 向stacked中添加堆叠在一起的图像tensor
                    stacked.append(default_collate(padded_samples))

                
                # 不看这个分支
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        # 返回DataContainer对象DataContainer[tensor]
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [
            multi_pipeline_collate_fn(samples, samples_per_gpu)
            for samples in transposed
        ]
    # 首先本配置文件的batch[0]是字典，所以会进入这个分支
    elif isinstance(batch[0], Mapping):
        # 这是一个比较复杂的列表推导式
        return {
            # 然后利用取出的key作为新字典的键，关于值，首先从batch中循环取字典d，然后根据key取出字典的d[key]，
            # 将batch中所有字典的key对应的值组合成一个列表，作为multi_pipeline_collate_fn方法的第一参数
            # 使用samples_per_gpu作为multi_pipeline_collate_fn方法的第二参数，得到方法的返回值作为新字典键key的值
            # 重复这个步骤,直到batch[0]中的所有键被遍历完毕
            key: multi_pipeline_collate_fn([d[key] for d in batch],
                                           samples_per_gpu)
            # 首先从字典batch[0]中取出键key
            for key in batch[0]
        }
        # 最后的返回值是一个字典{
        #                       'img_metas':DataContainer[[dict,dict]], 
        #                     'img':DataContainer[tensor], 
        #                     'gt_bboxes':DataContainer[[tensor,tensor]], 
        #                     'gt_labels':DataContainer[[tensor,tensor]]
        #                                                                   }
    else:
        return default_collate(batch)
