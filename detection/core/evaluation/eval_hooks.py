# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import Runner
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader


class QuerySupportEvalHook(BaseEvalHook):
    """Evaluation hook for query support data pipeline.

    This hook will first traverse `model_init_dataloader` to extract support
    features for model initialization and then evaluate the data from
    `val_dataloader`.

    Args:
        model_init_dataloader (DataLoader): A PyTorch dataloader of
            `model_init` dataset.
        val_dataloader (DataLoader): A PyTorch dataloader of dataset to be
            evaluated.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, model_init_dataloader: DataLoader,
                 val_dataloader: DataLoader, **eval_kwargs) -> None:
        super().__init__(val_dataloader, **eval_kwargs)
        self.model_init_dataloader = model_init_dataloader

    def _do_evaluate(self, runner: Runner) -> None:
        """perform evaluation and save checkpoint."""
        if not self._should_evaluate(runner):
            return
        # In meta-learning based detectors, model forward use `mode` to
        # identified the 'train', 'val' and 'model_init' stages instead
        # of `return_loss` in mmdet. Thus, `single_gpu_test` should be
        # imported from mmfewshot.
        from mmfewshot.detection.apis import (single_gpu_model_init,
                                              single_gpu_test)

        # `single_gpu_model_init` extracts features from
        # `model_init_dataloader` for model initialization with single gpu.
        single_gpu_model_init(runner.model, self.model_init_dataloader)
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class QuerySupportDistEvalHook(BaseDistEvalHook):
    """Distributed evaluation hook for query support data pipeline.

    This hook will first traverse `model_init_dataloader` to extract support
    features for model initialization and then evaluate the data from
    `val_dataloader`.

    Noted that `model_init_dataloader` should NOT use distributed sampler
    to make all the models on different gpus get same data results
    in same initialized models.

    Args:
        model_init_dataloader (DataLoader): A PyTorch dataloader of
            `model_init` dataset.
        val_dataloader (DataLoader): A PyTorch dataloader of dataset to be
            evaluated.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """
    # eval_kwargs = dict(interval=6000, metric='mAP',by_epoch=False)
    def __init__(self, model_init_dataloader: DataLoader,
                 val_dataloader: DataLoader, **eval_kwargs) -> None:
        super().__init__(val_dataloader, **eval_kwargs)
        # 模型初始化数据加载器
        self.model_init_dataloader = model_init_dataloader

    # 做测试的方法(重要)
    def _do_evaluate(self, runner: Runner) -> None:
        """perform evaluation and save checkpoint."""

        if not self._should_evaluate(runner):
            return
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        # 这部分不是很懂,先不用看,还是先关注原理的部分吧
        # True
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        # None
        tmpdir = self.tmpdir
        # 创建临时文件夹
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # In meta-learning based detectors, model forward use `mode` to
        # identified the 'train', 'val' and 'model_init' stages instead
        # of `return_loss` in mmdet. Thus, `multi_gpu_test` should be
        # imported from mmfewshot.
        # 从mmfewshot引入多gpu测试与模型初始化方法
        from mmfewshot.detection.apis import (multi_gpu_model_init,
                                              multi_gpu_test)

        # Noted that `model_init_dataloader` should NOT use distributed
        # sampler to make all the models on different gpus get same data
        # results in the same initialized models.
        # 此方法内部构建了一个进度条对象,向model中设置了装有支持集每个类别的原型的字典
        # 用以后续的测试阶段
        # 使用模型初始化数据加载器
        # 此方法有返回结果,但是此处未接收
        multi_gpu_model_init(runner.model, self.model_init_dataloader)

    # 此方法返回一个嵌套列表,[[numpy arrays...],[numpy arrays...]]
    # 外列表的长度等于本批次图片的数量
    # 需要说明的是每个内列表都有15个numpy arrays,这些numpy arrays为(边界框,置信度)
    # 每一个numpy arrays都代表一张图片的一个类别的边界框信息
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            # None
            tmpdir=tmpdir,
            # False
            gpu_collect=self.gpu_collect)
        
        if runner.rank == 0:
            sys.stdout.write('\n')
            # 更新日志缓冲输出
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            # 调用验证方法
            # 此方法在evaluation.py中实现
            # 返回值为None
            key_score = self.evaluate(runner, results)
            
            # None
            if self.save_best:
                self._save_ckpt(runner, key_score)
