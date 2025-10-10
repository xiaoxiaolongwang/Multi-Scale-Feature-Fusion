# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import cv2
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmdet.utils import collect_env

import mmfewshot  # noqa: F401, F403
from mmfewshot import __version__
from mmfewshot.detection.apis import train_detector
from mmfewshot.detection.datasets import build_dataset
from mmfewshot.detection.models import build_detector
from mmfewshot.utils import get_root_logger
# import pprint

# disable multithreading to avoid system being overloaded
cv2.setNumThreads(0)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a FewShot model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir', help='the directory to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        # 使用这个参数即为True，不使用则为False
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    # 思考一下有咩有可能GUPS压根就不是给你传的参数，而是给启动器的
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        # 看这个参数没用啊，因为后边说了非分布式训练我们只使用单gpu
        '--gpus',
        # 表示参数的值会被转换为整数
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        # 此参数已经被弃用
        '--gpu-ids',
        type=int,
        # 指定了命令行中应该提供多少个值，‘+’为提供任意个值--gpu-ids 0 1 2
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        # 现在最有用的参数，其实就是单GPU
        '--gpu-id',
        type=int,
        # 创建参数表即为0，不需要调用--gpu-id
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        # 使用了自定义的方法
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value '
        'to be overwritten is a list, it should be like key="[a,b]" or '
        'key=a,b It also allows nested list/tuple values, e.g. '
        'key="[(a,b),(c,d)]" Note that the quotation marks are necessary '
        'and that no white space is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    
    # 实例化参数表
    args = parser.parse_args()
    
    # 如果有需要添加到环境变量的值，就在这里
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


# 重要的注释前无制表符
def main():
# configs  configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py
# --launcher pytorch
    args = parse_args()
    
# arg.config是arg的一个属性，内为字符串型的地址
# 方法返回值为一种字典子类对象
    cfg = Config.fromfile(args.config)
    
    # import pdb; pdb.set_trace()
    
    # 不用管
    # 此处为空
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    # 如果参数表需要引入新包，在这里
    # 此处为空
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
        
    # set cudnn_benchmark
    # 不懂不需要看吧
    # 此处为空
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    
# 创建工作目录
# work_dir is determined in this priority: CLI > segment in file > filename
# 如果命令行传入了工作目录则使用他
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
        
    # 如果命令行没有传入工作目录，预定义配置也没有这个配置的话
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        # 使用根目录下的工作目录work_dirs以及 配置文件名（去掉.py）加起来
        # 并将其作为cfg的work_dir属性
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
        
    # 命令行传入参数中溯回点文件不为空
    # 此处为空
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        
    # 如果后传参gpus不为空
    # 此处为空
    if args.gpus is not None:
        # 此时返回一个可迭代对象
        cfg.gpu_ids = range(1)
        # 在非分布式训练中只使用单gpu
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
        
    if args.gpu_ids is not None:
        # 我么只要传入gpuids的第一个值
        cfg.gpu_ids = args.gpu_ids[0:1]
        # 使用gpu列表的第一个值
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
# 如果以上两个弃用的参数都为空，使用最好的参数
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
    
    # 这部分不用管的
    # init distributed env first, since logger depends on the dist info.
    # 启动器为空，不使用分布式训练
    if args.launcher == 'none':
        distributed = False
        rank = 0
    # 启动分布式训练
    else:
        # 分布式训练启动
        distributed = True
        # 初始化分布式训练
        init_dist(args.launcher, **cfg.dist_params)
        # 得到分布式训练参数
        rank, world_size = get_dist_info()
        # re-set gpu_ids with distributed training mode
        cfg.gpu_ids = range(world_size)
        
        
# create work_dir
# 确保工作目录存在，否则创建它
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# dump config
# 将cfg中的所有内容写入到work_dir中的一个.py文件中
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    
    # init the logger before other steps
# 获取时间
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
# 定义日志文件的地址和文件名的字符串
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
# 初始化日志对象（日志地址和日志的级别（超过这个重要性的信息会被记录））（日志路径与级别）
# 使用配置文件中定义的日志级别，log_level = 'INFO'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # 记录一些重要信息的字典，这将会被日志记录
    meta = dict()
    
    
    # log env info
# 这个变量装是有环境信息的字典
    env_info_dict = collect_env()
# 从字典中取出键值，返回一个列表对象
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
# 分割线字符串
    dash_line = '-' * 60 + '\n'
    
# 向日志文件写入环境信息
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    
    
# 向字典写入环境信息和配置字典的规范性文本
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    
    
    # log some basic info
# 向日志文件中加入分布式信息和配置字典的规范性文本
    logger.info(f'Distributed training: {distributed}')
    # 后续其实还有对于cfg的修改（代码第326行）
    logger.info(f'Config:\n{cfg.pretty_text}')

    
    # set random seeds
    if args.seed is not None:
        seed = args.seed
    # 此分支:42
    elif cfg.seed is not None:
        seed = cfg.seed
    elif distributed:
        seed = 0
        Warning(f'When using DistributedDataParallel, each rank will '
                f'initialize different random seed. It will cause different'
                f'random action for each rank. In few shot setting, novel '
                f'shots may be generated by random sampling. If all rank do '
                f'not use same seed, each rank will sample different data.'
                f'It will cause UNFAIR data usage. Therefore, seed is set '
                f'to {seed} for default.')
    else:
        seed = None

    if seed is not None:
# 向日志文件写入随机种子信息
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
        # 设置随机种子以及我不知道的一个参数（确定性，不懂）
        set_random_seed(seed, deterministic=args.deterministic)
        
        
# 向meta中写入随机种子和配置文件的名字
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    
    # build_detector will do three things, including building model,
    # initializing weights and freezing parameters (optional).
# 建立模型，初始化模型，冻结参数
# 另外就是写入log文件
    # import pdb; pdb.set_trace()
    model = build_detector(cfg.model, logger=logger)
    
    
    # build_dataset will do two things, including building dataset
    # and saving dataset into json file (optional).
# 构建训练数据集（后边可能还有测试步骤，所以是一个列表）
    datasets = [
        # 训练数据集包装器（包括查询集和支持集）
        build_dataset(
            # 数据的训练参数
            # 其实最后的支持集和查询集的配置除了是否使用难例与类别，其余都是一样的
            cfg.data.train,
            
            rank=rank,
            # 工作目录地址
            work_dir=cfg.work_dir,
            # 内部有使用到这个参数，只是因为不保存数据集而被跳过了使用的那个判断分支
            timestamp=timestamp)
    ]

    # 此处先跳过因为工作流列表的长为1
    # 如果工作状态为两个（训练和测试）
    if len(cfg.workflow) == 2:
        # 深拷贝验证数据设置
        val_dataset = copy.deepcopy(cfg.data.val)
        # 将预处理改为测试的预处理
        val_dataset.pipeline = cfg.data.train.pipeline
        # 向数据集列表中加入验证数据集对象（一般不用，毕竟现在谁还验证）
        datasets.append(build_dataset(val_dataset))
        
    # 若配置文件检查点配置不为空，向配置文件的检查点配置中写一个meta字典，内含mmfewshot版本和基类训练所需的类别名
    # （但我实在是搞不懂凭什么这个数据集能获得CLASSES这个属性）（没关系现在我懂了）
    if cfg.checkpoint_config is not None:
        # save mmfewshot version, config file content and class names in
        # checkpoints as meta data
        # 此处对于cfg修改
        cfg.checkpoint_config.meta = dict(
            # 这里先不看了，如果以后有机会再看
            mmfewshot_version=__version__ + get_git_hash()[:7],
            # 装饰器的基训练类别元组
            CLASSES=datasets[0].CLASSES)
        
        
    # import pdb; pdb.set_trace()
    # add an attribute for visualization convenience
    # 给模型添加基训练类别元组字段
    model.CLASSES = datasets[0].CLASSES
    
    
    train_detector(
        # 模型
        model,
        # 数据集列表
        datasets,
        # 配置字典
        cfg,
        # 分布式训练:True
        distributed=distributed,
        # 这个值在parse_args()中被设置，我们并没有使用到，所以这里的no_validate应该为False
        # 而我们在前边加了not所以传入的参数其实是True
        validate=(not args.no_validate),
        # 又传入相同时间？（感觉内部完全没有用到这个参数）(这是为了确保训练的信息被保存在与log文件同名的.log.json文件)
        # (该方法内部成这个是一个丑陋的解决方案，哈哈哈)
        timestamp=timestamp,
        meta=meta)
        # # 向字典写入环境信息和配置字典的规范性文本
        # meta['env_info'] = env_info
        # meta['config'] = cfg.pretty_text
        # # 向meta中写入随机种子和配置文件的名字
        # meta['seed'] = seed
        # meta['exp_name'] = osp.basename(args.config)
    # filename = cfg.work_dir.split('/')[-1]
    # basename = filename[:-3]
    # work_dir = f'./work_dirs/{basename}'
    # for filename in os.listdir(work_dir):
    #     if filename.startswith('best_'):
    #         os.rename(os.path.join(work_dir, filename), os.path.join(work_dir, 'best.pth'))
    #         print(f'Renamed {filename} to best.pth')
    #         break

if __name__ == '__main__':
    main()
