_base_ = [
    '../../../_base_/datasets/nway_kshot/base_voc.py',
    '../../../_base_/schedules/schedule.py', '../../meta-rcnn_r101_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=False,
        dataset=dict(classes='BASE_CLASSES_SPLIT1'),
        support_dataset=dict(classes='BASE_CLASSES_SPLIT1')),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'),
    model_init=dict(classes='BASE_CLASSES_SPLIT1'))
lr_config = dict(warmup_iters=100, step=[16000])
evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)
runner = dict(max_iters=18000)
optimizer = dict(lr=0.005)
# model settings
model = dict(
    backbone=dict( 
        out_indices=(0, 1, 2,),  # 每个状态产生的特征图输出的索引
        ),
    neck=dict(
        type='FPN',  # 检测器的 neck 是 FPN，我们同样支持 'NASFPN', 'PAFPN' 等，更多细节可以参考 https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.necks.FPN
        in_channels=[256, 512, 1024],  # 输入通道数，这与主干网络的输出通道一致
        out_channels=1024,  # 金字塔特征图每一层的输出通道
        num_outs=4),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=512,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            scale_major=False,
            strides=[4,8,16,32])),
    roi_head=dict(bbox_head=dict(num_classes=15, num_meta_classes=15)))
