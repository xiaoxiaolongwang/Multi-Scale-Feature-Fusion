_base_ = [
    '../../../../_base_/datasets/nway_kshot/base_nwpu.py',
    '../../../../_base_/schedules/schedule.py', '../../../meta-rcnn_r101_c4.py',
    '../../../../_base_/default_runtime.py'
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
evaluation = dict(
    interval=6000,  # 每隔 6000 次迭代评估一次
    metric='mAP',  # 使用 mAP 作为评估指标
    save_best='auto',  # 自动保存最佳模型
    rule='greater'  # 根据 mAP 的最大值保存最佳模型
)
checkpoint_config = dict(
    interval=6000,  # 每隔 6000 次迭代保存一次模型
    max_keep_ckpts=1  # 只保留一个最佳模型
)
runner = dict(max_iters=18000)
optimizer = dict(
    lr=0.005)
# model settings
model = dict(
    backbone=dict(dilations=(2, 1, 1),out_indices=(1, 2),frozen_stages=2,),
    neck=dict(type='FPN',in_channels=[512,1024],out_channels=1024,num_outs=2,),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            scale_major=False,
            strides=[8,16]),
        with_dcn=False),
    roi_head=dict(
        bbox_roi_extractor=dict(
            # type='GenericRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[8,16]),
        bbox_head=dict(num_classes=7, num_meta_classes=7)))
