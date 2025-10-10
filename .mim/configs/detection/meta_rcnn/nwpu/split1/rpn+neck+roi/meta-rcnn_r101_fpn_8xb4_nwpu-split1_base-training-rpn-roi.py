_base_ = [
    '../../../../_base_/datasets/nway_kshot/data_b.py',
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
evaluation = dict(interval=4000)
checkpoint_config = dict(interval=4000)
runner = dict(max_iters=18000)
optimizer = dict(
    lr=0.005,
    paramwise_cfg=dict(
        custom_keys={
            # 为偏移量生成网络设置更低的学习率
            'rpn_conv_offset.weight': dict(lr_mult=0.05),
            'rpn_conv_offset.bias': dict(lr_mult=0.05),
        }
    ))
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
        with_dcn=True),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='GenericRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[8,16]),
        bbox_head=dict(num_classes=7, num_meta_classes=7)))
