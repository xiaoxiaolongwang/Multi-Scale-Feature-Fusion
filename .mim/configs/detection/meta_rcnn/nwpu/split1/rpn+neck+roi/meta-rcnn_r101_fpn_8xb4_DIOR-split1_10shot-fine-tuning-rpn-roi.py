_base_ = [
    '../../../../_base_/datasets/nway_kshot/data_n10.py',
    '../../../../_base_/schedules/schedule.py', '../../../meta-rcnn_r101_c4.py',
    '../../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotnwpuDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT1_10SHOT')],
            num_novel_shots=10,
            num_base_shots=10,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=100,
    metric='mAP',
    save_best='auto',
    rule='greater',
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=100,max_keep_ckpts=1)
optimizer = dict(
    lr=0.001,
    paramwise_cfg=dict(
        custom_keys={
            # 为偏移量生成网络设置更低的学习率
            'rpn_conv_offset.weight': dict(lr_mult=0.05),
            'rpn_conv_offset.bias': dict(lr_mult=0.05),
        }
    ))
lr_config = dict(warmup=None)
runner = dict(max_iters=2000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/meta-rcnn_r101_fpn_8xb4_nwpu-split1_base-training-rpn-roi/best.pth'
# model settings
model = dict(
    frozen_parameters=[
    'backbone', 'shared_head',
],
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
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[8,16]),
        bbox_head=dict(num_classes=10, num_meta_classes=10)))
