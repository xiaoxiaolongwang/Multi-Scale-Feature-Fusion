_base_ = [
    '../../../../_base_/datasets/nway_kshot/few_shot_nwpu.py',
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
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT1_3SHOT')],
            num_novel_shots=3,
            num_base_shots=3,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=100, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=100)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=1000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/meta-rcnn_r101_c4_8xb4_nwpu-split1_base-training/latest.pth'
# model settings
model = dict(
    frozen_parameters=[
    'backbone', 'shared_head'
],
roi_head=dict(bbox_head=dict(num_classes=10, num_meta_classes=10)))
