'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* modified by Xiaoxin to adapt the mobile laster scanning lane detection and reconstruction
'''
import os
seed = 2021
load_from = None
finetune_from = None
log_dir = './logs'
view = False
number_lanes = 7

net = dict(
    type='Detector',
)

pcencoder = dict(
    type='PostProjector',
    resnet='resnet34',
    pretrained=False,
    replace_stride_with_dilation=[False, True, False],
    out_conv=True,
    in_channels=[64, 128, 256, -1]
)
featuremap_out_channel = 64
list_img_size_xy = [1152, 1152]

backbone = dict(
    type='VitSegNet', # GFC-T
    image_size=144,
    patch_size=8,
    channels=64,
    dim=512,
    depth=3,
    heads=16,
    output_channels=1024,
    expansion_factor=4,
    dim_head=64,
    dropout=0.,
    emb_dropout=0., # TBD
    # is_with_shared_mlp=False,
)

heads = dict(
    type='GridSeg',
    num_1=1024,
    num_2=2048,
    num_classes=number_lanes, # 11 lanes(0, 1, 2, ...) and background (255)
)

conf_thr = 0.5
view = True

# BGR Format to OpenCV
cls_lane_color = [
    (0, 0, 255),
    (0, 50, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 100),
    (255, 0, 255), # add by xiaoxin
    (255, 255, 0),  # add by xiaoxin
    (255, 100, 0),  # add by xiaoxin
    (0, 100, 255),  # add by xiaoxin
    (100, 0, 255),  # add by xiaoxin
    (100, 255, 0)  # add by xiaoxin
]

optimizer = dict(
  type = 'Adam', #'AdamW',
  lr = 0.0002,
)

epochs = 60
batch_size = 4
total_iter = (7687 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 2
save_ep = 1

### Setting Here ###
dataset_path = './data/LaserLane' # './data/KLane'
### Setting Here ###
dataset_type = 'LaserLane'
gt_init_vertex = os.path.join(dataset_path, 'labels', 'annotation_seq')
init_vertex_dir = None
test = False

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        mode='train',
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        mode='test',
    )
)
workers=12
