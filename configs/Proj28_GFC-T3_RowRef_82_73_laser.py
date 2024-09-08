'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* modified by Xiaoxin to adapt the mobile laster scanning lane detection and reconstruction
'''
import os
seed = 2021
distributed = False
load_from = None
finetune_from = None
log_dir = './logs'
view = False
number_lanes = 12
flip_label = False

net = dict(
    type='Detector1stage',
    head_type='row',
    loss_type='row_ce'
)

pcencoder = dict(
    type='PostProjector2',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, False],
    out_conv=True,
    in_channels=[64, 128, 256, -1]
)
featuremap_out_channel = 64
list_img_size_xy = [1152, 1152]

backbone = dict(
    type='VitSegNet', # GFC-T
    image_size=144,
    patch_h_size=8,
    patch_w_size=8,
    channels=64,
    dim=512,
    depth=3,
    heads=16,
    output_channels=1024,
    expansion_factor=4,
    dim_head=64,
    dropout=0.,
    emb_dropout=0., # TBD
    is_with_shared_mlp=False,
)

heads = dict(
    # type='RowSharNotReducRef_3',    # 4 heads: existence & vertex  regression per row & endpoints regression & endpoints offset regression
    # type='RowSharNotReducRef_2',    # 3 heads: existence & vertex  regression per row & endpoints regression
    type='RowSharNotReducRef',    # 2 heads: existence & vertex regression per row
    dim_feat=8, # input feat channels
    row_size=144,
    dim_shared=512,
    lambda_cls=1.,
    thr_ext = 0.3,
    off_grid = 2,
    dim_token = 1024,
    tr_depth = 1,
    tr_heads = 16,
    tr_dim_head = 64,
    tr_mlp_dim = 2048,
    tr_dropout = 0.,
    tr_emb_dropout = 0.,
    is_reuse_same_network = False,
)

conf_thr = 0.5
view = True
show_result = False  # if draw detected lane on the the raw map

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
  lr = 0.0001,
)

epochs = 45
batch_size = 6
total_iter = (2904 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 1
save_ep = 1

### Setting Here ###
dataset_path =   './../All' #'./data/LaserLane/All' # './data/KLane'
### Setting Here ###
# dataset_type = 'KLane'
dataset_type = 'LaserLane'
gt_init_vertex = os.path.join(dataset_path, 'labels', 'sparse_seq')
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
