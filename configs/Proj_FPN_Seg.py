'''
* Copyright (c) Xiaoxin Mi. All rights reserved.
* author: Xiaoxin Mi
* e-mail: mixiaoxin@whu.edu.cn
'''
import os

seed = 2021
gpus = 1
distributed = False
load_from = './logs/2023-06-03-15-05-11/ckpt/best.pth' #None #
finetune_from = None #'./logs/9-global-endp-attr-fpn-weighted/ckpt/best.pth'        #None
log_dir = './logs'
view = False
number_lanes = 12
number_orients = 11  # [0, 1, 2, ,,, 10]
flip_label = False


net = dict(
    type='Segmentor',
    head_type='seg',
    loss_type='ce'
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

conf_thr =  0.1  #0.3
exist_thr = 0.2  #0.3
view = True
seg_thre = 0.1
endp_thre = 0.1

# BGR Format to OpenCV
cls_lane_color = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (42, 42, 128), # add by xiaoxin
    (158, 168, 3),  # add by xiaoxin
    (240, 32, 160),  # add by xiaoxin
    (84, 46, 8),  # add by xiaoxin
    (255, 97, 0),  # add by xiaoxin
    (100, 255, 0)  # add by xiaoxin
]

optimizer = dict(
  type = 'Adam', #'AdamW',
  lr = 0.0001,  # 0.0001
)

epochs = 16
batch_size = 6
total_iter = (1132 // batch_size) * epochs   # used to be 2094
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 1
save_ep = 4

### Setting Here ###
dataset_path = './data/LaserLane/All'              # 'All-ordered' means with ordered anchor; 'All' means ordered without anchor
data_split_file= 'data_split-shuffle.json'
### Setting Here ###
dataset_type = 'LaserLane'
gt_init_vertex = os.path.join(dataset_path, 'labels', 'sparse_seq')
init_vertex_dir = None
test = False

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        # data_split_file=data_split_file,
        mode='train',
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        # data_split_file=data_split_file,
        mode='val',
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        # data_split_file=data_split_file,
        mode='test',
    )
)
# When train the network by DDP, the number_workers should be set as 0. Otherwise, the " ...cannot pickle the 'Module' error" occurs.
workers= 12
