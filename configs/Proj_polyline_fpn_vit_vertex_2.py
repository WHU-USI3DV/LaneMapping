'''
* Thanks: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* modified by Xiaoxin to adapt the mobile laster scanning lane mapping
'''
import os

seed = 2021
gpus = 1
distributed = False
load_from = None #None 
finetune_from = None   #None
log_dir = './logs'
view = False
number_lanes = 12
number_orients = 11  # [0, 1, 2, ,,, 10]
gt_downsample_ratio = 8  # GT downsample range
flip_label = False
use_lidar = False  # if load ego lidar points
is_gt_avai = True  # if grounf truth is available

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
    patch_h_size=8,    # 8
    patch_w_size=8,     # 8
    channels=64, 
    dim=512,   # 512
    depth=3,
    heads=16,
    output_channels=8,
    expansion_factor=4,
    dim_head=64,
    dropout=0.,
    emb_dropout=0., # TBD
    is_with_shared_mlp=False,
    is_with_llm=False,
)

heads = dict(
    type='ColumnProposal2',    # column attention & column proposal regression
    dim_feat=8, # input feat channels
    row_size=144,
    dim_shared=100,  # 100, used to be 512
    num_prop = 72,   # 72, 36,  18
    prop_width = 2,  # 2,   4,  8  #144 / num_prop
    prop_half_buff = 4, # 4, 3, 1, #
    dim_token = 512,   
    tr_depth = 1,
    tr_heads = 16,
    tr_dim_head = 64,
    tr_mlp_dim = 512,
    tr_dropout = 0.,
    tr_emb_dropout = 0.,
    row_dim_token = 96,   # 256 for relative row position embedding; 96 for single row embedding;
    row_tr_depth=1,  
    row_tr_heads=12,  
    row_tr_dim_head=8,
    row_tr_mlp_dim=144,
    row_tr_dropout=0.,
    row_tr_emb_dropout=0.,
    endp_mode = 'endp_est',   
    cls_exp = True,
    ext_w = 3.,
    ext_smooth_w = 1.,
    lambda_cls=1.,
    mean_loss_w = 1., #0.05
    cls_smooth_loss_w = 10,  # 1
    orient_w = 1.,
    endp_loss_w = 10.,
    offset_w = 1.,  # 0.5
    freeze_endp = False,
    freeze_ori = False
)

proposal_obj_thre = 0.3
exist_thre = 0.2  #0.3
coor_thre = 0.2
endp_thre =  0.08  #0.3
show_result = False  # if draw detected lane on the the raw map
view_detail = False  # if draw detailed polylines

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
  lr = 0.00015,  # 0.0001 when batch_size = 4
)

epochs = 45
batch_size = 6
total_iter = (2904 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 1
save_ep = 5
validate_buffer=10

### Setting Here ###
dataset_path = './data/LaserLane/TrainValAll'            # 'All-ordered' means with ordered anchor; 'All' means ordered without anchor
# dataset_path = './data/LaserLane/Test-Area-1'          # test on Area-1
# dataset_path = './data/LaserLane/Test-Area-2'          # test on Area-2
data_split_file= 'data_split-shuffle.json'
### Setting Here ###
# dataset_type = 'KLane'   # for comparison, then dataset_path should be "'./data/LaserLane/All-ordered' ", lane markings sholud be arranged in order from left to right
dataset_type = 'LaserLaneProposal'
dataset_color_augment = False
gt_init_vertex = os.path.join(dataset_path, 'labels', 'sparse_seq')
init_vertex_dir = None
test = False

dataset = dict(    
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        data_split_file=data_split_file,
        mode='train',
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        data_split_file=data_split_file,
        mode='val',
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        data_split_file=data_split_file,
        mode='test',
    )
)
# When train the network by DDP, the number_workers should be set as 0. Otherwise, the " ...cannot pickle the 'Module' error" occurs.
workers= 12 # 12

# Ablation study
vit_seg = True
column_att = False
column_transformer_decoder=False
spatial_att = True
cls_smooth = False
