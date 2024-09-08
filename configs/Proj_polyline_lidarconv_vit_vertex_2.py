'''
* Thanks: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* We also thank the MMDetection3D and MapTR for lidar encoder
* modified by Xiaoxin to adapt the mobile laster scanning lane detection and reconstruction
'''
import os

seed = 2021
gpus = 1
distributed = False
load_from = None 
finetune_from = None 
log_dir = './logs'
view = False
number_lanes = 12
number_orients = 11  # [0, 1, 2, ,,, 10]
gt_downsample_ratio = 8  # GT downsample range
flip_label = False
use_lidar = True  # if load ego lidar points

net = dict(
    type='Detector1stage',
    head_type='row',
    loss_type='row_ce'
)

lidar_point_cloud_range = [-15., -25., -2., 15., 25., 2.]
grid_size = [576, 576, 10]  # cooresponding to : x, y ,z
pcencoder = dict(
    type='LidarEncoder',  # 若encoder是点云，则真值也需要从点云坐标系开始处理
    Xn=144,
    Yn=144,
    out_channels=64,
    lidar_encoder=dict(
        voxelize=dict(
            point_cloud_range=lidar_point_cloud_range,
            max_num_points=10,         
            grid_shape=grid_size,
            max_voxels=1000000),
        backnone=dict(
            type='SparseEncoder',
            in_channels=4,   # WHU is 4: x, y, z, intensity; NuScenes is 5
            sparse_shape=[21, 600, 600],   # z, y, x A: ok  -- D,H,W
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                        128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
            block_type='basicblock'
        ),
    )
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
    output_channels=8,
    expansion_factor=4,
    dim_head=64,
    dropout=0.,
    emb_dropout=0., 
    is_with_shared_mlp=False,
    is_with_llm=False,
)

heads = dict(
    type='ColumnProposal2',    # column attention & column proposal regression
    dim_feat=8, # input feat channels
    row_size=144,
    dim_shared=100,  
    num_prop = 72,   # 72, 36,  18
    prop_width = 2,  # 2,   4,  8  #144 / num_prop
    prop_half_buff = 4, # 4, 3, 1, #
    dim_token = 512,   # 1024
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
    endp_mode = 'endp_est',   # endpoint detection mode: "endpoint" for local+global, "endp_est" for local.
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
exist_thre = 0.2  
coor_thre = 0.2
endp_thre =  0.08  
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
  lr = 0.00015,  # 0.0001
)

epochs = 45
batch_size = 4 # 8

workers= 8 # 12
total_iter = (2904 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 1
save_ep = 5
validate_buffer=10

### Setting Here ###
dataset_path = './data/LaserLane/TrainValAll'             # 'All-ordered' means with ordered anchor; 'All' means ordered without anchor
data_split_file= 'data_split-shuffle-lidar-range.json'
# data_split_file= 'data_split-small.json'    # test small dataset

### Setting Here ###
dataset_type = 'LaserLaneProposalEgo'
dataset_color_augment = False
gt_init_vertex = os.path.join(dataset_path, 'labels_inside_lidar_range', 'sparse_seq')
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


# Ablation study
vit_seg = True
column_att = False
column_transformer_decoder=False
spatial_att = True
cls_smooth = False

lidar_pipeline=[
    dict(
        type='Pack3DDetInputs',
        keys=['points'],
        meta_keys=[]),
]
        
