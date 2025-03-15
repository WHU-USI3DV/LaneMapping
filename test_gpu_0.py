'''
* The source code is from Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* 
* We modify the code to fit our project. mixiaoxin@whu.edu.cn
'''
import os
GPUS_EN = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS_EN
import torch.backends.cudnn as cudnn
import time
import cv2
import open3d as o3d
import pickle

time_now = time.localtime()
time_log = '%04d-%02d-%02d-%02d-%02d-%02d' % (time_now.tm_year, time_now.tm_mon, time_now.tm_mday, time_now.tm_hour, time_now.tm_min, time_now.tm_sec)

from baseline.utils.config import Config
from baseline.engine.runner import Runner

from baseline.engine.runner import load_config_and_runner

def main():
    ### Set here ###
    time_start = time.localtime()
    # for Xiaoxin's Metho: row ref
    path_config = 'logs/PretrainedLaserLaneProposal/configs_Proj_polyline_fpn_vit_vertex_2.py'
    path_ckpt = 'logs/PretrainedLaserLaneProposal/ckpt/best.pth'
    
    # for KLane:
    # path_config = 'logs/klane-2/configs_Proj28_GFC-T3_RowRef_82_73_laser.py'
    # path_ckpt = 'logs/klane-2/ckpt/best.pth'

    # for segmentation     
    # path_config = 'logs/fpn_seg/configs_Proj_FPN_Seg.py'
    # path_ckpt = 'logs/fpn_seg/ckpt/best.pth'
    ### Set here ###

    ### Settings ###
    cudnn.benchmark = True
    cfg, runner = load_config_and_runner(path_config, GPUS_EN)

    cfg.gpus = len(GPUS_EN.split(','))
    print(f'* Config: [{path_config}] is loaded')
    runner.load_ckpt(path_ckpt)
    print(f'* ckpt: [{path_ckpt}] is loaded')
    ### Settings ###

    # runner.eval_conditional()  # original
    runner.cfg.show_result = True
    # if cfg.view_detail is True, the network will output multiple polylines: predicted, predicted_smooth, exp_smooth and offset_smooth
    # Otherwise, the network will only output the polylines with offsets
    runner.cfg.view_detail = False
    # print("cfg: ", runner.cfg)
    # Xiaoxin's Method:
    # mode_data = cfg.dataset.infer_only  # if only infer without evaluation
    mode_data = cfg.dataset.test        # if infer with evaluation (ground truth available)
    runner.infer_lane_coordinate_endpoint_semantics(path_ckpt=path_ckpt, mode_data=mode_data,  mode_view=True, gt_avail=cfg.is_gt_avai,\
                                                    write_lane_vertex=True, \
                                                    eval_coor=True, eval_endp=False, eval_semantic=True
                                                    )
    
    
    # KLane:
    # runner.infer_lane_coordinate(path_ckpt=path_ckpt, mode_view=True, gt_avail=True, write_lane_vertex=False)
    
    # Segmentor:
    # runner.infer_lane_geometry_segmentation_segmentor(path_ckpt=path_ckpt, mode_view=True)
    
    time_terminal = time.localtime()
    print(f'start time: {[time_start]}')
    print(f'terminal time: {[time_terminal]}')


if __name__ == '__main__':
    main()
