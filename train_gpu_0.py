'''
@Author： Xiaoxin Mi (mixiaoxin@whu.edu.cn, xiaoxin.mi@whut.edu.cn)
@ Thanks： KLane for the original framework
'''
import os
GPUS_EN = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS_EN
import torch
import torch.backends.cudnn as cudnn
import time
import shutil

time_now = time.localtime()
time_log = '%04d-%02d-%02d-%02d-%02d-%02d' % (time_now.tm_year, time_now.tm_mon, time_now.tm_mday, time_now.tm_hour, time_now.tm_min, time_now.tm_sec)

from baseline.utils.config import Config
from baseline.engine.runner import Runner

def main():
    path_config = './configs/Proj_polyline_fpn_vit_vertex_2.py'   # Xiaoxin's method
    # path_config = './configs/Proj_polyline_lidarconv_vit_vertex_2.py'   # Xiaoxin's method with lidar sparse covn
    # path_config = './configs/Proj_polyline_fpn_mixseg_vertex.py'  # comparision: vit to mixseg
    # path_config = './configs/Proj28_GFC-T3_RowRef_82_73_laser.py' # original KLane
    # path_config = './configs/Proj_FPN_Seg.py' # solo resnet-34 for semantic segmentation

    path_split = path_config.split('/')
    cfg = Config.fromfile(path_config)
    cfg.log_dir = cfg.log_dir + '/' + time_log
    cfg.time_log = time_log
    os.makedirs(cfg.log_dir, exist_ok=True)
    shutil.copyfile(path_config, cfg.log_dir + '/' + path_split[-2] + '_' + path_split[-1].split('.')[0] + '.py')
    cfg.work_dirs = cfg.log_dir + '/' + cfg.dataset.train.type
    cfg.gpus = len(GPUS_EN.split(','))

    cudnn.benchmark = False
    runner = Runner(cfg)
    runner.train()
    # runner.train_small(train_batch=2, valid_samples=40)

if __name__ == '__main__':
    main()
    