'''
* author: Xiaoxin Mi
* e-mail: mixiaoxin@whu.edu.cn
'''
import torch
import torch.nn as nn
import numpy as np

from baseline.models.registry import HEADS

@HEADS.register_module
class PixelSeg(nn.Module):
    
    def __init__(self,
                semantic_num=2,  # exlcude background
                num_classes=7,
                cfg=None):
        super(PixelSeg, self).__init__()
        self.cfg=cfg

        self.class_predictor = nn.Sequential(
            nn.Conv2d(num_1, num_2, 1),
            nn.Conv2d(num_2, num_classes, 1)
        )

    def forward(self, x):
        class_output = self.class_predictor(x)
        
        return class_output

    

    def get_lane_map_numpy_with_label(self, output, data):        
        lane_maps = dict()

        
        return lane_maps

    def get_rgb_img_from_cls_map(self, cls_map):
        temp_rgb_img = np.zeros((144, 144, 3), dtype=np.uint8)

        return temp_rgb_img
        