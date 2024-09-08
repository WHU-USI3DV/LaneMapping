'''
@Author: Xiaoxin Mi
@Email: mixiaoxin@whu.edu.cn
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.registry import NET
from ..registry import build_pcencoder, build_backbone
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


@NET.register_module
class Segmentor(nn.Module):
    def __init__(self,
                 head_type='seg',
                 loss_type='ce',
                 cfg=None):
        super(Segmentor, self).__init__()
        self.cfg = cfg
        self.pcencoder = build_pcencoder(cfg)
        self.head_type = head_type
        self.loss_type = loss_type

    def forward(self, batch):
        output = {}
        # fea = self.pcencoder(batch)  # Only resnet
        _, _, bi_seg, endp_est = self.pcencoder(batch)  # Only resnet
        pred = dict()
        pred['seg'] = bi_seg
        pred['endp'] = endp_est
        if self.training:
            output.update(self.pcencoder.loss(pred, batch))
        else:
            output.update(self.pcencoder.infer_validate(pred, seg_thre=self.cfg.seg_thre, endp_thre=self.cfg.endp_thre))
            if self.cfg.view is True:
                output.update(self.pcencoder.get_pred_seg_endp_displays(output, batch))

        del endp_est, bi_seg
        return output


    def init_weights(self, m, pretrained=None):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)