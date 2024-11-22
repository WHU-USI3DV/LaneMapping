import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.registry import NET
from ..registry import build_pcencoder, build_backbone, build_heads, build_bkdecoder
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


@NET.register_module
class Detector1stage(nn.Module):
    def __init__(self,
                 head_type='seg',
                 loss_type='row_ce',
                 cfg=None):
        super(Detector1stage, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(144, 144, 3, 3)
        self.pcencoder = build_pcencoder(cfg)
        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg)
        self.head_type = head_type
        self.loss_type = loss_type

    def forward(self, batch, is_get_features=False, stack_local_global_features=False):
        output = {}
        # fea = self.pcencoder(batch)  # Only resnet       
        fea, fea_up, bi_seg, endp_est = self.pcencoder(batch)  # Only resnet
        

        # print('feature shape after prencoder: ', fea.shape)  # torch.Size([4, 64, 144, 144])
        # print('UPSAMPLED feature shape after prencoder: ', fea_up.shape)  # torch.Size([4, 8, 288, 288])
        # _, _, h_fpn, w_fpn = fea_fpn.shape
        # upsample the feature to original shape:
        # fpn_feature_map = nn.functional.interpolate(fea_fpn, size=(1152, 1152), mode='bilinear', align_corners=True)
        # print('feature shape befor backbone, ', fea.shape)
        if is_get_features:
            fea, list_features = self.backbone(fea, True)
            output.update({'features': list_features})
        else:
            if self.cfg.vit_seg == True:
                # print("with vit seg: ", self.cfg.vit_seg)
                fea = self.backbone(fea)
        # print('feature shape after backbone, ', fea.shape)  # torch.Size([4, 8, 144, 144])

        if self.cfg.heads.type == 'RowSharNotReducRef':
            out = self.heads(fea)
        else:
            out = self.heads(fea, fea_up, endp_est)
        out['semantic_seg'] = bi_seg
        out['endp_est'] = endp_est

        if self.training:
            # self.heads.apply(self.init_weights)   #TODO: initialize weight by Xiaoxin
            output.update(self.heads.loss(out, batch, self.loss_type))
        else:
            output.update(self.heads.get_exist_coor_endp_dict(out))
            output.update({
                    'lane_maps': self.heads.get_lane_map_numpy_with_label(
                        output, batch, is_flip=self.cfg.flip_label, is_img=self.cfg.view, is_get_1_stage_result=False,
                        is_gt_avai=self.cfg.is_gt_avai)})
            if self.cfg.show_result:
                output.update({
                    'pred_maps': self.heads.get_lane_map_on_source_image(output, batch)})

        del endp_est, bi_seg, fea
        return output

    def init_weights(self, m, pretrained=None):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
