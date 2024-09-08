import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape, DynamicScatter3D #, Voxelization
from mmdet3d.registry import MODELS


from baseline.models.registry import PCENCODER

@PCENCODER.register_module
class LidarEncoder(nn.Module):
    
    def __init__(self,
                 Xn=144,
                 Yn=144,
                 out_channels=8,
                lidar_encoder=None,
                cfg=None):
        super(LidarEncoder, self).__init__()
        self.cfg=cfg
        self.out_channels = out_channels
        self.Xn = Xn
        self.Yn = Yn

        if lidar_encoder is not None:
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = VoxelizationByGridShape(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter3D(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": MODELS.build(lidar_encoder["backnone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)

        self.fea_aligner = nn.Sequential(
            nn.Conv2d(lidar_encoder.backnone.output_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        # feature from high resolution to low resolution
        self.fea_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        # decrease feature channels
        self.output_layer_fea = nn.Conv2d(out_channels, 8, kernel_size=1, stride=1, padding=0)

        # to fit the other predictions
        self.output_layer_binary_seg = nn.Conv2d(out_channels, 3, kernel_size=1, stride=1, padding=0)
        self.output_layer_endp = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, sample):      
        lidar_pts = [item.data for item in sample['points']]
        batch_size = len(lidar_pts)
        lidar_feat = self.extract_lidar_feat(lidar_pts)   # N, C * D, H, W

        # flip H to match BEV annotations
        # lidar feature dim is:  N， C*D， H， w
        lidar_feat = torch.flip(lidar_feat, dims=[2])

        lidar_fea_up = nn.functional.interpolate(lidar_feat, size=(self.Yn*2, self.Xn*2), mode='bicubic', align_corners=False)
        lidar_fea_up = self.fea_aligner(lidar_fea_up)

        # to fit other predictions
        lidar_fea = self.fea_conv(lidar_fea_up)
        fea_bi_seg = self._upsample(self.output_layer_binary_seg(F.relu(lidar_fea_up)), h=self.Yn * self.cfg.gt_downsample_ratio, w=self.Yn * self.cfg.gt_downsample_ratio)
        fea_end = self._upsample(self.output_layer_endp(F.relu(lidar_fea_up)), h=self.Yn * self.cfg.gt_downsample_ratio, w=self.Yn * self.cfg.gt_downsample_ratio)    
        lidar_fea_up = self.output_layer_fea(lidar_fea_up)

        return lidar_fea, lidar_fea_up, fea_bi_seg, fea_end

    def extract_lidar_feat(self,points):
        # print('point size in extrac lidar feature: ', len(points))  # [1 or 4]
        # print('points[0] size in extract lidar feature: ', points[0].size())  #[xxx, 4]
        feats, coords, sizes = self.voxelize(points)
        
        # 360000个Voxel
        # torch.Size([90000, 10, 4]) torch.Size([90000, 3]) torch.Size([90000])
        # [非空Voxel数量，Voxel内最多点数，输入通道数量]；【非空Voxel数量，3表示（z_id，y_id， x_id）】
        # print("coords[-1, 0]: ", coords[-1, :])   # tensor([  3, 112, 403,  23], device='cuda:0', dtype=torch.int32)
        batch_size = coords[-1, 0] + 1   
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size)
        
        return lidar_feat
    
    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            
            ret = self.lidar_modal_extractor["voxelize"](res)
            # print('ret is: ', ret[0].size(), ret[1].size(), ret[2].size())
            # torch.Size([90000, 10, 4]) torch.Size([90000, 3]) torch.Size([90000])
            # [非空Voxel数量，Voxel内最多点数，输入通道数量]；【非空Voxel数量，3表示（z_id，y_id， x_id）】
            
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))   # 给coords上的增加一维（位于第0维），标识数字所在的batch id
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes