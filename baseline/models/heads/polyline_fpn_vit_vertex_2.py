'''
* author: Xiaoxin Mi
* e-mail:  mixiaoxin@whu.edu.cn

* modified by Xiaoxin for Lane regression
* Taking the Klane, ULSD as the reference
* change the lane attention part
* this file modifies the row self attention: 1) absolute self attention; 2) relative self attention(no improvement)
* this file modifies the binary segmentation inside the column proposal to enhance the lane detection
* this file merges the lines whose endpoints overlapped, but keep the semantics of vertexes and topology changing points
* this file predicts endpoints at the image level but not the column proposal level.
'''
import math
import random
import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision
import skimage

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from baseline.models.registry import HEADS
from baseline.models.loss import FocalLoss
from baseline.models.loss import MeanLoss
from baseline.utils.vis_utils import get_lane_on_raw_image_coordinates, \
    get_semantic_lane_on_raw_image_coordinates, get_endp_on_raw_image, get_gt_endp_on_raw_image
from baseline.utils.polyline_utils import polyline_uniform_semantics_by_statistics, smooth_cls_line_per_batch,\
                                          sort_lines_from_left_to_right, polyline_NMS2, remove_short_polyline,\
                                          renew_semantic_map


from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding   # add by Xiaoxin
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .transformer import Transformer, FeedForward
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Conv_Pool_2d(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(input_dim, input_dim, kernel_size=(5,3), padding=(2, 1)))
        for in_c, out_c in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]):
            self.layers.append(nn.Sequential(nn.ReLU(inplace=True),
                          nn.BatchNorm2d(in_c), # very important
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)))
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


@HEADS.register_module
class ColumnProposal2(nn.Module):
    def __init__(self,
                dim_feat=8, # input feat channels
                row_size=144,
                dim_shared=512, # 
                num_prop = 72,
                prop_width=2,
                prop_half_buff = 4,
                dim_token = 1024,  # 
                tr_depth = 1,      # 
                tr_heads = 16,     # 
                tr_dim_head = 64,
                tr_mlp_dim = 2048,
                tr_dropout = 0.,
                tr_emb_dropout = 0.,
                row_dim_token = 64,
                row_tr_depth=1,  # 
                row_tr_heads=10,  # 
                row_tr_dim_head=12,
                row_tr_mlp_dim=128,
                row_tr_dropout=0.,
                row_tr_emb_dropout=0.,
                endp_mode = 'Regr',
                cls_exp = False,
                ext_w=1.,
                ext_smooth_w=1.,
                lambda_cls=1.,
                mean_loss_w = 0.,
                cls_smooth_loss_w = 0.,
                orient_w = 1.,
                endp_loss_w = 1.,
                offset_w = 1.,
                freeze_endp=False,
                freeze_ori=False,
                cfg=None):
        super(ColumnProposal2, self).__init__()
        self.cfg=cfg
        self.flip_label=self.cfg.flip_label
        print("USE CLASS EXPECTATION: ", cls_exp)
        print("FREEZE ENDPOINT BRANCH: ", freeze_endp)
        print("FREEZE ORIENTATION BRANCH: ", freeze_ori)

        ### Making Labels ###
        self.num_cls = self.cfg.number_lanes
        self.num_prop = num_prop
        self.prop_width = prop_width  # 144 / num_prop
        self.prop_half_buff = prop_half_buff
        self.num_orients = self.cfg.number_orients
        self.ext_w = ext_w
        self.ext_smooth_w = ext_smooth_w
        self.lambda_cls = lambda_cls
        self.mean_loss_w = mean_loss_w
        self.cls_smooth_loss_w = cls_smooth_loss_w
        self.endp_mode = endp_mode
        self.orient_w = orient_w
        self.endp_loss_w = endp_loss_w
        self.offset_loss_w = offset_w
        self.row_size = row_size

        self.tr_row_heads = row_tr_heads

        self.N_s = self.prop_width
        self.prop_fea_width = self.prop_width + 2 * self.prop_half_buff
        ### Making Labels ###


        ### MLP Encoder (1st Stage) ###
        self.row_tensor_maker = Rearrange('b c h w -> b (c w) h')
        self.img_tensor_maker = Rearrange('b c h w -> b (c w h)')
        self.img_embed_dims = dim_feat * row_size * row_size

        # reference: DETR from mmdetection repository, added by Xiaoxin
        self.reg_ffn = FeedForward(dim_feat, dim_feat*4)
        self.zero_pad_2d = nn.ZeroPad2d(padding=(self.prop_half_buff, self.prop_half_buff, 0, 0))
        self.zero_pad_2d_prop = nn.ZeroPad2d(padding=(int(self.prop_half_buff*2), int(self.prop_half_buff*2), 0, 0))
        self.zero_pad_2d_org = nn.ZeroPad2d(padding=(int(self.prop_half_buff*8), int(self.prop_half_buff*8), 0, 0))

        # First stage: Proposal generation
        # Feature map from 8 * 144 * 144 to 16 * 72 * 72
        # Every proposal has the receiption field 5 * 5
        # Proposal shape is: 72 * 5
        # Proposal amount is: 72 along column axis
        if self.num_prop == 72:
            self.generate_line_proposal = nn.Sequential(Conv_Pool_2d(dim_feat, hidden_dims=[], output_dim=2*dim_feat))
        if self.num_prop == 36:
            self.generate_line_proposal = nn.Sequential(Conv_Pool_2d(dim_feat, hidden_dims=[2*dim_feat], output_dim=4*dim_feat))
        if self.num_prop == 18:
            self.generate_line_proposal = nn.Sequential(Conv_Pool_2d(dim_feat, hidden_dims=[2*dim_feat, 4*dim_feat], output_dim=8*dim_feat))


        # Followed up with proposal attention module
        in_token_channel = 1* self.num_prop *dim_feat* self.prop_width
        self.to_token = nn.Sequential(
            Rearrange('c h -> (c h)'),  # w=1
            nn.Linear(in_token_channel, dim_token)
        )

        # old version for position embedding
        for idx_prop in range(self.num_prop):
            setattr(self, f'emb_{idx_prop}', nn.Parameter(torch.randn(dim_token)))
        
        self.emb_dropout = None
        if tr_emb_dropout != 0.:
            self.emb_dropout = nn.Dropout(tr_emb_dropout)
        self.tr_lane_correlator = nn.Sequential(
            Transformer(dim_token, tr_depth, tr_heads, tr_dim_head, tr_mlp_dim, tr_dropout),
            nn.LayerNorm(dim_token)
        )



        self.line_expand = nn.Sequential(
            nn.Linear(dim_token, in_token_channel),
            Rearrange('b n (c h w) -> b n c h w', c=dim_feat, h=144)
        )

        # Followed up with the line regression, downsample part  ### Refinement (2nd Stage) ###
        header_fea_dim = dim_feat*2
        self.head_common_layers = nn.Sequential( # resolution: 288 * 288 -> 144 * 144; dim: 16 -> 8
            nn.Conv2d(dim_feat*2, header_fea_dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(header_fea_dim),
            nn.Conv2d(header_fea_dim, header_fea_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(header_fea_dim)
        )

        self.to_token_row_seg_attention = nn.Sequential(
            Rearrange('b c h w -> b h (c w)'),
            # nn.Linear(dim_feat * self.prop_fea_width, row_dim_token),
            Rearrange('b h c -> b c h')
        )

        # Followed up with the line proposal ranking: Loss: nearest line distance
        # input is the result of the proposal token
        # output: the confidence of each proposal: positive or negative
        self.proposal_confidence = nn.Sequential(
            Rearrange('b c w -> b (c w)'),  # h=1
            nn.Linear(header_fea_dim * self.prop_fea_width *row_size, 2)
            #nn.Linear(dim_token, 2)
        )

        self.ext2 = nn.Sequential(
            nn.Conv1d(header_fea_dim * self.prop_fea_width, dim_shared, 1, 1, 0),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(dim_shared),
            nn.Conv1d(dim_shared, 3, 1, 1, 0),
            Rearrange('b c h -> b h c')
        )

        self.cls2 = nn.Sequential(
            nn.Conv1d(header_fea_dim * self.prop_fea_width, dim_shared, 1, 1, 0),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(dim_shared),
            nn.Conv1d(dim_shared, self.prop_fea_width, 1, 1, 0),
            Rearrange('b w h -> b h w')
        )

        self.offset2 = nn.Sequential(
            nn.Conv1d(header_fea_dim * self.prop_fea_width, dim_shared, 1, 1, 0),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(dim_shared),
            nn.Conv1d(dim_shared, self.prop_fea_width, 1, 1, 0),
            Rearrange('b w h -> b h w')
        )

        # input: 144 * 144
        # output: 144 * 144
        self.orient = nn.Sequential(
            nn.Conv2d(header_fea_dim, int(header_fea_dim / 2), kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(header_fea_dim / 2)),
            nn.Conv2d(int(header_fea_dim / 2), self.num_orients, 3, 1, 1)
        )

        # Upsample: Per proposal line segmentation
        self.head_upsample_layers = nn.Sequential(  # resolution: 288 * 288 -> 288 * 288; dim: 24 -> 8
            nn.Conv2d(dim_feat * 2, dim_feat, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim_feat),
            nn.Conv2d(dim_feat, dim_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_feat)
        )

        # self.bi_seg_proposal = nn.Conv2d(dim_feat, 1, kernel_size=1, stride=1, padding=0)     # for feature with 8 channels
        self.bi_seg_proposal = nn.Conv2d(dim_feat*2, 1, kernel_size=1, stride=1, padding=0)   # for feature with 16 channels


        # input: 288 * 288
        # output: 1152 * 1152
        self.endpoint = nn.Sequential(
            nn.Conv2d(2*dim_feat+1, int(dim_feat / 2), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(dim_feat / 2)),
            nn.Conv2d(int(dim_feat / 2), 1, 3, 1, 1)  # 0 channel for endpoints, 1 channel for background
            # Rearrange('b 1 h w -> b h w')
        )


        if freeze_endp:
            for idx_cls in range(self.num_cls):
                for param in self.endp.parameters():
                    param.requires_grad = False

        if freeze_ori:
            for idx_cls in range(self.num_cls):
                for param in self.orient.parameters():
                    param.requires_grad = False

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def _downsample_multiply(self, x, y):
        return torch.mul(F.avg_pool2d(x, kernel_size=8), y)

    def _upsample_cat(self, x, y):
        _, _, H, W = y.size()
        return torch.cat([F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True), y], dim=1)



    '''
    @param-x: the feature map after column attention
    @param-x_up: the feature map extracted from fpn for semantic segmentation
    @param-x_endp: the output from the fpn for endpoint detection
    '''
    def forward(self, x, x_up, x_endp):
        out_dict = dict()
        b_size, dim_feat, img_h, img_w = x.shape  # 4, 8, 144, 144
        self.b_size = b_size
 

        # Second stage: Proposal attention Module    
        ### Finish: 2nd Stage Processing ###
        if self.cfg.column_att:
            col_feats_batch = []
            feat_down = self.generate_line_proposal(x) # 4 * 16 * 72 * 72
            _, feat_down_dim, feat_down_H, feat_down_W = feat_down.shape
            for idx_b in range(b_size):
                ext_lane_tokens = []
                for idx_w in range(feat_down_W): # 72
                    temp_token = feat_down[idx_b, :, :, idx_w] # batch , channel, height, column
                    temp_token = self.to_token(temp_token) + getattr(self, f'emb_{idx_w}')  # (256,)
                    # expand because of unsqueeze backward error
                    ext_lane_tokens.append((temp_token.unsqueeze(0)).unsqueeze(0))  # (n_proposals72, 1, 1, 256)
                token_before = torch.cat(ext_lane_tokens, dim=1)
                if self.cfg.column_att:
                    tokens = self.tr_lane_correlator(token_before)  # (1, n_proposal72, (fea_dim8, 72, 1))
                else:
                    tokens = token_before

                column_fea = self.line_expand(tokens)
                tmp_fea_down = torch.squeeze(column_fea.squeeze(dim=0), dim=-1)
                col_feats_batch.append(tmp_fea_down.permute(1, 2, 0).unsqueeze(dim=0))

            
            col_feats_batch = torch.cat(col_feats_batch, dim=0)   # column feature after column attention: [4, 16, 72, 72]
            ### Finish: 2nd Stage Processing ###

            # Local and global feature concatenation
            col_fea_up = self._upsample_cat(col_feats_batch, x_up)  # feat_down: [4, 24, 288, 288]
            del col_feats_batch, feat_down, x_up
            torch.cuda.empty_cache()
        elif self.cfg.column_transformer_decoder:
            ############ whole batch #####################
            sparse_pe, img_pe = self.pe.forward(bs=self.b_size)
                # print("spare_pe, dense_pe, image_pe", sparse_pe.shape, dense_pe.shape, img_pe.shape)
            qs, _ = self.line_decoder(self.to_patch_embedding(x), image_pe=img_pe, sparse_prompt_embeddings=sparse_pe.cuda())  # [b, dim_token, 1, 72]
            col_feats_batch = self.reverse_query_embedding(qs)
            # print("col_fea_att", col_fea_att.shape)
            col_fea_up = self._upsample_cat(col_feats_batch, x_up)  # feat_down: [4, 24, 288, 288]
            del col_feats_batch, x_up
            torch.cuda.empty_cache()
            # print("col_fea_up", con_feat_up.shape)
        else:
            # Local and global feature concatenation
            col_fea_up = self._upsample_cat(x, x_up)  # feat_down: [4, 16, 288, 288]


        total_iter_round = 1
        tmp_iter = 0
        while tmp_iter < total_iter_round:
            tmp_iter += 1
            # 3-1: attributes aux
            # (1) semantic segmentation; (2) attribute points detection
            # out_dict.update({'endpoint': torch.sigmoid(self.endpoint(row_feat))})
            # out_dict.update({'endpoint': self._upsample(self.endpoint(F.relu(self._upsample_cat(propoal_fea_up, x_endp))), img_h * 8,
            #                                img_w * 8)})
            out_dict.update({'endpoint': self._upsample(
                self.endpoint(F.relu(self._upsample_cat(col_fea_up, x_endp))), img_h * 8,
                img_w * 8)})

            ### Third stage: positive proposal selection, line regression
            row_fea_up = self.head_common_layers(col_fea_up)  # input: [4, 16, 288, 288]; output: [4, 16, 144, 144]
            # propoal_fea_up = self.head_upsample_layers(con_feat_up)  # input: [4, 16, 288, 288]; output: [4, 8, 288, 288]
            # 3-2: Lane regression
            # (1) orient estimation; (2) existence estimation; (3) column coordinates estimation; (4) offset regression
            out_dict.update({'orient': self.orient(row_fea_up)})

            row_fea_up = self.zero_pad_2d(row_fea_up)
            col_fea_up = self.zero_pad_2d_prop(col_fea_up)
            proposal_obj = []
            proposal_ext2 = []
            proposal_cls2 = []
            proposal_offset2 = []
            proposal_bi_seg = []

            for id in range(self.num_prop):  # proposal size: 72
                # inter-row self attention
                local_prop_fea = row_fea_up[:, :, :, self.N_s*id:(self.N_s*id+self.prop_fea_width)]   # plus 2, 2 means the field of proposal on this feature map

                # segmentation attention version
                sp_bi_seg = torch.zeros((self.b_size, 1, img_h * 8, self.prop_fea_width * 8)).cuda()
                if self.cfg.spatial_att == True:
                    upsample_prop_fea = col_fea_up[:, :, :, 2 * self.N_s * id:(2 * self.N_s * id + 2 * self.prop_fea_width)]  # the feature map size is 2 times of local one

                    # for upsampled output:
                    sp_bi_seg = self._upsample(self.bi_seg_proposal(F.relu(upsample_prop_fea)), img_h * 8, self.prop_fea_width * 8)
                    # sp_bi_seg = self._upsample(self.bi_seg_proposal(F.relu(local_prop_fea)), img_h * 8, self.prop_fea_width * 8)
                    tokens_before = self._downsample_multiply(sp_bi_seg, local_prop_fea)
                else:
                    tokens_before = local_prop_fea
                tokens_after = self.to_token_row_seg_attention(tokens_before)
                del local_prop_fea, tokens_before
                torch.cuda.empty_cache()

                # column proposal objectiveness
                sp_obj = self.proposal_confidence(tokens_after)

                
                sp_ext = self.ext2(tokens_after)  # input: [4, c, 144]
                sp_cls = self.cls2(tokens_after)
                sp_offset = self.offset2(tokens_after)
                del tokens_after
                proposal_obj.append(torch.unsqueeze(sp_obj, dim=1))
                proposal_ext2.append(torch.unsqueeze(sp_ext, dim=1))
                proposal_cls2.append(torch.unsqueeze(sp_cls, dim=1))
                proposal_offset2.append(torch.unsqueeze(sp_offset, dim=1))
                proposal_bi_seg.append(torch.unsqueeze(sp_bi_seg, dim=1))

            proposal_confs_out = torch.cat(proposal_obj, dim=1)
            proposal_ext2_out = torch.cat(proposal_ext2, dim=1)
            proposal_cls2_out = torch.cat(proposal_cls2, dim=1)
            proposal_offset2_out = torch.cat(proposal_offset2, dim=1)
            proposal_bi_seg_out = torch.cat(proposal_bi_seg, dim=1)


        out_dict.update({'proposal_conf': proposal_confs_out})  # [4, 72, 2]
        out_dict.update({'ext2': proposal_ext2_out})            # [4, 72, 144, 3]
        out_dict.update({'cls2': proposal_cls2_out})            # [4, 72, 144, 22]
        out_dict.update({'offset2': proposal_offset2_out})      # [4, 72, 144, 22]
        out_dict.update({'prop_bi_seg': proposal_bi_seg_out})   # [4, 72, 2, 1152, 160+16]
        return out_dict

    def weighted_l1_loss(logits, target, mask=None):
        loss = F.l1_loss(logits, target, reduction='none')
        if mask is not None:
            # w = mask.mean(3, True).mean(2, True)
            # w[w == 0] = 1
            # loss = loss * (mask / w)
            loss = loss * mask
        return loss.mean()

    def loss(self, out, batch, loss_type=None):
        gt_proposal = batch['prop_obj']
        gt_exist = batch['prop_ext']
        gt_coors = batch['prop_coor']
        gt_offset = batch['prop_offset']
        gt_offset_mask = batch['prop_offset_mask']
        gt_bi_seg = batch['prop_bi_seg']

        lb_semantic_label_raw = batch['semantic_label_raw']
        lb_lc_endp = batch['endp_map']
        lb_lc_orient = batch['lc_orient']
        num_proposal = self.cfg.heads.num_prop


        # ignore the vertexes that are (1) out of the proposal RoI [0, 20]; (2) invalid in ground truth
        invalid_v = torch.where((gt_coors >= (self.prop_fea_width)) | (gt_coors < 0.) | (gt_exist==0))
        gt_coors[invalid_v] = -1.
        gt_exist[invalid_v] = 0
        vertex_valid = torch.where(gt_exist > 0)
        vertex_valid_size = len(vertex_valid[0])

        # calculate the line regression loss batch, proposals
        # if the corresponding min GT distance is bigger than 10, then this one is negative sample
        proposal_positive = torch.where(torch.sum(gt_exist, dim=2)>2)  # ([batch_ids], [proposal_ids])
        gt_proposal[proposal_positive[0], proposal_positive[1], 1] = 1
        proposal_negative = torch.where(gt_proposal[:, :, 1]==0)
        gt_proposal[proposal_negative[0], proposal_negative[1], 0] = 1
        


        EPS = 1e-12

        # A: vertex existence loss
        ### 2nd Stage ###
        proposal_loss = 0.
        ext_loss2 = 0.
        cls_loss2 = 0.
        cls_mean_loss2 = 0.
        cls_smooth_loss2=0.
        offset_loss = 0.
        orient_loss = 0.

        endp_loss = 0.
        binary_seg_loss = 0.

        orient_exist = torch.where(lb_lc_orient > 0)
        orient_loss += F.cross_entropy((out['orient'].permute(0, 2, 3, 1))[orient_exist], lb_lc_orient[orient_exist].long(), reduction='sum')

        # global semantic segmentation
        semantic_seg_loss = F.cross_entropy(out['semantic_seg'], lb_semantic_label_raw.type(torch.int64), reduction='sum')

        # global endpoint estimation
        endp_exist = torch.where(torch.sum(lb_lc_endp, dim=(1, 2)) > 1.)  # batch_id
        endp_weight = lb_lc_endp.clone()
        endp_weight[torch.where(endp_weight > EPS)] *= 4
        endp_weight[torch.where(endp_weight < EPS)] = 0.5
        lb_lc_endp[torch.where(lb_lc_endp > EPS)] = 1
        lb_lc_endp[torch.where(lb_lc_endp < EPS)] = 0
        if self.cfg.heads.endp_mode == 'endpoint':
            endp_loss_none = torchvision.ops.sigmoid_focal_loss(out['endpoint'][:, 0, :, :][endp_exist], lb_lc_endp[endp_exist], reduction='none')
        else:
            endp_loss_none = torchvision.ops.sigmoid_focal_loss(out['endp_est'][:, 0, :, :][endp_exist], lb_lc_endp[endp_exist], reduction='none')
        
        endp_loss += torch.sum(endp_weight[endp_exist] * endp_loss_none)
        # for b_id in range(self.b_size):
        #     endp_label = ls_lb_endp[b_id, :, :].detach().cpu().numpy()
        #     endp_weight_np = endp_weight[b_id, :, :].detach().cpu().numpy()
        #     cv2.imshow("end_label", endp_label)
        #     cv2.imshow("endp_weight", endp_weight_np)
        #     cv2.waitKey(0)
        del endp_loss_none, endp_weight
        torch.cuda.empty_cache()

        # semantic_org_lane_label = torch.sum(semantic_org_lane_label, dim=1)
        # semantic_org_lane_label[torch.where(semantic_org_lane_label > 0.)] = 1

        # binary segmentation loss in proposal
        if self.cfg.spatial_att == True:
            binary_seg_loss += torchvision.ops.sigmoid_focal_loss(out['prop_bi_seg'][proposal_positive[0], proposal_positive[1], :, :].reshape(-1, 1), gt_bi_seg[proposal_positive[0], proposal_positive[1], :, :].reshape(-1, 1), reduction='sum')
        else:
            binary_seg_loss = torch.tensor(binary_seg_loss).cuda()
        # binary_seg_loss += F.cross_entropy(out['prop_bi_seg'][proposal_positive[0], proposal_positive[1], ...].reshape(-1, 2), gt_bi_seg[proposal_positive[0], proposal_positive[1], ...].long().view(-1), reduction='sum')
        col_index = torch.arange(self.prop_fea_width).cuda()
        col_index_expand = col_index.repeat(self.b_size, num_proposal, self.row_size, 1)
        proposal_loss += F.binary_cross_entropy_with_logits(out['proposal_conf'], gt_proposal)
        ext_loss2 += F.cross_entropy(out['ext2'][proposal_positive[0], proposal_positive[1], :].reshape(-1, 3), gt_exist[proposal_positive[0], proposal_positive[1], :].long().view(-1), reduction='sum')
        row_idx = torch.arange(self.row_size).cuda()
        orient_idx = torch.arange(self.cfg.number_orients).cuda()
        orient_idx_expand = orient_idx.repeat(self.b_size,  self.row_size, self.row_size, 1)
        if self.cfg.heads.cls_exp:
            corr_idx_pred = torch.sum(col_index_expand * (out['cls2'].softmax(dim=3)), dim=3)
            cls_mean_loss2 += F.smooth_l1_loss(corr_idx_pred[vertex_valid], gt_coors[vertex_valid], reduction='sum')
            cls_loss2 += F.cross_entropy(out['cls2'][vertex_valid], gt_coors[vertex_valid].long(), reduction='sum')

            if self.cfg.cls_smooth == True:
                orient_idx_exp = torch.sum(orient_idx_expand * (out['orient'].softmax(dim=1)).permute(0, 2,3,1), dim=3)
                delta_orient_d = (orient_idx_exp - 5) * 0.5  # [batch_size, n_proposal, n_row, prop_width]
                delta_orient_d = self.zero_pad_2d(delta_orient_d)
                delta_orient_roi = torch.zeros_like(corr_idx_pred)
                delta_pred_d = torch.zeros_like(corr_idx_pred)
                delta_pred_d[:, :, 1:] = corr_idx_pred[:, :, 1:] - corr_idx_pred[:, :, :-1]
                for id_b in range(self.b_size):
                    for id_p in range(self.num_prop):
                        prop_min_id = 2 * id_p
                        prop_max_id = 2 * id_p + (self.prop_fea_width)
                        local_orient = delta_orient_d[id_b, :, prop_min_id:prop_max_id]
                        delta_orient_roi[id_b, id_p, :] = local_orient[row_idx, corr_idx_pred[id_b, id_p, :].long()]
                cls_smooth_loss2 += F.smooth_l1_loss(delta_pred_d[vertex_valid], delta_orient_roi[vertex_valid], reduction='sum')
                del orient_idx_exp, delta_pred_d, delta_orient_roi, delta_orient_d
                torch.cuda.empty_cache()
            else:
                cls_smooth_loss2 = torch.tensor(cls_smooth_loss2).cuda()
        else:
            cls_loss2 += -torch.sum(gt_coors[vertex_valid].long() * torch.log(out['cls2'][vertex_valid] + EPS))

        # offset loss
        offset_loss += F.smooth_l1_loss((out['offset2'] * gt_offset_mask),
                                        (gt_offset * gt_offset_mask), reduction='sum')
        

        del gt_proposal, gt_exist, gt_coors, gt_offset, gt_offset_mask, gt_bi_seg
        del invalid_v, proposal_positive, proposal_negative
        torch.cuda.empty_cache()

        if len(orient_exist[0]) > 0:
            orient_loss = self.orient_w * orient_loss / len(orient_exist[0])
        semantic_seg_loss /= (self.row_size * self.row_size * 8 * 8)
        endp_loss = self.endp_loss_w * endp_loss / (self.row_size * self.row_size * self.b_size)  # for Heatmap
        binary_seg_loss = binary_seg_loss / (self.row_size * self.row_size * 8 * self.b_size)
        # proposal_loss = proposal_loss / (proposal_size)
        ext_loss2 = self.ext_w * ext_loss2 / (num_proposal * self.row_size * self.b_size)

        if vertex_valid_size > 0:
            cls_mean_loss2 = self.mean_loss_w * cls_mean_loss2 / vertex_valid_size
            cls_loss2 = self.lambda_cls * cls_loss2/vertex_valid_size
            offset_loss = self.offset_loss_w * offset_loss / vertex_valid_size
            cls_smooth_loss2 = self.cls_smooth_loss_w * cls_smooth_loss2 / vertex_valid_size

        # print(f'ext_loss2 = {ext_loss2}, cls_loss2 = {cls_loss2}, cls_mean_loss2 = {cls_mean_loss2}, '
        #       f'endp_loss = {endp_loss}, '
        #       f'ext_smooth_loss = {ext_smooth_loss}, cls_smooth_loss2={cls_smooth_loss2}')
        # print(f'ext_loss2 = {ext_loss2} , cls_loss2 = {cls_loss2}, cls_mean_loss2 = {cls_mean_loss2}')
        # print(f'endp_loss = {endp_loss}, offset_loss={offset_loss}, bi_seg_loss={binary_seg_loss}, prop_loss={proposal_loss}')
        loss = proposal_loss + ext_loss2 + cls_mean_loss2 + cls_loss2  + cls_smooth_loss2 +\
               endp_loss + orient_loss + binary_seg_loss + offset_loss + semantic_seg_loss

        re_loss = {'loss': loss, 'loss_stats': \
            { 'proposal_loss': proposal_loss,
             'ext_loss2': ext_loss2, 'cls_loss2': cls_loss2, 'cls_mean_loss2': cls_mean_loss2,
              'cls_smooth_loss2': cls_smooth_loss2, 'endp_loss': endp_loss,
             'orient_loss':orient_loss, 'binary_seg_loss':binary_seg_loss, 'offset_loss':offset_loss,\
             'semantic_seg_loss': semantic_seg_loss
             }}

        return re_loss

    def get_exist_coor_endp_dict(self, out):
        # (b, num_proposal) proposal_conf
        # (b, num_proposal, img_h, 3) proposal_ext2
        # (b, num_proposal, img_h, 20) proposal_cls2
        # (b, num_proposal, img_h, 20) proposal_offset2
        # 2 means second stage in forward process

        b_size, num_prop, img_h, prop_w = out['cls2'].shape
        out['proposal_conf'] = out['proposal_conf'].softmax(2).detach().cpu()
        
        ###
        # for ORIENT: BEGIN
        ###
        orient_cls = out['orient'].argmax(1).detach().cpu()
        ###
        # for ORIENT: END
        ###

        ###
        # for binary segmentation: BEGIN
        ###
        # bi_seg = torch.sigmoid(out['bi_seg'][:, 0, :, :]).detach().cpu()
        # print("output['bi_seg'].shape: ", out['bi_seg'].shape)
        out['prop_bi_seg'] = torch.sigmoid(out['prop_bi_seg'])
        # bi_seg = out['bi_seg'].argmax(1).detach().cpu()
        out['semantic_seg'] = out['semantic_seg'].softmax(1).detach().cpu()
        semantic_seg = torch.zeros((out['semantic_seg'].shape[0], out['semantic_seg'].shape[2], out['semantic_seg'].shape[3]))
        semantic_seg[torch.where((out['semantic_seg'][:, 1, :, :] > out['semantic_seg'][:, 2, :, :]) & (out['semantic_seg'][:, 1, :, :] > self.cfg.coor_thre))] = 1
        semantic_seg[torch.where((out['semantic_seg'][:, 2, :, :] > out['semantic_seg'][:, 1, :, :]) & (out['semantic_seg'][:, 2, :, :] > self.cfg.coor_thre))] = 2
        bi_seg_weight_raw = out['semantic_seg'][:, 1, :, :] + out['semantic_seg'][:, 2, :, :]
        bi_seg_weight_raw = torch.squeeze(bi_seg_weight_raw, dim=1)

        ###
        # for binary segmentation: END
        ###

        ###
        # for endpoints: BEGIN
        ###
        org_img_h, org_img_w = img_h * 8, img_h * 8
        arr_endp = torch.zeros((b_size, org_img_h, org_img_w))
        clip_w = 20
        endp_thre = self.cfg.endp_thre
        # print("org h - w: ", org_img_h, org_img_w)
        # print("endp_est shape: ", out['endp_est'].shape)
        for idx_b in range(b_size):
            # temp_endp_score = out[f'endpoint'][idx_b, :, :]
            # temp_endp_score = torch.squeeze(out[f'endpoint'][idx_b, 0, clip_w:(img_h-clip_w), clip_w:(img_w-clip_w)])
            if self.cfg.heads.endp_mode == 'endpoint':
                temp_endp_score = torch.squeeze(out['endpoint'][idx_b, 0, clip_w:(org_img_h - clip_w), clip_w:(org_img_w - clip_w)])
            else:
                temp_endp_score = torch.squeeze(out['endp_est'][idx_b, 0, clip_w:(org_img_h - clip_w), clip_w:(org_img_w - clip_w)])

            temp_endp_score = torch.sigmoid(temp_endp_score)
            temp_endp_score_flat = temp_endp_score.flatten()
            # Whether we need mask here to choose the endpoints in the area where temp_exist is True
            # temp_endp_score_flat = temp_endp_score_flat * (arr_endp_mask.view(-1))
            sorted_endp_score_flat = torch.argsort(temp_endp_score_flat, descending=True)

            # BEGIN: if we get the end points through the clustering
            local_endp_topK = self.num_cls * 2 * 10
            loop_flag = True
            while loop_flag:
                topk_index = sorted_endp_score_flat[:local_endp_topK]  # we need topK
                topk_score = temp_endp_score_flat[topk_index]
                # topk_h, topk_w = topk_index // (org_img_w - 2 * clip_w), topk_index % (org_img_w - 2 * clip_w)  #old version
                topk_h, topk_w = torch.div(topk_index, (org_img_w - 2 * clip_w), rounding_mode='floor'), topk_index % (org_img_w - 2 * clip_w)
                # add clustering method and select 2 clustering centers
                topk_h, topk_w = self.cluster_select_topK_pts(topk_h.detach().cpu().numpy(), topk_w.detach().cpu().numpy(), cluster_r=20, select_K=self.num_cls)
                if (len(topk_h) > 4) or (local_endp_topK > 500):
                    loop_flag = False
                else:
                    local_endp_topK += 10
                    # print("topk_idx: ", topk_index)
            # END: if we get the end points through the clustering
            # BEGIN: NO CLUSTERING
            # select_K = min(int((temp_endp_score_flat >endp_thre).sum().item()), endp_topK)
            # topk_index = sorted_endp_score_flat[:select_K]  # we need topK
            # topk_index_score = temp_endp_score_flat[select_K]
            # topk_h, topk_w = topk_index // (org_img_w - 2*clip_w), topk_index % (org_img_w - 2*clip_w)
            # END: NO CLUSTERING

            topk_h += clip_w
            topk_w += clip_w
            # print("topk_index: ", topk_index)
            # print("topk_h & topk_w: ", topk_h, topk_w)
            arr_endp[idx_b, topk_h.long(), topk_w.long()] = 1
        ###
        # for endpoints: END
        ###


        out['ext2'] = torch.softmax(out['ext2'], dim=3)
        prop_v_ext = torch.zeros((self.b_size, num_prop, self.row_size))
        prop_v_ext[torch.where((out['ext2'][..., 1] > out['ext2'][..., 2]) & (out['ext2'][..., 1] > self.cfg.exist_thre))] = 1
        prop_v_ext[torch.where((out['ext2'][..., 2] > out['ext2'][..., 1]) & (out['ext2'][..., 2] > self.cfg.exist_thre))] = 2
        # prop_ext_conf = 1.0 - out['ext2'][..., 0].detach().cpu()

        local_width = 2
        out['cls2'] = torch.nn.functional.softmax(out['cls2'], dim=-1)
        cls_max_indices = out['cls2'].argmax(-1).cpu()

        all_ind = torch.zeros((b_size, num_prop, img_h, 1+local_width*2))
        cls_max_indices_min = cls_max_indices - local_width
        cls_max_indices_min[torch.where(cls_max_indices_min<0)] = 0
        cls_max_indices_min[torch.where(cls_max_indices_min>(prop_w-local_width-1))] = prop_w-local_width-1
        all_ind[:, :, :, 0] = cls_max_indices_min
        for id_n in range(local_width*2):
            all_ind[:, :, :, id_n+1] = all_ind[:, :, :, id_n] + 1
        # all_ind = torch.tensor(list(range(max(0, (cls_max_indices - local_width)),
        #                                   min((prop_w - 1),
        #                                       (cls_max_indices + local_width + 1)))))  # 5 neighbours expectation
        # select the location neighbourhood with maximumn confidence, and clculate the expectation of the location
        corr_exp = torch.zeros_like(cls_max_indices, dtype=torch.float64)
        corr_offset = torch.zeros_like(cls_max_indices, dtype=torch.float64)
        for b_id in range(b_size):
            for prop_id in range(num_prop):
                for h_id in range(img_h):
                    cls_max_idx = cls_max_indices[b_id, prop_id, h_id]
                    tmp_all_ind = torch.tensor(list(range(max(0, (cls_max_idx - local_width)),
                                          min((prop_w - 1),
                                              (cls_max_idx + local_width + 1)))))  # 5 neighbours expectation
                    tmp_coor_exp_id = (out['cls2'][b_id, prop_id, h_id, tmp_all_ind].softmax(-1).cpu() * tmp_all_ind.float()).sum()
                    corr_exp[b_id, prop_id, h_id] = tmp_coor_exp_id
                    corr_offset[b_id, prop_id, h_id] = cls_max_idx + out['offset2'][b_id, prop_id, h_id, cls_max_idx]
        # corr_exp = (out['cls2'][:, :, :, all_ind.long()].softmax(-1).cpu() * all_ind.float()).sum()
        corr_idx = cls_max_indices.to(torch.float64)
        corr_idx[torch.where(corr_idx>prop_w)] = prop_w
        corr_exp[torch.where(corr_exp>prop_w)] = prop_w
        # corr_offset = corr_idx + out['offset2'][corr_idx]
        corr_offset[torch.where(corr_offset>prop_w)] = prop_w

        # add the proposal offset for each predicted lines
        for id_p in range(num_prop):
            corr_idx[:, id_p, :] += (self.prop_width*id_p - self.prop_half_buff)
            corr_exp[:, id_p, :] += (self.prop_width*id_p - self.prop_half_buff)
            corr_offset[:, id_p, :] += (self.prop_width*id_p - self.prop_half_buff)
            # print("proposal id: ", id_p)
            # print("predict coordinate: ", corr_idx[:,id_p,:])
            # print("predict coordinate expectation: ", corr_exp[:, id_p, :])


        # arr_conf[np.where(arr_cls[:,idx_bg,:,:]==0.)] = 1.  # confidence of where exist a lane
        # dict_ret = {'conf': torch.tensor(arr_conf), 'cls': torch.tensor(arr_cls), 'endp': torch.tensor(arr_endp), 'offset':torch.tensor(arr_endp_off)}
        if self.cfg.view_detail:
            dict_ret = {'prop_conf': out['proposal_conf'],
                    'prop_v_ext': prop_v_ext, 'prop_cls_conf': out['cls2'],
                    'endp': arr_endp, 'orient':orient_cls, 'bi_seg':bi_seg_weight_raw, 'semantic_seg':semantic_seg,
                    'cls': corr_idx, 'cls_exp': corr_exp, \
                    'cls_offset': corr_offset}
        else:
            dict_ret = {'prop_conf': out['proposal_conf'],
                        'prop_v_ext': prop_v_ext, 'prop_cls_conf': out['cls2'],
                        'endp': arr_endp, 'orient': orient_cls, 'bi_seg': bi_seg_weight_raw,
                        'semantic_seg': semantic_seg,
                        'cls_offset': corr_offset}

        return dict_ret
    
    def get_lane_map_numpy_with_label(self, output, data, is_flip=True, is_img=False, is_get_1_stage_result=False, is_gt_avai=True):
        '''
        * in : output feature map
        * out: lane map with class or confidence
        *       per batch
        *       ### Label ###
        *       'cls_exist': (n_lane, 144) / 0(not exist), 1(exist)
        *       'cls_label': (n_lane, 144) / col coordinate at the original resolution
        *       ### Raw Prediction ###
        *       'prop_conf': (batch, n_proposal, 2)
        *       'prop_v_ext': (Batch, n_proposal, 144) / 0 ~ 1
        *       'pro_cls_conf': (Batch, n_proposal, 144, 20) / 0 ~ 1 (softmax)
        *       'cls': (Batch, n_proposal, 144) / 0. ~ 144.
        *       'cls_exp': (Batch, n_proposal, 144) / 0. ~ 144.
        *       'cls_offset': (Batch, n_proposal, 144) / 0. ~ 144.
        *       'endp': (Batch, n_lane, 4) / [endp1_h, endp1_w, endp2_h, endp2_w]
        *       ### Modified Prediction ### modification includes NMS and line merging
        *       'm_cls': [Batch, n_lines, 144, 2] / 0. ~ 1152.
        *       'm_cls_exp': (Batch, n_lines, 144, 2) / 0. ~ 1152.
        *       'm_cls_offset': (Batch, n_lines, 144, 2) / 0. ~ 1152.
        '''
        data_mode = self.cfg.dataset_type #'LaserLane' # 'LaserLane' (background=0) or 'KLane' (background=255)
        lane_maps = dict()

        # for batch
        list_conf_label_raw = [] # (1152, 1152), 0, 1
        list_coor_label = []   # (n_lane, 144): coordinate(-1, 0~1152)

        list_cls_coor = []    # (n_lane, 144): coordinate(-1, 0~1152)
        list_cls_coor_smooth = []  # (n_lane, 144): coordinate(-1, 0~1152)
        list_cls_exp_smooth = []  # (n_lane, 144): coordinate(-1, 0~1152)
        list_cls_offset_smooth = []  # (n_lane, 144): coordinate(-1, 0~1152)

        list_endp_pred = []
        list_semantic_line_pred = []

        batch_size, num_prop, num_hv, num_wc = output['prop_cls_conf'].shape
        if is_gt_avai:
            coor_label_all = data['lc_coor_raw']
            for batch_idx in range(batch_size):
                # _, coor_label, _, _ = self.get_line_existence_and_cls_wise_maps_per_batch(raw_label[batch_idx], n_cls=self.num_cls, downsample=False)
                coor_label = coor_label_all[batch_idx, ...].cpu().numpy()
                list_coor_label.append(coor_label)
                
        for batch_idx in range(batch_size):
            # 144, 144
            prop_conf = output['prop_conf'][batch_idx, :, 1]  # channel-1 is the proposal existing confidence
            prop_exist_v = output['prop_v_ext'][batch_idx, :, :].cpu().numpy() # (n_proposal, row_num)
            prop_cls_conf_v = output['prop_cls_conf'][batch_idx, :, :, :].cpu().numpy()   # (n_proposal, row_num, 20)
            
            # whether exist the lane in the proposal
            prop_no_exist = torch.where(prop_conf < self.cfg.proposal_obj_thre)[0]
            prop_exist_v[prop_no_exist, :] = 0.
            # filter the predicted lines in the several first and last columns:
            prop_exist_v[0:4, :] = 0.  # first four
            prop_exist_v[-6:, :] = 0.  # last six

            # n_lane, 144
            v_exist_pred = np.where(prop_exist_v > 0.5, prop_exist_v, -1)
            # exist_pred = torch.where(exist_pred>self.cfg.exist_thr, 1, -1).cpu().numpy()
            if self.cfg.view_detail:
                cls_coor_pred = output['cls'][batch_idx].cpu().numpy() / self.row_size * 1152 + 4
                cls_coor_pred = np.where(v_exist_pred==(-1), -1, cls_coor_pred)
                cls_coor_exp = output['cls_exp'][batch_idx].cpu().numpy() / self.row_size * 1152
                cls_coor_exp = np.where(v_exist_pred==(-1), -1, cls_coor_exp)
                cls_coor_pred[np.where(cls_coor_pred < 0)] = 0
                cls_coor_pred[np.where(cls_coor_pred > 1151)] = 1151
                cls_coor_exp[np.where(cls_coor_exp < 0)] = 0
                cls_coor_exp[np.where(cls_coor_exp > 1151)] = 1151
            cls_coor_offset = output['cls_offset'][batch_idx].cpu().numpy() / self.row_size * 1152
            cls_coor_offset = np.where(v_exist_pred==(-1), -1, cls_coor_offset)
            cls_coor_offset[np.where(cls_coor_offset < 0)] = 0
            cls_coor_offset[np.where(cls_coor_offset > 1151)] = 1151
            semantic_line_map = np.zeros((1152, 1152))  # detected points location
            for idx_line in range(self.num_prop):
                ph_idx = np.where(cls_coor_offset[idx_line, :] > 0)
                semantic_line_map[ph_idx[0] * 8 + 3, (cls_coor_offset[idx_line, ph_idx[0]]).astype(int)] = v_exist_pred[idx_line, ph_idx[0]]


            # smooth n_lane coordinates:
            if self.cfg.view_detail:
                cls_coor_pred_smooth = smooth_cls_line_per_batch(cls_coor_pred, output['orient'][batch_idx].cpu().numpy(), complete_inner_nodes=True, out_seg_conf=output['bi_seg'][batch_idx].cpu().numpy())
                cls_coor_exp_smooth = smooth_cls_line_per_batch(cls_coor_exp, output['orient'][batch_idx].cpu().numpy(), complete_inner_nodes=True, out_seg_conf=output['bi_seg'][batch_idx].cpu().numpy())
                cls_coor_pred_smooth = polyline_NMS(cls_coor_pred_smooth, output['bi_seg'][batch_idx].cpu().numpy())
                cls_coor_exp_smooth = polyline_NMS(cls_coor_exp_smooth, output['bi_seg'][batch_idx].cpu().numpy())

            cls_coor_offset_smooth = smooth_cls_line_per_batch(cls_coor_offset, output['orient'][batch_idx].cpu().numpy(), complete_inner_nodes=True, out_seg_conf=output['bi_seg'][batch_idx].cpu().numpy())
            cls_coor_offset_smooth = polyline_NMS2(cls_coor_offset_smooth, output['bi_seg'][batch_idx].cpu().numpy())

            # for endpoint
            endp_pred_raw = output['endp'][batch_idx].cpu().numpy()
            # for semantic segmentation
            # semantic_seg_raw = torch.squeeze(output['semantic_seg'][batch_idx, :, :]).numpy()
            semantic_lines_map, line_vertex_semantic = self.get_pred_semantic_lane_coordinates(cls_coor_offset_smooth, semantic_line_map)
            cls_coor_offset_smooth = np.concatenate([np.expand_dims(cls_coor_offset_smooth, axis=2), np.expand_dims(line_vertex_semantic, axis=2)], axis=2)
            cls_coor_offset_smooth, endp_pred_raw = polyline_uniform_semantics_by_statistics(cls_coor_offset_smooth, endp_pred_raw, r_buff=20)
            cls_coor_offset_smooth = remove_short_polyline(cls_coor_offset_smooth, min_v_count=8)
            semantic_line_map_renew = renew_semantic_map(cls_coor_offset_smooth)
            list_cls_offset_smooth.append(cls_coor_offset_smooth)
            list_endp_pred.append(endp_pred_raw)
            list_semantic_line_pred.append(semantic_line_map_renew)

            if self.cfg.view_detail:
                list_cls_coor.append(cls_coor_pred)
                list_cls_coor_smooth.append(cls_coor_pred_smooth)
                list_cls_exp_smooth.append(cls_coor_exp_smooth)


        if self.cfg.view_detail:
            lane_maps.update({
                'coor_label': list_coor_label,
                'cls_coor_pred': list_cls_coor,
                'cls_coor_pred_smooth': list_cls_coor_smooth,
                'cls_exp_smooth': list_cls_exp_smooth,
                'cls_offset_smooth': list_cls_offset_smooth,
                'endp_by_cls': list_endp_pred,
                'semantic_line': list_semantic_line_pred
            })
        else:
            lane_maps.update({
                'coor_label': list_coor_label,
                'cls_offset_smooth': list_cls_offset_smooth,
                'endp_by_cls': list_endp_pred,
                'semantic_line': list_semantic_line_pred
            })
        return lane_maps


    def modify_topology(self, lane_vertex):
        num_l, num_v = lane_vertex.shape
        lane_vertex_copy = self.sort_lines(lane_vertex)

        # check vertexes row by row: keep the column values increasing with the lane id
        for v_id in range(num_v):
            exist_l = np.where(lane_vertex_copy[:, v_id] > 0)
            exist_value = lane_vertex_copy[exist_l, v_id]
            exist_value_sorted = np.sort(exist_value)
            lane_vertex_copy[exist_l, v_id] = exist_value_sorted


        return lane_vertex_copy

    def cluster_select_topK_pts(self, pts_h, pts_w, cluster_r = 4, select_K=2):
        X = np.concatenate((pts_h.reshape((len(pts_h), 1)), pts_w.reshape((len(pts_w), 1))), axis=1)
        # print("X.shape", X.shape)
        clustering = DBSCAN(eps=cluster_r, min_samples=1, metric="euclidean").fit(X)
        labels_idx = clustering.labels_
        cluster_labels, cluster_sizes = np.unique(labels_idx,return_counts=True)
        centroid_labels = np.zeros((len(cluster_labels), X.shape[1]))
        nearest_sample_2_centroid = np.zeros_like(centroid_labels)

        for id, label in enumerate(cluster_labels):
            label_id_values = X[np.where(labels_idx == label)[0], :]
            label_center = np.mean(label_id_values, axis=0)
            centroid_labels[id, :] = label_center

            knearest = NearestNeighbors().fit(label_id_values)
            n_dist, n_idx = knearest.kneighbors([label_center], n_neighbors=1)
            nearest_sample_2_centroid[id, :] = label_id_values[n_idx[0], :]
        cluster_size_sorted = np.argsort(cluster_sizes)
        # choose the nearest pt or cluster center?
        center_h = [nearest_sample_2_centroid[k][0] for k in cluster_size_sorted]
        center_w = [nearest_sample_2_centroid[k][1] for k in cluster_size_sorted]
        return torch.tensor(center_h), torch.tensor(center_w)

    def get_lane_map_on_source_image(self, output, data, is_img=True):
        '''
        * in : output feature map
        * out: lane map with class or confidence
        *       per batch
        *       ### Classification ###
        *       'cls_idx': (144, 144) / 0, 1, 2, 3, 4, 5(lane), 255(ground)
        *       'conf_cls_idx': (144, 144) / (get cls idx in conf true positive)
        *       'list_endp_pred': (144, 144) / 0, 1, 3 ,4,5(lane), 255 (ground)
        *       ### RGB Image ###
        *       'rgb_img_cls_label': (144, 144, 3)
        *       'rgb_img_cls_idx': (144, 144, 3)
        *       'rgb_img_conf_cls_coors': (1152, 1152, 3)
        '''
        data_mode = self.cfg.dataset_type  # 'LaserLane' # 'LaserLane' (background=0) or 'KLane' (background=255)
        number_lanes = self.num_cls
        lane_maps = dict()
        list_source_img = []
        list_gt_on_img = []
        list_lanes_on_img = []
        list_org_lanes_on_img = []
        list_org_lanes_on_img_smooth = []
        list_org_lanes_on_img_exp = []
        list_org_lanes_on_img_offset = []
        list_org_lanes_smooth_vertex = []
        list_org_lanes_binary_seg = []

        # for batch
        batch_size, prop_size, _ = output['prop_conf'].shape
        for batch_idx in range(batch_size):
            raw_image = data['proj'][batch_idx].cpu().numpy()
            raw_image_gray = cv2.cvtColor(raw_image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            raw_image_gray = cv2.cvtColor(raw_image_gray, cv2.COLOR_GRAY2RGB)
            raw_image_gray *= 255
            if self.cfg.is_gt_avai:
                # gt_endp_init = (data['initp'][batch_idx] + data['init_offset'][batch_idx]).cpu().numpy() * 8
                # gt_endp_end = (data['endp'][batch_idx] + data['end_offset'][batch_idx]).cpu().numpy() * 8
                gt_endp_init = (data['initp'][batch_idx]).cpu().numpy()
                gt_endp_end = (data['endp'][batch_idx]).cpu().numpy()
                gt_endp_coors = [[gt_endp_init[lane_id], gt_endp_end[lane_id]] for lane_id in range(self.num_cls)]
            # raw_image_gray = get_gt_endp_on_raw_image(gt_endp_coors, raw_image_gray) # NOT SHOW GT ENDPOINTS
            raw_image_gray_gt = raw_image_gray.copy()
            raw_image_gray_bi_seg = raw_image_gray.copy()
            raw_image_gray_coor_offset = raw_image_gray.copy()

            if self.cfg.view_detail:
                raw_image_gray_coor = raw_image_gray.copy()
                raw_image_gray_coor_smooth = raw_image_gray.copy()
                raw_image_gray_coor_exp = raw_image_gray.copy()
                # upsample lane
                # discrete_lane = output['lane_maps']['cls_idx'][batch_idx]
                # lane_coors = [np.where(discrete_lane == lane_id) for lane_id in range(self.num_cls)]
                # lane_coors = np.array(lane_coors)
                # lane_coors = lane_coors * 8 + 4   # upsample coordinate

                # predicted lane coordinates:
                pred_lane_coors = np.zeros((prop_size, self.row_size, 2))
                pred_lane_coors[:, :, 0] = np.arange(3, 1152, 8)
                pred_lane_coors[:, :, 1] = output['lane_maps']['cls_coor_pred'][batch_idx]

                # predicted lane coordinates after smoothing:
                pred_lane_coors_smooth = np.zeros((prop_size, self.row_size, 2))
                pred_lane_coors_smooth[:, :, 0] = np.arange(3, 1152, 8)
                pred_lane_coors_smooth[:, :, 1] = output['lane_maps']['cls_coor_pred_smooth'][batch_idx]

                # coordinates expectations:
                pred_lane_coors_exp = np.zeros((prop_size, self.row_size, 2))
                pred_lane_coors_exp[:, :, 0] = np.arange(3, 1152, 8)
                pred_lane_coors_exp[:, :, 1] = output['lane_maps']['cls_exp_smooth'][batch_idx]

            # coordinates with offsets:
            pred_lane_coors_offset = np.zeros((prop_size, self.row_size, 3))
            pred_lane_coors_offset[:, :, 0] = np.arange(3, 1152, 8)
            pred_lane_coors_offset[:, :, 1] = output['lane_maps']['cls_offset_smooth'][batch_idx][:, :, 0]
            pred_lane_coors_offset[:, :, 2] = output['lane_maps']['cls_offset_smooth'][batch_idx][:, :, 1]
            
            # for ground truth
            if self.cfg.is_gt_avai:
                gt_lane_coors = np.zeros((self.cfg.number_lanes, self.row_size, 2))
                gt_lane_coors[:, :, 0] = np.arange(3, 1152, 8)
                gt_lane_coors[:, :, 1] = output['lane_maps']['coor_label'][batch_idx]
                for lane_id in range(self.cfg.number_lanes): 
                    lane_coor = gt_lane_coors[lane_id, ...]
                    lane_coor = lane_coor[np.where(lane_coor[:, 1] > 0)]
                    raw_image_gray_gt = get_lane_on_raw_image_coordinates(lane_coor, lane_id, raw_image_gray_gt)
           
            # upsample endpoint
            discrete_endp = output['lane_maps']['endp_by_cls'][batch_idx]
            endp_coors = np.where(discrete_endp == 1 )
            endp_coors = np.array(endp_coors)

            # binary segmentation result
            bi_seg_map = output['lane_maps']['semantic_line'][batch_idx]
            for semantic_id in [1, 2]:
                for lane_id in range(prop_size):
                    # for predicted lane with offset:
                    pred_lane_coor = pred_lane_coors_offset[lane_id, :, :2]
                    pred_lane_coor = pred_lane_coor[np.where(((pred_lane_coors_offset[lane_id, :, 2] == semantic_id) & \
                                                             (pred_lane_coors_offset[lane_id, :, 1] > 0)))]
                    raw_image_gray_bi_seg = get_semantic_lane_on_raw_image_coordinates(pred_lane_coor, semantic_id,
                                                                                        raw_image_gray_bi_seg)

            

            for lane_id in range(prop_size):
                # for predicted lane with offset:
                pred_lane_offset = pred_lane_coors_offset[lane_id, :, :2]
                pred_lane_offset = pred_lane_offset[np.where(pred_lane_offset[:, 1] > 0)]
                raw_image_gray_coor_offset = get_lane_on_raw_image_coordinates(pred_lane_offset, lane_id,
                                                                                    raw_image_gray_coor_offset)
                
                if self.cfg.view_detail:
                    # for predicted lane coordinates:
                    pred_lane = pred_lane_coors[lane_id, :, :]
                    pred_lane = pred_lane[np.where(pred_lane[:, 1] > 0)]
                    raw_image_gray_coor = get_lane_on_raw_image_coordinates(pred_lane, lane_id, raw_image_gray_coor)

                    pred_lane_smooth = pred_lane_coors_smooth[lane_id, :, :]
                    pred_lane_smooth = pred_lane_smooth[np.where(pred_lane_smooth[:, 1] > 0)]
                    raw_image_gray_coor_smooth = get_lane_on_raw_image_coordinates(pred_lane_smooth, lane_id,
                                                                                        raw_image_gray_coor_smooth)
                    # for predicted lane expectation:
                    pred_lane_exp = pred_lane_coors_exp[lane_id, :, :]
                    pred_lane_exp = pred_lane_exp[np.where(pred_lane_exp[:, 1] > 0)]
                    raw_image_gray_coor_exp = get_lane_on_raw_image_coordinates(pred_lane_exp, lane_id,
                                                                                     raw_image_gray_coor_exp)

            list_source_img.append(raw_image_gray)
            if self.cfg.is_gt_avai:
                list_gt_on_img.append(raw_image_gray_gt)
            list_org_lanes_smooth_vertex.append(pred_lane_coors_offset)
            list_org_lanes_binary_seg.append(raw_image_gray_bi_seg)
            list_org_lanes_on_img_offset.append(raw_image_gray_coor_offset)
            lane_maps.update({
                'source_img_gray': list_source_img,
                'gt_on_img': list_gt_on_img,
                'pred_smooth_lane_vertex': list_org_lanes_smooth_vertex,
                'pred_bi_seg_on_image': list_org_lanes_binary_seg,
                'pred_offset_lanes_on_image': list_org_lanes_on_img_offset
            })

            if self.cfg.view_detail:
                list_lanes_on_img.append(raw_image_gray)
                list_org_lanes_on_img.append(raw_image_gray_coor)
                list_org_lanes_on_img_smooth.append(raw_image_gray_coor_smooth)
                # list_org_lanes_smooth_vertex.append(pred_lane_coors_smooth)
                list_org_lanes_on_img_exp.append(raw_image_gray_coor_exp)
                lane_maps.update({
                    'pred_lanes_on_image': list_lanes_on_img,
                    'pred_org_lanes_on_image': list_org_lanes_on_img,
                    'pred_smooth_lanes_on_image': list_org_lanes_on_img_smooth,
                    'pred_smooth_lane_vertex': list_org_lanes_smooth_vertex,
                    'pred_bi_seg_on_image': list_org_lanes_binary_seg,
                    'pred_exp_lanes_on_image': list_org_lanes_on_img_exp,
                    'pred_offset_lanes_on_image': list_org_lanes_on_img_offset
                })

        return lane_maps
    
    '''
    @param_in: discrete_lane_coors: (n_lane, n_vertex)
    @param_in: raw_semantic_lanes: (1152, 1152)
    @return_1: semantic map: 0(background), 1(continuous line), 2(dashed line)
    @return_2: semantic coordinates: vertex coordinates and corresponding semantics
    '''
    def get_pred_semantic_lane_coordinates(self, discrete_lane_coors, raw_semantic_lanes):
        num_lane, num_vertex = discrete_lane_coors.shape
        semantic_lane_img = np.zeros((1152, 1152))
        discrete_lane_semantics = np.zeros_like(discrete_lane_coors)

        for id_lane in range(num_lane):
            for p_idx in range(num_vertex - 1) :  # shape: N, 2
                pt1_y = int(discrete_lane_coors[id_lane][p_idx])
                pt2_y = int(discrete_lane_coors[id_lane][p_idx + 1])
                if (pt1_y < 0) or (pt2_y < 0):
                    continue
                else: 
                    pt1 = (pt1_y, int(p_idx*8 +3))
                    pt2 = (pt2_y, int((p_idx + 1) *8 +3))
                    if (raw_semantic_lanes[int(p_idx*8 +3), pt1_y] == 2) or (raw_semantic_lanes[int((p_idx+1)*8 +3), pt2_y] == 2):
                        color = 2  # dashed line
                    else:
                        color = 1  # solid line

                    cv2.line(semantic_lane_img, pt1, pt2, color=color, thickness=1)
                    discrete_lane_semantics[id_lane, p_idx] = color
                    if (p_idx == (num_vertex - 2)) and (pt2_y > 0):
                        discrete_lane_semantics[id_lane, p_idx + 1] = color

        return semantic_lane_img, discrete_lane_semantics