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


from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding   # add by Xiaoxin
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

@HEADS.register_module
class RowSharNotReducRef_Base(nn.Module):
    def __init__(self,
                dim_feat=8, # input feat channels
                row_size=144,
                dim_shared=512, # row_size * dim_feat ? expand the rows
                thr_ext = 0.3,
                num_prop = 72,
                prop_width=2,
                prop_half_buff = 4,
                dim_token = 1024,  # what is the meaning of this parameter, down-dimesion
                tr_depth = 1,      # parameter for transformer
                tr_heads = 16,     # parameter for transformer ?
                tr_dim_head = 64,
                tr_mlp_dim = 2048,
                tr_dropout = 0.,
                tr_emb_dropout = 0.,
                row_dim_token = 64,
                row_tr_depth=1,  # parameter for transformer
                row_tr_heads=10,  # parameter for transformer ?
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
        super(RowSharNotReducRef_Base, self).__init__()
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
        # self.fc_reg = Linear(self.img_embed_dims, 4) # regress 4 parameters: endp1_h, endp1_w, endp2_h, endp2_w
        self.fc_reg = nn.Sequential(
            nn.Linear(144*144, 144),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(144, 4),
            nn.Sigmoid()
        )

        # First stage: Proposal generation
        # Feature map from 8 * 144 * 144 to 16 * 72 * 72
        # Every proposal has the receiption field 5 * 5
        # Proposal shape is: 72 * 5
        # Proposal amount is: 72 along column axis
        self.generate_line_proposal = nn.Sequential(
            nn.Conv2d(dim_feat, 2*dim_feat, kernel_size=(5, 3), stride=1, padding=(2, 1)),
            # nn.ReLU(),
            nn.BatchNorm2d(2*dim_feat),
            nn.Conv2d(2*dim_feat, 2*dim_feat, kernel_size=3, stride=2, padding=1)  # downsample: compared to pooling, conv+stride can keep more details
        )
        # if self.num_prop == 72:
        #     self.generate_line_proposal = nn.Sequential(Conv_Pool_2d(dim_feat, hidden_dims=[], output_dim=2*dim_feat)).cuda()
        # if self.num_prop == 36:
        #     self.generate_line_proposal = nn.Sequential(Conv_Pool_2d(dim_feat, hidden_dims=[2*dim_feat], output_dim=4*dim_feat)).cuda()
        # if self.num_prop == 18:
        #     self.generate_line_proposal = nn.Sequential(Conv_Pool_2d(dim_feat, hidden_dims=[2*dim_feat, 4*dim_feat], output_dim=8*dim_feat)).cuda()


        # Followed up with proposal attention module
        self.thr_ext = thr_ext

        in_token_channel = 1* self.num_prop *dim_feat* self.prop_width
        self.to_token = nn.Sequential(
            Rearrange('c h -> (c h)'),  # w=1
            nn.Linear(in_token_channel, dim_token)
        )

        # old version for position embedding
        for idx_prop in range(self.num_prop):
            setattr(self, f'emb_{idx_prop}', nn.Parameter(torch.randn(dim_token)).cuda())
        # new version for position embedding
        col_token_scale = dim_token ** (-0.5)
        col_token_pos_emb = nn.Parameter(torch.randn(self.num_prop, dim_token) * col_token_scale).cuda()

        self.emb_dropout = None
        if tr_emb_dropout != 0.:
            self.emb_dropout = nn.Dropout(tr_emb_dropout)
        self.tr_lane_correlator = nn.Sequential(
            Transformer(dim_token, tr_depth, tr_heads, tr_dim_head, tr_mlp_dim, tr_dropout),
            nn.LayerNorm(dim_token)
        )

        # Followed up with the line proposal ranking: Loss: nearest line distance
        # input is the result of the proposal token
        # output: the confidence of each proposal: positive or negative
        self.proposal_confidence = nn.Sequential(
            nn.Linear(dim_token, 2)
        )

        self.line_expand = nn.Sequential(
            nn.Linear(dim_token, in_token_channel),
            Rearrange('b n (c h w) -> b n c h w', c=dim_feat*2, h=72)
        )

        # Followed up with the line regression, downsample part  ### Refinement (2nd Stage) ###
        self.head_common_layers = nn.Sequential( # resolution: 288 * 288 -> 144 * 144; dim: 16 -> 8
            nn.Conv2d(dim_feat*3, dim_feat*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_feat*2),
            nn.Conv2d(dim_feat*2, dim_feat, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_feat)
        )

        self.to_token_row_seg_attention = nn.Sequential(
            Rearrange('b c h w -> b h (c w)'),
            nn.Linear(dim_feat * self.prop_fea_width, row_dim_token),
            Rearrange('b h c -> b c h')
        )

        # Followed up with the row self attention to consruct the relationship among rows in a proposal RoI
        self.row_emb_dropout = None
        if row_tr_emb_dropout != 0.:
            self.row_emb_dropout = nn.Dropout(row_tr_emb_dropout)

        # old row position embedding
        self.to_token_row = nn.Sequential(
            Rearrange('b c w -> b (c w)'),  # h=1
            nn.Linear(dim_feat * self.prop_fea_width, row_dim_token)
        )
        for idx_row in range(self.row_size):
            setattr(self, f'emb_row_{idx_row}', nn.Parameter(torch.randn(row_dim_token)).cuda())
        self.tr_row_correlator = nn.Sequential(
            Transformer(row_dim_token, tr_depth, tr_heads, tr_dim_head, tr_mlp_dim, tr_dropout),
            nn.LayerNorm(row_dim_token),
            Rearrange('b h c -> b c h')
        )


        # new row position embedding
        # self.to_token_row_all = nn.Sequential(
        #     Rearrange('b c h w -> b h (c w)'),  # h=1
        #     nn.Linear(dim_feat * self.prop_width, row_dim_token),
        #     Rearrange('b h (s t) ->b s h t', s = self.tr_row_heads)
        # )
        # self.row_pos_emb = nn.Sequential(
        #     RelPosEmb1DAISummer(tokens=self.row_size, dim_head=row_tr_dim_head, heads=row_tr_heads),
        #     Rearrange('b s h t -> b h (s t)'))
        # self.tr_row_correlator = nn.Sequential(
        #     Transformer(dim=self.row_size*row_tr_heads, depth=row_tr_depth, heads=row_tr_heads, dim_head=self.row_size, mlp_dim=row_tr_mlp_dim, dropout=row_tr_dropout),
        #     nn.LayerNorm(self.row_size*row_tr_heads),
        #     nn.Linear(self.row_size*row_tr_heads, row_dim_token),
        #     Rearrange('b h c -> b c h')
        # )

        # without neighbouring rows constraints, we use Conv1d to predict every row
        # with neighbouring rows constraints, we use both conv2d and conv1d

        self.ext2 = nn.Sequential(
            nn.Conv1d(row_dim_token, dim_shared, 1, 1, 0),
            nn.BatchNorm1d(dim_shared),
            nn.Conv1d(dim_shared, 3, 1, 1, 0),
            Rearrange('b c h -> b h c')
        )

        self.cls2 = nn.Sequential(
            nn.Conv1d(row_dim_token, dim_shared, 1, 1, 0),
            nn.BatchNorm1d(dim_shared),
            nn.Conv1d(dim_shared, self.prop_fea_width, 1, 1, 0),
            Rearrange('b w h -> b h w')
        )

        self.offset2 = nn.Sequential(
            nn.Conv1d(row_dim_token, dim_shared, 1, 1, 0),
            nn.BatchNorm1d(dim_shared),
            nn.Conv1d(dim_shared, self.prop_fea_width, 1, 1, 0),
            Rearrange('b w h -> b h w')
        )

        # input: 144 * 144
        # output: 144 * 144
        self.orient = nn.Sequential(
            nn.Conv2d(dim_feat, int(dim_feat / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(dim_feat / 2)),
            nn.Conv2d(int(dim_feat / 2), self.num_orients, 3, 1, 1)
        )

        # Upsample: Per proposal line segmentation
        self.head_upsample_layers = nn.Sequential(  # resolution: 288 * 288 -> 288 * 288; dim: 24 -> 8
            nn.Conv2d(dim_feat * 3, dim_feat * 2, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
            nn.BatchNorm2d(dim_feat * 2),
            nn.Conv2d(dim_feat * 2, dim_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_feat)
        )

        self.bi_seg_proposal = nn.Conv2d(dim_feat, 1, kernel_size=1, stride=1, padding=0)
        # self.endp_proposal = nn.Sequential(
        #     nn.Conv2d(dim_feat, int(dim_feat * 0.5), kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(int(dim_feat * 0.5), int(dim_feat * 0.5)),
        #     nn.Conv2d(int(dim_feat * 0.5), 1, kernel_size=1, stride=1, padding=0)
        # )


        # input: 288 * 288
        # output: 1152 * 1152
        self.endpoint = nn.Sequential(
            nn.Conv2d(dim_feat, int(dim_feat / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(dim_feat / 2)),
            nn.ReLU(inplace=True),
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


    def forward(self, x, x_up):
        out_dict = dict()
        b_size, dim_feat, img_h, img_w = x.shape  # 4, 8, 144, 144
        self.b_size = b_size
        num_cls = self.num_cls

        row_feat = x # self.botn_layer(x)
        # print("feature shape after CNN feature extraction: ", row_feat.shape)  #torch.Size([4, 8, 144, 144])
        # row_feat_up = self._upsample_add(row_feat, x_up)

        # First stage: Proposal generation
        # Feature map from 144 * 144 to 72 * 72
        # Every proposal has the receiption field 5 * 5
        # Proposal shape is: 72 * 5
        # Proposal amount is: 72 along column axis
        feat_down = self.generate_line_proposal(row_feat) # 4 * 8 * 72 * 72
        _, feat_down_dim, feat_down_H, feat_down_W = feat_down.shape
        ### Finish: 1st Stage Processing ###


        # Second stage: Proposal attention Module
        proposal_confs = []
        col_feats_batch = []

        for idx_b in range(b_size):
            ext_lane_tokens = []
            for idx_w in range(feat_down_W): # 72
                temp_token = feat_down[idx_b, :, :, idx_w] # batch , channel, height, column
                temp_token0 = self.to_token(temp_token) + getattr(self, f'emb_{idx_w}')  # (256,)
                # expand because of unsqueeze backward error
                ext_lane_tokens.append((temp_token0.unsqueeze(0)).unsqueeze(0))  # (n_proposals72, 1, 1, 256)
            token_before = torch.cat(ext_lane_tokens, dim=1)
            tokens = self.tr_lane_correlator(token_before)  # (1, n_proposal72, (fea_dim8, 72, 1))
            prop_conf = self.proposal_confidence(tokens) #(1,72,2)
            proposal_confs.append(prop_conf)

            column_fea = self.line_expand(tokens)
            tmp_fea_down = torch.squeeze(column_fea.squeeze(dim=0), dim=-1)
            col_feats_batch.append(tmp_fea_down.permute(1, 2, 0).unsqueeze(dim=0))

        # # print("after row shared stage, the row feature shape is: ", row_feat.shape)  # [4, 8, 144, 144]
        proposal_confs_out = torch.cat(proposal_confs, dim=0).squeeze(dim=2)  # [4, 72, 2]
        col_fea_att = torch.cat(col_feats_batch, dim=0)   # column feature after column attention: [4, 16, 72, 72]
        ### Finish: 2nd Stage Processing ###

        # Local and global feature concatenation
        con_feat_up = self._upsample_cat(col_fea_att, x_up)  # feat_down: [4, 24, 288, 288]

        ### Third stage: positive proposal selection, line regression
        row_feat_up = self.head_common_layers(con_feat_up) # input: [4, 24, 288, 288]; output: [4, 8, 144, 144]
        propoal_fea_up = self.head_upsample_layers(con_feat_up)  # input: [4, 24, 288, 288]; output: [4, 8, 288, 288]

        total_iter_round = 1
        tmp_iter = 0
        while tmp_iter < total_iter_round:
            tmp_iter += 1
            # 3-1: attributes aux
            # (1) semantic segmentation; (2) attribute points detection
            # out_dict.update({'endpoint': torch.sigmoid(self.endpoint(row_feat))})
            out_dict.update({'endpoint': self._upsample(self.endpoint(F.relu(row_feat_up)), img_h * 8,
                                           img_w * 8)})

            # 3-2: Lane regression
            # (1) orient estimation; (2) existence estimation; (3) column coordinates estimation; (4) offset regression
            out_dict.update({'orient': self.orient(row_feat_up)})


            row_feat_up = self.zero_pad_2d(row_feat_up)
            propoal_fea_up = self.zero_pad_2d_prop(propoal_fea_up)

            proposal_ext2 = []
            proposal_cls2 = []
            proposal_offset2 = []
            proposal_bi_seg = []

            for id in range(feat_down_W):  # proposal size: 72
                # inter-row self attention
                local_prop_fea = row_feat_up[:, :, :, self.N_s*id:(self.N_s*id+self.prop_fea_width)]   # plus 2, 2 means the field of proposal on this feature map
                upsample_prop_fea = propoal_fea_up[:, :, :, 2*self.N_s*id:(2*self.N_s*id+2*self.prop_fea_width)]  # the feature map size is 2 times of local one

                # for upsampled output:
                sp_bi_seg = self._upsample(self.bi_seg_proposal(F.relu(upsample_prop_fea)), img_h * 8,
                                           self.prop_fea_width * 8)

                # segmentation attention version
                tokens_before = self._downsample_multiply(sp_bi_seg, local_prop_fea)
                tokens_after = self.to_token_row_seg_attention(tokens_before)


                # old version for row attention
                # row_tokens = []
                # for idx_h in range(self.row_size):  # 144
                #     temp_token = local_prop_fea[:, :, idx_h, :]  # batch , channel, height, column
                #     temp_token0 = self.to_token_row(temp_token) + getattr(self, f'emb_row_{idx_h}')  # (256,)
                #     # expand because of unsqueeze backward error
                #     row_tokens.append(temp_token0.unsqueeze(1))  # [n_row, b, 256]
                # token_before = torch.cat(row_tokens, dim=1)     # [b, h, c]

                # new version
                # row_tokens = self.to_token_row_all(local_prop_fea)
                # token_before = self.row_pos_emb(row_tokens)
                # tokens_after = self.tr_row_correlator(token_before)  # (b, n_proposal72, c)

                # segmentation & row attention
                # local_prop_fea_com = self._downsample_multiply(sp_bi_seg, local_prop_fea)
                # row_tokens = []
                # for idx_h in range(self.row_size):  # 144
                #     temp_token = local_prop_fea_com[:, :, idx_h, :]  # batch , channel, height, column
                #     temp_token0 = self.to_token_row(temp_token) + getattr(self, f'emb_row_{idx_h}')  # (256,)
                #     # expand because of unsqueeze backward error
                #     row_tokens.append(temp_token0.unsqueeze(1))  # [n_row, b, 256]
                # token_before = torch.cat(row_tokens, dim=1)  # [b, h, c]
                # tokens_after = self.tr_row_correlator(token_before)  # (b, n_proposal72, c)

                sp_ext = self.ext2(tokens_after)  # input: [4, c, 144]
                sp_cls = self.cls2(tokens_after)
                sp_offset = self.offset2(tokens_after)

                proposal_ext2.append(torch.unsqueeze(sp_ext, dim=1))
                proposal_cls2.append(torch.unsqueeze(sp_cls, dim=1))
                proposal_offset2.append(torch.unsqueeze(sp_offset, dim=1))
                proposal_bi_seg.append(torch.unsqueeze(sp_bi_seg, dim=1))

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
        train_label = batch['label']
        lanes_label_raw = batch['label_raw']
        orient_label_raw = batch['ori']
        semantic_label_raw = batch['mask'].long()
        train_label_initp = batch['initp']
        train_label_endp = batch['endp']
        train_label_initp_offset = batch['init_offset']
        train_label_endp_offset = batch['end_offset']
        line_semantic = batch['semantic']

        # print("semantic label raw: ", semantic_label_raw[0, torch.where(semantic_label_raw[0, :, :]==1)[0], torch.where(semantic_label_raw[0, :, :]==1)[1]])
        # print("semantic label raw location: ", torch.where(semantic_label_raw[0, :, :]==1))
        lanes_label = train_label[:, :, :144]   # instance label after downsample: 144 * 144
        _, org_h, org_w = lanes_label_raw.shape # instance label: 1152 * 1152

        # print('lanes_label shape in loss function: ', lanes_label.shape)
        # for three heads: existence, class-wise, endpoint heatmaps
        lb_lc_ext, lb_lc_cls, lb_lc_offset, lb_lc_offset_mask, lb_lc_endp, lb_lc_orient, bi_org_lane_label, semantic_org_lane_label = \
            self.get_lane_exist_and_cls_wise_and_endpoints_maps(lanes_label, train_label_initp, train_label_endp,
                                                                train_label_initp_offset, train_label_endp_offset,
                                                                lanes_label_raw,
                                                                orient_label_tensor=orient_label_raw,
                                                                line_semantic_tensor=line_semantic,
                                                                merge_connect_lines=True, is_ret_list=False)

        ### 1st Stage ###
        with_mooth_mask = False

        # 1. caluculate the minimumn mean distance from the each proposal to the gt, keep this distance and GT id
        # distance size: (b, n_proposal)
        # GT id: (b, n_proposal)
        b_size, proposal_size, _ = out['proposal_conf'].shape #(4, 72, 2)
        col_index = self.prop_width * torch.arange(proposal_size, dtype=torch.float32).cuda()   # 0, 2, 4, 6 ,,,,
        dist_prop_line = col_index.repeat(self.b_size, self.row_size, self.num_cls, 1)
        dist_prop_line = dist_prop_line.permute(0, 3, 1, 2) #[4, 72, 144, 9]: for every proposal has its own initial value: 0, 2, 4,,,
        dist_prop_line_valid = torch.ones_like(dist_prop_line)
        prop_cls = lb_lc_cls.repeat(proposal_size, 1, 1, 1)  # [72, 4, 9, 144]
        # begin: constraint the GT for proposal buffer
        for p_id in range(proposal_size):
            left_border = self.prop_width*p_id-(self.prop_half_buff)
            right_border = self.prop_width*p_id + self.prop_half_buff + self.prop_width
            outside_l = torch.where( (prop_cls[p_id, ...]< left_border) | (prop_cls[p_id, ...]> right_border) )
            prop_cls[p_id, ...][outside_l] = -1
        # end: constraint the GT for proposal buffer

        prop_cls = prop_cls.permute(1, 0, 3, 2)  # [4, 72, 144, 9]

        invalid_cls_loc = torch.where(prop_cls < 0)
        dist_prop_line -= prop_cls
        dist_prop_line[invalid_cls_loc] = 0.         # distance from invalid vertexes is set as 0. For conenience to calculating the mean invalid distance
        dist_prop_line_valid[invalid_cls_loc] = 0   # other vertexes (1) are valid, contribute nothing to mean distance


        dist_prop_line = torch.abs(dist_prop_line)
        dist_prop_line = torch.sum(dist_prop_line, dim=2)    # [4, 72, 9]
        line_valid_sum = torch.sum(dist_prop_line_valid, dim=2)  # [4, 72, 9]
        line_valid_sum[torch.where(line_valid_sum < 1)] = 1  # avoid Nan divide
        dist_prop_line = dist_prop_line / line_valid_sum     # average distance to each gt lane
        dist_prop_line[torch.where(dist_prop_line == 0.)] = 143.  # distance from the proposal to the empty line instance is 143.

        # expand the offset map, bi_seg_map, endp_map
        lb_lc_offset = self.zero_pad_2d(lb_lc_offset)        # [4, 9, 144, 144 + 10 + 10]
        lb_lc_offset_mask_expand = self.zero_pad_2d(lb_lc_offset_mask)  # # [4, 9, 144, 144 + 10 + 10]
        bi_org_lane_label = self.zero_pad_2d_org(bi_org_lane_label)     # [4, 9, 1152, 1152 + 80 + 80]
        # lb_lc_endp = self.zero_pad_2d_org(lb_lc_endp)      # [4, 9, 1152, 1152 + 80 + 80]

        # for every proposal, the min distance to the GT-line
        dist_prop_line_min = torch.amin(dist_prop_line, dim=-1)   # [4, 72, 9]  --> (batch, proposal_size)
        # for every proposal, the corresponding GT-line ID to each proposal
        dist_prop_line_id = torch.argmin(dist_prop_line, dim=-1)  # [4, 72, 9]  --> (batch, proposal_size)

        del dist_prop_line_valid

        # objective_loss =torch.sum(dist_prop_line)
        gt_proposal = torch.zeros((self.b_size, proposal_size, 2)).cuda()
        gt_exist = torch.zeros((self.b_size, proposal_size, self.row_size)).cuda()
        gt_coors = torch.zeros((self.b_size, proposal_size, self.row_size)).cuda()
        gt_offset = torch.zeros((self.b_size, proposal_size, self.row_size, (self.prop_fea_width))).cuda()
        gt_offset_mask = torch.zeros((self.b_size, proposal_size, self.row_size, (self.prop_fea_width))).cuda()
        gt_bi_seg = torch.zeros((self.b_size, proposal_size, org_h, (self.prop_fea_width)*8)).cuda()
        # gt_endps = torch.zeros((self.b_size, proposal_size, org_h, (self.prop_fea_width)*8)).cuda()
        for id_b in range(self.b_size):
            for id_p in range(proposal_size):
                gt_exist[id_b, id_p, :] = lb_lc_ext[id_b, dist_prop_line_id[id_b, id_p], :]
                gt_coors[id_b, id_p, :] = lb_lc_cls[id_b, dist_prop_line_id[id_b, id_p], :] - (self.prop_width * id_p - self.prop_half_buff)  # absolute coordinate
                prop_min_id = self.prop_width*id_p
                prop_max_id = self.prop_width*id_p+(self.prop_fea_width)
                org_min_id = 8*prop_min_id
                org_max_id = 8*prop_max_id
                gt_offset[id_b, id_p, :, :] = lb_lc_offset[id_b, dist_prop_line_id[id_b, id_p], :, prop_min_id:prop_max_id]
                gt_offset_mask[id_b, id_p, :, :] = lb_lc_offset_mask_expand[id_b, dist_prop_line_id[id_b, id_p], :, prop_min_id:prop_max_id]
                gt_bi_seg[id_b, id_p, :, :] = bi_org_lane_label[id_b, dist_prop_line_id[id_b, id_p], :, org_min_id: org_max_id]
                # gt_endps[id_b, id_p, :, :] = lb_lc_endp[id_b, dist_prop_line_id[id_b, id_p], :, org_min_id: org_max_id]

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

        endp_exist = torch.where(torch.sum(lb_lc_endp, dim=(1, 2)) > 1.)  # batch_id
        endp_weight = lb_lc_endp.clone()
        endp_weight[torch.where(endp_weight > EPS)] *= 4
        endp_weight[torch.where(endp_weight < EPS)] = 0.5
        lb_lc_endp[torch.where(lb_lc_endp > EPS)] = 1
        lb_lc_endp[torch.where(lb_lc_endp < EPS)] = 0
        endp_loss_none = torchvision.ops.sigmoid_focal_loss(out['endpoint'][:, 0, :, :][endp_exist], lb_lc_endp[endp_exist], reduction='none')
        endp_loss += torch.sum(endp_weight[endp_exist] * endp_loss_none)
        # for b_id in range(self.b_size):
        #     endp_label = ls_lb_endp[b_id, :, :].detach().cpu().numpy()
        #     endp_weight_np = endp_weight[b_id, :, :].detach().cpu().numpy()
        #     cv2.imshow("end_label", endp_label)
        #     cv2.imshow("endp_weight", endp_weight_np)
        #     cv2.waitKey(0)
        del endp_loss_none, endp_weight

        # binary segmentation loss

        binary_seg_loss += torchvision.ops.sigmoid_focal_loss(out['prop_bi_seg'][proposal_positive[0], proposal_positive[1], :, :].reshape(-1, 1), gt_bi_seg[proposal_positive[0], proposal_positive[1], :, :].reshape(-1, 1), reduction='sum')
        # binary_seg_loss += F.cross_entropy(out['prop_bi_seg'][proposal_positive[0], proposal_positive[1], ...].reshape(-1, 2), gt_bi_seg[proposal_positive[0], proposal_positive[1], ...].long().view(-1), reduction='sum')
        col_index = torch.arange(self.prop_fea_width).cuda()
        col_index_expand = col_index.repeat(self.b_size, proposal_size, self.row_size, 1)
        proposal_loss += F.binary_cross_entropy_with_logits(out['proposal_conf'], gt_proposal)
        ext_loss2 += F.cross_entropy(out['ext2'][proposal_positive[0], proposal_positive[1], :].reshape(-1, 3), gt_exist[proposal_positive[0], proposal_positive[1], :].long().view(-1), reduction='sum')
        row_idx = torch.arange(self.row_size).cuda()
        orient_idx = torch.arange(self.cfg.number_orients).cuda()
        orient_idx_expand = orient_idx.repeat(self.b_size,  self.row_size, self.row_size, 1)
        if self.cfg.heads.cls_exp:
            corr_idx_pred = torch.sum(col_index_expand * (out['cls2'].softmax(dim=3)), dim=3)
            cls_mean_loss2 += F.smooth_l1_loss(corr_idx_pred[vertex_valid], gt_coors[vertex_valid], reduction='sum')
            cls_loss2 += F.cross_entropy(out['cls2'][vertex_valid], gt_coors[vertex_valid].long(), reduction='sum')

            cls_smooth = True
            if cls_smooth == True:
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

        else:
            cls_loss2 += -torch.sum(gt_coors[vertex_valid].long() * torch.log(out['cls2'][vertex_valid] + EPS))

        # offset loss
        offset_loss += F.smooth_l1_loss((out['offset2'] * gt_offset_mask),
                                        (gt_offset * gt_offset_mask), reduction='sum')

        if len(orient_exist[0]) > 0:
            orient_loss = self.orient_w * orient_loss / len(orient_exist[0])
        endp_loss = self.endp_loss_w * endp_loss / (self.row_size * self.row_size * self.b_size)  # for Heatmap
        binary_seg_loss = binary_seg_loss / (self.row_size * self.row_size * 8 * self.b_size)
        # proposal_loss = proposal_loss / (proposal_size)
        ext_loss2 = self.ext_w * ext_loss2 / (proposal_size * self.row_size * self.b_size)

        if vertex_valid_size > 0:
            cls_mean_loss2 = self.mean_loss_w * cls_mean_loss2 / vertex_valid_size
            cls_loss2 = self.lambda_cls * cls_loss2/vertex_valid_size
            offset_loss = self.offset_loss_w * offset_loss / vertex_valid_size
            cls_smooth_loss2 = 5 * self.lambda_cls * cls_smooth_loss2 / vertex_valid_size

        # print(f'ext_loss2 = {ext_loss2}, cls_loss2 = {cls_loss2}, cls_mean_loss2 = {cls_mean_loss2}, '
        #       f'endp_loss = {endp_loss}, '
        #       f'ext_smooth_loss = {ext_smooth_loss}, cls_smooth_loss2={cls_smooth_loss2}')
        # print(f'ext_loss2 = {ext_loss2} , cls_loss2 = {cls_loss2}, cls_mean_loss2 = {cls_mean_loss2}')
        # print(f'endp_loss = {endp_loss}, offset_loss={offset_loss}, bi_seg_loss={binary_seg_loss}, prop_loss={proposal_loss}')
        loss = proposal_loss + ext_loss2 + cls_mean_loss2 + cls_loss2  + cls_smooth_loss2 +\
               endp_loss + orient_loss + binary_seg_loss + offset_loss

        re_loss = {'loss': loss, 'loss_stats': \
            { 'proposal_loss': proposal_loss,
             'ext_loss2': ext_loss2, 'cls_loss2': cls_loss2, 'cls_mean_loss2': cls_mean_loss2,
              'cls_smooth_loss2': cls_smooth_loss2, 'endp_loss': endp_loss,
             'orient_loss':orient_loss, 'binary_seg_loss':binary_seg_loss, 'offset_loss':offset_loss
             }}

        return re_loss

    def get_conf_cls_and_endp_dict(self, out, is_get_1_stage_result=True, with_offset=False):
        # (b, num_proposal) proposal_conf
        # (b, num_proposal, img_h, 3) proposal_ext2
        # (b, num_proposal, img_h, 20) proposal_cls2
        # (b, num_proposal, img_h, 20) proposal_offset2
        # 2 means second stage in forward process

        b_size, num_prop, img_h, prop_w = out['cls2'].shape
        out['proposal_conf'] = out['proposal_conf'].softmax(2).detach().cpu()
        with_endp_clip = False
        endp_topK = self.num_cls * 10
        arr_endp_mask = torch.ones((b_size, img_h, img_h)).cuda()

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
        # semantic_seg = torch.zeros((out['bi_seg'].shape[0], out['bi_seg'].shape[2], out['bi_seg'].shape[3]))
        # semantic_seg[torch.where((out['bi_seg'][:, 1, :, :] > out['bi_seg'][:, 2, :, :]) & (out['bi_seg'][:, 1, :, :] > 0.08))] = 1
        # semantic_seg[torch.where((out['bi_seg'][:, 2, :, :] > out['bi_seg'][:, 1, :, :]) & (out['bi_seg'][:, 2, :, :] > 0.08))] = 2
        # bi_seg_weight_raw = out['bi_seg'][:, 1, :, :] + out['bi_seg'][:, 2, :, :]
        # bi_seg_weight_raw = torch.squeeze(bi_seg_weight_raw, dim=1)
        # bi_seg_weight = F.max_pool2d(bi_seg_weight_raw, kernel_size=(8, 8), stride=(8,8))
        # print("bi_seg_max.shape: ", bi_seg.shape)
        # print("semantic: ", torch.where(bi_seg>0))
        _, _, _, seg_h, seg_w = out['prop_bi_seg'].shape
        semantic_seg = torch.zeros((self.b_size, seg_h, seg_h))
        bi_seg_weight_raw = torch.zeros_like(semantic_seg).cuda()
        for id_b in range(self.b_size):
            for id_p in range(self.num_prop):
                if out['proposal_conf'][id_b, id_p, 1] > self.cfg.exist_thr:
                    pixel_id = torch.where(out['prop_bi_seg'][id_b, id_p, 0, :, :] > self.cfg.conf_thr)
                    # org_w = pixel_id[1] + (8*2*id_p - 80)
                    org_w = pixel_id[1] + int(8 * (self.prop_width * id_p - self.prop_half_buff))
                    org_w[org_w > 1151] = 1151
                    org_w[org_w < 0] = 0
                    semantic_seg[id_b, pixel_id[0], org_w] = 1
                    bi_seg_weight_raw[id_b, pixel_id[0], org_w] = out['prop_bi_seg'][id_b, id_p, 0, pixel_id[0], pixel_id[1]]

        bi_seg_weight = F.max_pool2d(bi_seg_weight_raw, kernel_size=(8, 8), stride=(8, 8))

        ###
        # for binary segmentation: END
        ###

        ###
        # for endpoints: BEGIN
        ###
        org_img_h, org_img_w = img_h * 8, img_h * 8
        arr_endp = torch.zeros((b_size, org_img_h, org_img_w))
        clip_w = 20
        endp_thre = 0.08
        # print("org h - w: ", org_img_h, org_img_w)
        # print("endp_est shape: ", out['endp_est'].shape)
        for idx_b in range(b_size):
            # temp_endp_score = out[f'endpoint'][idx_b, :, :]
            # temp_endp_score = torch.squeeze(out[f'endpoint'][idx_b, 0, clip_w:(img_h-clip_w), clip_w:(img_w-clip_w)])
            temp_endp_score = torch.squeeze(
                out['endpoint'][idx_b, 0, clip_w:(org_img_h - clip_w), clip_w:(org_img_w - clip_w)])
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
                topk_h, topk_w = topk_index // (org_img_w - 2 * clip_w), topk_index % (org_img_w - 2 * clip_w)

                # add clustering method and select 2 clustering centers
                topk_h, topk_w = self.cluster_select_topK_pts(topk_h, topk_w, cluster_r=20, select_K=self.num_cls)
                if len(topk_h) > 4:
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
        # temp_ext = torch.zeros((self.b_size, num_prop, self.row_size))
        # temp_ext[torch.where((out['ext2'][..., 1] > out['ext2'][..., 2]) & (out['ext2'][..., 1] > self.cfg.exist_thr))] = 1
        # temp_ext[torch.where((out['ext2'][..., 2] > out['ext2'][..., 1]) & (out['ext2'][..., 2] > self.cfg.exist_thr))] = 2
        prop_ext_conf = 1.0 - out['ext2'][..., 0].detach().cpu()

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
        dict_ret = {'prop_conf': out['proposal_conf'],
                    'prop_ext_conf': prop_ext_conf, 'prop_cls_conf': out['cls2'],
                    'endp': arr_endp, 'orient':orient_cls, 'bi_seg':bi_seg_weight_raw, 'semantic_seg':semantic_seg,
                    'cls': corr_idx, 'cls_exp': corr_exp, 'cls_offset': corr_offset}

        return dict_ret

    def binary_label_formatting(self, raw_label, is_flip=False):
        # Output image: top-left of the image is farthest-left
        batch_n, label_h, label_w = raw_label.shape  # torch.Size([4, 1152, 1152])
        label_tensor = torch.zeros((batch_n, self.num_cls, label_h, label_w))  # torch.Size([4, n_lane, 1152, 1152])
        for b_id in range(batch_n):
            for lane_id in range(self.num_cls):
                coor_idx = torch.where(raw_label[b_id, :, :] == lane_id)
                label_tensor[b_id, lane_id, coor_idx[0], coor_idx[1]] = 1
        if is_flip:
            label_tensor = torch.flip(label_tensor, (2, 3))

        return label_tensor

    # def label_formatting(self, raw_label, is_flip=True):
    #     # Output image: top-left of the image is farthest-left
    #     num_of_labels = len(raw_label)
    #     label_tensor = torch.zeros((num_of_labels, 2, 144, 144), dtype=torch.long)
    #
    #     for k in range(num_of_labels):
    #         label_temp = torch.zeros((144, 144, 2), dtype=torch.long)
    #         label_data = raw_label[k]
    #         if is_flip:
    #             label_temp[..., 0] = torch.flip(label_data, (0, 1))
    #         else:
    #             label_temp[..., 0] = label_data
    #
    #         label_temp[..., 0][torch.where(label_temp[..., 0] == 255)] = self.num_cls
    #         label_temp[..., 1][torch.where(label_temp[..., 0] < self.num_cls)] = 1
    #         label_tensor[k, :, :, :] = label_temp.permute(2, 0, 1)
    #     return label_tensor


    # original resolution
    def line_label_formatting(self, raw_label, is_flip=True):
        # Output image: top-left of the image is farthest-left
        batch_n, label_h, label_w = raw_label.shape  # torch.Size([4, 1152, 1152])
        label_tensor = torch.zeros((batch_n, 2, label_h, label_w), dtype=torch.long)

        for k in range(batch_n):
            label_temp = torch.zeros((label_h, label_w, 2), dtype=torch.long)
            label_data = raw_label[k]
            if is_flip:
                label_temp[..., 0] = torch.flip(label_data, (0, 1))
            else:
                label_temp[..., 0] = label_data
            label_temp[..., 0][torch.where(label_temp[..., 0] == 255)] = self.num_cls  # label id
            label_temp[..., 1][torch.where(label_temp[..., 0] < self.num_cls)] = 1  # existence
            label_tensor[k, :, :, :] = label_temp.permute(2, 0, 1)
        return label_tensor


    def get_lane_exist_and_cls_wise_and_endpoints_maps(self, lanes_label_down,
                                                       init_point_tensor, end_point_tensor,
                                                       init_offset_tensor, end_offset_tensor,
                                                       lanes_label_raw,
                                                       orient_label_tensor=None,
                                                       line_semantic_tensor=None,
                                                       merge_connect_lines=False, is_ret_list=True):
        label_tensor = self.line_label_formatting(lanes_label_down, is_flip=self.flip_label) # channel0 = line number, channel1 = confidence
        line_label_tensor = self.line_label_formatting(lanes_label_raw, is_flip=self.flip_label)
        b, _, img_h, img_w = label_tensor.shape  # _, 2, 144, 144
        _, _, org_img_h, org_img_w = line_label_tensor.shape
        n_cls = self.num_cls
        # print('init point tensor: ', init_point_tensor.shape) # (batch_size, number_lane, 2)
        # print('end points tensor: ', endpoint_tensor.shape)

        ### Vis each batch ###
        # temp_conf_tensor = np.squeeze(label_tensor[0,1,:,:])
        # temp_cls_tensor = np.squeeze(label_tensor[0,0,:,:])
        # # print('temp confidence tensor: ', temp_conf_tensor)
        #
        # temp_conf_vis_img = np.zeros((144,144), dtype=np.uint8)
        # temp_conf_vis_img[np.where(temp_conf_tensor==1)] = 255
        #
        # temp_cls_vis_img_2 = np.zeros((144,144), dtype=np.uint8)
        # temp_cls_vis_img_2[np.where(temp_cls_tensor==2)] = 255
        #
        # temp_cls_vis_img_3 = np.zeros((144,144), dtype=np.uint8)
        # temp_cls_vis_img_3[np.where(temp_cls_tensor==3)] = 255
        #
        # print(np.unique(temp_conf_tensor))
        # print(np.unique(temp_cls_tensor))
        # temp_endpoint_vis_img = np.zeros((144, 144, 3), dtype=np.uint8)
        # for id in range(init_point_tensor.shape[1]):
        #     temp_endpoint_vis_img[int(init_point_tensor[0, id, 0].item()), int(init_point_tensor[0, id, 1].item()), :] = (0, 0, 255)
        #     temp_endpoint_vis_img[int(endpoint_tensor[0][id][0].cpu().item()), int(endpoint_tensor[0][id][1].cpu().item()), :] = (0, 255, 255)
        #
        # temp_endpoint_vis_img = np.flip(np.flip(temp_endpoint_vis_img, 0), 1)
        # cv2.imshow('temp_conf', temp_conf_vis_img)
        # cv2.imshow('temp_2', temp_cls_vis_img_2)
        # cv2.imshow('temp_3', temp_cls_vis_img_3)
        # cv2.imshow('endpoint', temp_endpoint_vis_img)
        # cv2.waitKey(0)
        ### Vis each batch ###
        lb_cls_raw = torch.squeeze(line_label_tensor[:, 0, :, :])  # 0~5: cls, 6: background
        ret_exist = torch.zeros((b, n_cls, img_h))  # 2,6,144
        ret_maps = torch.zeros((b, n_cls, img_h))
        ret_offset_maps = torch.zeros((b, n_cls, img_h, img_w))
        ret_offset_mask = torch.zeros((b, n_cls, img_h, img_w))
        ret_endpoint_maps = torch.zeros((b, org_img_h, org_img_w), dtype=torch.float32)
        ret_endpoint_coors = torch.zeros((b, n_cls, 4))
        ret_orient_maps = torch.zeros((b, img_h, img_w))
        ret_bi_seg = self.binary_label_formatting(lanes_label_raw, is_flip=self.flip_label)   #[4, n_cls, 1151, 1152], binary segmentation
        ret_semantic_seg = torch.zeros_like(ret_bi_seg)



        for idx_b in range(b):
            # for semantic segmentation
            for c_id in range(n_cls):
                if line_semantic_tensor[idx_b, c_id] > 0:
                    ppp = torch.where(ret_bi_seg[idx_b, c_id, :, :] > 0)
                    ret_semantic_seg[idx_b, c_id, ppp[0], ppp[1] ] = float(line_semantic_tensor[idx_b, c_id])

            ret_exist[idx_b, :, :], ret_maps[idx_b, :, :], ret_offset_maps[idx_b, :, :, :], \
                ret_offset_mask[idx_b, :, :,:], ret_orient_maps[idx_b, :, :] = \
                self.get_line_existence_and_cls_wise_maps_per_batch(torch.squeeze(lb_cls_raw[idx_b, :, :]),
                                                                    n_cls=n_cls,
                                                                    raw_orient_map=orient_label_tensor[idx_b, :, :],
                                                                    line_semantic=line_semantic_tensor[idx_b, :])
            ret_endpoint_maps[idx_b, :, :], _ = self.get_endpoint_maps_per_batch(
                np.squeeze(init_point_tensor[idx_b, :, :]),
                np.squeeze(end_point_tensor[idx_b, :, :]),
                lb_initoffs=np.squeeze(init_offset_tensor[idx_b, :, :]),
                lb_endoffs=np.squeeze(end_offset_tensor[idx_b, :, :]),
                n_cls=n_cls, img_h=org_img_h, img_w=org_img_w, is_flip=self.flip_label, merge_endp_map=True)


        if merge_connect_lines:
            for b_id in range(self.b_size):
                for lane_id1 in range(self.num_cls):
                    end_p1 = end_point_tensor[b_id, lane_id1, :]
                    if (end_p1[0]) > 0 and (end_p1[1] > 0):  # lane_id1 exists
                        for lane_id2 in range(self.num_cls):
                            if lane_id2 == lane_id1:         # lane_id2 is not lane_id1
                                continue
                            start_p2 = init_point_tensor[b_id, lane_id2, :]
                            # lane_id2 exist, the start point of lane_id2 is close to the terminate point of lane_id1, then merge
                            if (start_p2[0] > 0) and (start_p2[1] > 0) and (abs(end_p1[0] - start_p2[0]) < 2) and (abs(end_p1[1] - start_p2[1]) < 2):
                                ext_row_ids = torch.where(ret_exist[b_id, lane_id2, :] > 0)[0]
                                # merge existence
                                ret_exist[b_id, lane_id1, ext_row_ids] = ret_exist[b_id, lane_id2, ext_row_ids]

                                # merge coordinate
                                ret_maps[b_id, lane_id1, ext_row_ids] = ret_maps[b_id, lane_id2, ext_row_ids]
                                ret_offset_maps[b_id, lane_id1, ext_row_ids, :] = ret_offset_maps[b_id, lane_id2, ext_row_ids, :]
                                ret_offset_mask[b_id, lane_id1, ext_row_ids, :] = ret_offset_mask[b_id, lane_id2, ext_row_ids, :]


                                # merge binary segmentation label
                                pixels = torch.where(ret_bi_seg[b_id, lane_id2, :, :]>0)
                                ret_bi_seg[b_id, lane_id1, pixels[0], pixels[1]] = 1

                                # merge semantic segmentation label
                                ret_semantic_seg[b_id, lane_id1, pixels[0], pixels[1]] = line_semantic_tensor[b_id, lane_id2].cpu().float()

                                ret_exist[b_id, lane_id2, ext_row_ids] = 0
                                ret_maps[b_id, lane_id2, ext_row_ids] = -1
                                ret_offset_maps[b_id, lane_id2, ext_row_ids, :] = 0
                                ret_offset_mask[b_id, lane_id2, ext_row_ids, :] = 0
                                init_point_tensor[b_id, lane_id2, :] = 0
                                end_point_tensor[b_id, lane_id2, :] = 0
                                ret_bi_seg[b_id, lane_id2, :, :] = 0
                                ret_semantic_seg[b_id, lane_id2, pixels[0], pixels[1]] = 0

            # # visualization:
            # for cls_id in range(n_cls):
            #     lane_cls_map = np.zeros((self.row_size, self.row_size))
            #     col_ind = ret_maps[idx_b,cls_id, torch.where(ret_maps[idx_b, cls_id, :] > 0.)[0]].detach().cpu().numpy()
            #     #     print("index: ", np.floor(col_ind))
            #     lane_cls_map[torch.where(ret_exist[idx_b, cls_id, :]>0.)[0].detach().cpu(), (col_ind).astype(np.int)] = 255
            #     cv2.imshow(f'class map {cls_id}: ', lane_cls_map)
            #     draw_map = ret_endpoint_maps[idx_b, cls_id, :, :]
            #     draw_map[torch.where(draw_map>0)] = 1
            #     cv2.imshow('endpoints map: ', draw_map.detach().cpu().numpy())
            #     cv2.imshow('binary segmentation map: ', ret_bi_seg[idx_b, cls_id, :, :])
            #     cv2.waitKey(0)
            #     endp_locat = np.where(ret_endpoint_maps[idx_b, :,:, :] > 0.9)
            #     print("ret_endp: ", ret_endpoint_maps[idx_b, endp_locat[0], endp_locat[1]])

            # # end visualization

        if is_ret_list:
            list_ext = []
            list_cls = []
            list_offset = []
            list_offset_mask = []
            list_endp_coor = []
            # print("to construct the rer list, ret_exist.shape: ", ret_exist.shape)  # (4, 10, 144, 2)
            for idx_cls in range(self.num_cls):
                list_ext.append(torch.squeeze(ret_exist[:, idx_cls,
                                              :]).cuda())  # original code: it was 3 dimension, why it is 4 dimension here ? because the one_hot parameter is True
                list_cls.append(torch.squeeze(ret_maps[:, idx_cls, :]).cuda())
                list_offset.append(torch.squeeze(ret_offset_maps[:, idx_cls, :, :]).cuda())
                list_offset_mask.append(torch.squeeze(ret_offset_mask[:, idx_cls, :, :]).cuda())
                # list_endp_coor.append(torch.tensor(np.squeeze(ret_endpoint_coors[:, idx_cls, :])).cuda())
            return list_ext, list_cls, list_offset, list_offset_mask, ret_endpoint_maps.cuda(), \
                ret_orient_maps.cuda(), ret_bi_seg.cuda(), ret_semantic_seg.cuda()
        else:
            return ret_exist.cuda(), ret_maps.cuda(), ret_offset_maps.cuda(), ret_offset_mask.cuda(), \
                ret_endpoint_maps.cuda(), ret_orient_maps.cuda(), ret_bi_seg.cuda(), ret_semantic_seg.cuda()

    def get_line_existence_and_cls_wise_maps_per_batch(self, lb_cls, n_cls=6, img_h=144, img_w=144, downsample=True,
                                                       raw_orient_map=None, line_semantic=None):
        # print(lb_cls.shape) # torch.Size([144, 144])
        cls_maps_raw = torch.zeros((n_cls, img_h * 8))
        cls_maps = torch.zeros((n_cls, img_h))
        # semantic_maps = torch.zeros((n_cls, img_h)).cuda()

        line_ext = torch.zeros((n_cls, img_h))
        orient_map = torch.zeros((img_h, img_w)).cuda()

        col_index = torch.arange(img_w, dtype=torch.float32)
        cls_offset_maps = col_index.repeat(n_cls, img_h, 1)
        cls_offset_mask = torch.zeros((n_cls, img_h, img_w))
        for idx_cls in range(n_cls):
            pixels = torch.where(lb_cls == idx_cls)
            if downsample:
                cls_maps_raw[idx_cls, pixels[0]] = pixels[1] / 8.
            else:
                cls_maps_raw[idx_cls, pixels[0]] = pixels[1] / 1.
            cls_maps[idx_cls, :] = cls_maps_raw[idx_cls, 3:1152:8].clone()
            # if raw_mask_map is not None:
            #     semantic_maps_raw = torch.zeros((1, img_h*8)).cuda()
            #     semantic_maps_raw[0, pixels[0]] = raw_mask_map[pixels[0], pixels[1]]
            #     semantic_maps[idx_cls, :] = semantic_maps_raw[0, 3:1152:8].clone()

            cls_offset_maps[idx_cls, :, :] = torch.transpose(cls_maps[idx_cls, :].repeat(img_h, 1), 0,
                                                             1) - cls_offset_maps[idx_cls, :, :]

            if len(torch.where(cls_maps[idx_cls, :] == 0.)[0]) > 0:
                cls_maps[idx_cls, torch.where(cls_maps[idx_cls, :] == 0.)[0]] = -1.
            if len(torch.where(cls_maps[idx_cls, :] > 0.)[0]) > 0:
                if line_semantic is not None:
                    line_ext[idx_cls, torch.where(cls_maps[idx_cls, :] > 0.)[0]] = float(line_semantic[idx_cls])
                else:
                    line_ext[idx_cls, torch.where(cls_maps[idx_cls, :] > 0.)[0]] = 1.

            if raw_orient_map != None:
                down_row = (torch.where(cls_maps[idx_cls, :] > 0.))[0]
                if (len(down_row) < 2):
                    continue
                pixel_num = len(down_row)
                down_col = cls_maps[idx_cls, down_row].long()
                down_col_left = down_col - 1
                down_col_left = torch.where(down_col_left < 0, 0, down_col_left)
                down_col_right = down_col + 1
                down_col_right = torch.where(down_col_right > (img_w - 1), img_w - 1, down_col_right)
                up_row = down_row * 8 + 3
                up_col = (cls_maps[idx_cls, down_row] * 8).long()
                for idx_p in range(pixel_num):
                    orient_map[down_row[idx_p], down_col[idx_p]] = raw_orient_map[up_row[idx_p], up_col[idx_p]]
                    orient_map[down_row[idx_p], down_col_left[idx_p]] = raw_orient_map[up_row[idx_p], up_col[idx_p]]
                    orient_map[down_row[idx_p], down_col_right[idx_p]] = raw_orient_map[up_row[idx_p], up_col[idx_p]]
        cls_offset_mask[torch.where(torch.abs(cls_offset_maps) < 3.0)] = 1.  # set the offset mask
        cls_offset_mask[:, :, :3] = 0.  # avoid the first 2 columns
        # if semantic_maps_raw is not None:
        #     line_ext = semantic_maps
        # for visualization
        # for i in range(n_cls):
        #     cls_map = np.zeros((self.row_size, self.row_size))
        #     col_ind = cls_maps[i, torch.where(cls_maps[i, :]>0.)[0]].detach().cpu().numpy()
        #     print("index: ", np.floor(col_ind))
        #     cls_map[torch.where(line_ext[i, :]>0.)[0].detach().cpu(), (col_ind).astype(np.int)] = 255
        #     cv2.imshow(f'hi_{i}', cls_map)
        #     offset_mask = cls_offset_mask[i, :, :]
        #     cv2.imshow(f'offset_{i}', offset_mask.detach().cpu().numpy())
        #     exist_map = np.zeros((self.row_size, self.row_size))
        #     exist_map[torch.where(line_ext[i, :] > 0.)[0].detach().cpu(), (col_ind).astype(np.int)] = 255
        #     print("semantic index: ", line_ext[i, torch.where(line_ext[i, :]>0.)[0]])
        #     cv2.imshow(f'semantic_{i}', exist_map)
        #     if raw_orient_map != None:
        #         orient_map_show = orient_map.detach().cpu().numpy()
        #         orient_map_show = np.where(orient_map_show > 0, orient_map_show + 200, 0.)
        #         cv2.imshow("orient_map: ", orient_map_show)
        # cv2.waitKey(0)
        # print("line exist shape: ", line_ext.shape) # (n_class, h)

        if raw_orient_map != None:
            return line_ext, cls_maps, cls_offset_maps, cls_offset_mask, orient_map
        else:
            return line_ext, cls_maps, cls_offset_maps, cls_offset_mask


    def get_cls_wise_lines_per_batch(self, lb_cls, n_cls=6, img_h=144, img_w=144):
        line_coors = np.zeros((n_cls, img_h))
        pp = np.arange(n_cls)
        for idx_cls in pp:
            pixels = np.where(lb_cls == idx_cls)
            line_coors[idx_cls, pixels[0]] = pixels[1]
            line_coors[idx_cls, np.where(line_coors[idx_cls, :] == 0)] = img_w

        return line_coors

    def gaussian(self, x_id, y_id, H, W, sigma=5):
        ## Function that creates the heatmaps ##
        channel = [math.exp(-((x - x_id) ** 2 + (y - y_id) ** 2) / (2 * sigma ** 2)) for x in range(W) for y in
                   range(H)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))
        return channel

    def get_endpoint_maps_per_batch(self, lb_initpoints, lb_endpoints, lb_initoffs=None, lb_endoffs=None,
                                    n_cls=6, img_h=144, img_w=144, is_flip=True, merge_endp_map=False):
        EPS = 1e-3
        with_gaussian_kernel = True
        endpoint_maps = torch.zeros((n_cls, img_h, img_w))
        endpoint_offs = torch.zeros((n_cls, 2, img_h, img_w), dtype=torch.float32)

        # for endpoint
        kernel_size = 4
        clip_width = kernel_size * 5
        if with_gaussian_kernel:
            for idx_cls in range(n_cls):
                if (torch.abs(lb_endpoints[idx_cls][0] - lb_initpoints[idx_cls][0]) < EPS) and (
                        torch.abs(lb_endpoints[idx_cls][1] - lb_initpoints[idx_cls][1]) < EPS):
                    # print("No lane instance for class_id is: ", idx_cls)
                    continue
                # for init point: Gaussian kernel roi
                heatmap1 = np.zeros((img_h, img_w))
                heatmap2 = np.zeros((img_h, img_w))
                if lb_initpoints[idx_cls][0] > clip_width and lb_initpoints[idx_cls][0] < (img_h - clip_width) and \
                        lb_initpoints[idx_cls][1] > clip_width and lb_initpoints[idx_cls][1] < (img_w - clip_width):
                    heatmap1 = self.gaussian(int(lb_initpoints[idx_cls][0]), int(lb_initpoints[idx_cls][1]), img_h,
                                             img_w, sigma=kernel_size / 2)
                # for end point: Gaussian kernel roi
                if lb_endpoints[idx_cls][0] > clip_width and lb_endpoints[idx_cls][0] < (img_h - clip_width) and \
                        lb_endpoints[idx_cls][1] > clip_width and lb_endpoints[idx_cls][1] < (img_w - clip_width):
                    heatmap2 = self.gaussian(int(lb_endpoints[idx_cls][0]), int(lb_endpoints[idx_cls][1]), img_h, img_w,
                                             sigma=kernel_size / 2)
                two_maps = np.array([heatmap1, heatmap2])
                two_maps = np.max(two_maps, axis=0)
                endpoint_maps[idx_cls, :, :] = torch.tensor(two_maps)
        # make sure the endpoint is eaqual to 1
        for idx_cls in range(n_cls):
            if (torch.abs(lb_endpoints[idx_cls][0] - lb_initpoints[idx_cls][0]) < EPS) and (
                    torch.abs(lb_endpoints[idx_cls][1] - lb_initpoints[idx_cls][1]) < EPS):
                # print("No lane instance for class_id is: ", idx_cls)
                continue
            else:
                if lb_initpoints[idx_cls][0] > clip_width and lb_initpoints[idx_cls][0] < (img_h - clip_width) and \
                        lb_initpoints[idx_cls][1] > clip_width and lb_initpoints[idx_cls][1] < (img_w - clip_width):
                    endpoint_maps[idx_cls, int(lb_initpoints[idx_cls][0]), int(lb_initpoints[idx_cls][1])] = 1
                if lb_endpoints[idx_cls][0] > clip_width and lb_endpoints[idx_cls][0] < (img_h - clip_width) and \
                        lb_endpoints[idx_cls][1] > clip_width and lb_endpoints[idx_cls][1] < (img_w - clip_width):
                    endpoint_maps[idx_cls, int(lb_endpoints[idx_cls][0]), int(lb_endpoints[idx_cls][1])] = 1

        # offsets of the endpoint
        if (lb_initoffs is not None) and (lb_endoffs is not None):
            endpoint_offs[idx_cls, 0, int(lb_initpoints[idx_cls][0]), int(lb_initpoints[idx_cls][1])] = \
                lb_initoffs[idx_cls][0]
            endpoint_offs[idx_cls, 1, int(lb_initpoints[idx_cls][0]), int(lb_initpoints[idx_cls][1])] = \
                lb_initoffs[idx_cls][1]
            endpoint_offs[idx_cls, 0, int(lb_endpoints[idx_cls][0]), int(lb_endpoints[idx_cls][1])] = \
                lb_endoffs[idx_cls][0]
            endpoint_offs[idx_cls, 1, int(lb_endpoints[idx_cls][0]), int(lb_endpoints[idx_cls][1])] = \
                lb_endoffs[idx_cls][1]
        if is_flip:
            endpoint_maps = torch.flip(torch.flip(endpoint_maps, 1), 2)
            endpoint_offs = torch.flip(torch.flip(torch.flip(endpoint_offs, 0), 1), 2)
        if merge_endp_map:
            endpoint_maps = torch.amax(endpoint_maps, dim=0, keepdim=True)#.astype(torch.float32)
            # endpoint_maps[endpoint_maps>0.1] = 1
            # endpoint_maps[endpoint_maps <= 0.1] = 0
        return endpoint_maps, endpoint_offs

    def get_endpoint_coors_per_batch(self, lb_initpoints, lb_endpoints, lb_initoffs=None, lb_endoffs=None, n_cls=6,
                                     img_h=144, img_w=144):
        endpoint_coors = np.zeros((n_cls, 4), dtype=np.float32)
        for idx_cls in range(n_cls):
            # print('idx_cls: ', idx_cls)
            # print('initpoints: ', lb_initpoints)
            # print('lb_endpoint[idx_cls]: ', lb_endpoints[idx_cls])
            # print('lb_initpoint[idx_cls]: ', lb_initpoints[idx_cls])
            if ((lb_endpoints[idx_cls][0] - lb_initpoints[idx_cls][0]) == 0) and (
                    (lb_endpoints[idx_cls][1] - lb_initpoints[idx_cls][1]) == 0):
                # print("No lane instance for class_id is: ", idx_cls)
                continue
            else:
                if (lb_initoffs is not None) and (lb_endoffs is not None):
                    endpoint_coors[idx_cls, :] = [
                        (int(lb_initpoints[idx_cls][0].item()) + lb_initoffs[idx_cls][0].item()) / img_h,
                        (int(lb_initpoints[idx_cls][1].item()) + lb_initoffs[idx_cls][1].item()) / img_w,
                        (int(lb_endpoints[idx_cls][0].item()) + lb_endoffs[idx_cls][0].item()) / img_h,
                        (int(lb_endpoints[idx_cls][1].item()) + lb_endoffs[idx_cls][1].item()) / img_w]
                else:
                    endpoint_coors[idx_cls, :] = [
                        (int(lb_initpoints[idx_cls][0].item())) / img_h,
                        (int(lb_initpoints[idx_cls][1].item())) / img_w,
                        (int(lb_endpoints[idx_cls][0].item())) / img_h,
                        (int(lb_endpoints[idx_cls][1].item())) / img_w]

        return endpoint_coors

    def get_center_point_from_2_endpoints(self, ls_lb_endp_coor):
        # ls_lb_endp_coor = np.zeros((b, n_cls, 4))
        # print("endp_coordinate: ", ls_lb_endp_coor) # shape: class, batch, 4
        center_w = 0.5 * (ls_lb_endp_coor[:, :, 1] + ls_lb_endp_coor[:, :, 3])
        delta_w = torch.abs(ls_lb_endp_coor[:, :, 1] - ls_lb_endp_coor[:, :, 3])
        # print("center_w: ", center_w)  # shape: class, batch
        # print("delta_w: ", delta_w)    # shape: class, batch
        return center_w, delta_w

    def smooth_cls_line_per_batch(self, out_cls, out_orient, complete_inner_nodes=False, out_seg_conf=None):
        # traverse every lane from the first lane:
        buff_width = 8
        buff_depth = 15
        line_num, vertex_num = out_cls.shape
        smooth_cls = out_cls.copy()
        smooth_cls_total = np.zeros_like(out_cls) - 1
        exist_lane_length = np.zeros(line_num)
        flag_in0 = np.zeros((self.row_size, 1152))  # detected points location
        for idx_line in range(line_num):
            ph_idx = np.where(out_cls[idx_line, :] > 0)
            flag_in0[ph_idx[0], (out_cls[idx_line, ph_idx[0]]).astype(int)] = 1
        # print("flag_in sumL ", np.sum(flag_in0))
        if out_seg_conf is not None:
            flag_in = self.occupancy_filter(flag_in0, out_seg_conf[3:1152:8, :], half_k_size=3)
            # flag_in = flag_in0

        # print("flag_in sumL2 ", np.sum(flag_in))
        # print("diff: ", np.where(flag_in != flag_in0))
        while flag_in.sum() > 2 and exist_lane_length.min() < 2:  # free vertex & empty output lane id
            temp_smooth_cls = np.zeros_like(out_cls) - 1 # initial vertex
            tmp_lane_length = np.zeros(line_num)     # length of each lane
            for idx_line in range(line_num):
                flag_start = False
                last_h = 0
                idx_h = 0
                active_lane_id = idx_line
                while idx_h < self.row_size:
                    if flag_start and (idx_h - last_h > buff_depth):  # the distance between adjacent vertexes is too large
                        break
                    if not flag_start:  # spot the first vertex of the lane
                        if smooth_cls[idx_line, idx_h] > 0 and flag_in[idx_h, int(smooth_cls[idx_line, idx_h])] > 0:
                            current_h = idx_h
                            current_col = smooth_cls[idx_line, idx_h]
                            current_dir = out_orient[idx_h, int(current_col / 8)]
                            # next_pred_col = current_col + (current_dir - 5) * 4

                            flag_start = True
                            flag_in[idx_h, int(smooth_cls[idx_line, idx_h])] = 0  # this vertex is occupied
                            temp_smooth_cls[idx_line, idx_h] = current_col
                            tmp_lane_length[idx_line] += 1
                            last_h = idx_h
                            active_lane_id = idx_line
                        idx_h += 1  # move to next row
                    else:  # vertex string
                        next_pred_col = current_col + (current_dir - 5) * 4
                        near_dist = 1152
                        near_id = line_num
                        near_h = idx_h
                        # width traverse for searching
                        for sub_idx_line in range(line_num):  # list the candidate, choose the closest one
                            if smooth_cls[sub_idx_line, idx_h] > 0 and flag_in[idx_h, int(smooth_cls[sub_idx_line, idx_h])] > 0:
                                tmp_dist = np.abs(next_pred_col - smooth_cls[sub_idx_line, idx_h])
                                if tmp_dist < near_dist:
                                    near_dist = tmp_dist
                                    near_id = sub_idx_line
                                    near_h = idx_h
                        # depth traverse for searching
                        for next_h_idx in range(idx_h+1, self.row_size):
                            if (next_h_idx - idx_h) > buff_depth:
                                break
                            if smooth_cls[active_lane_id, next_h_idx] > 0 and flag_in[next_h_idx, int(smooth_cls[active_lane_id, next_h_idx])] > 0:
                                tmp_dist = np.abs(next_pred_col - smooth_cls[active_lane_id, next_h_idx])
                                if tmp_dist < near_dist:
                                    near_dist = tmp_dist
                                    near_id = active_lane_id
                                    near_h = next_h_idx
                                break  # finish as soon as searched the first vertex

                        if near_dist < buff_width:  # succeed in finding next vertex
                            temp_smooth_cls[idx_line, near_h] = smooth_cls[near_id, near_h]
                            tmp_lane_length[idx_line] += 1
                            # renew the coordinates
                            current_col = smooth_cls[near_id, near_h]
                            current_dir = out_orient[near_h, int(current_col / 8)]
                            current_h = near_h
                            next_pred_col = current_col + (current_dir - 5) * 4
                            flag_in[near_h, int(smooth_cls[near_id, near_h])] = 0  # this vertex is occupied
                            last_h = near_h
                            idx_h = near_h + 1
                            active_lane_id = near_id
                        else:  # fail in finding the next vertex
                            temp_smooth_cls[idx_line, idx_h] = -1  # no vertex
                            # we find no next vertex, then stop extending this line.
                            #break
                            idx_h += 1
                    # print("idx_h: ", idx_h)
            # print("flag_in sumL3 ", np.sum(flag_in))
            # print("minimun length: ", exist_lane_length.min())
            # merge to total result:
            for idx_line in range(line_num):
                if tmp_lane_length[idx_line] > 2:
                    tmp_vertex_idx = np.where(temp_smooth_cls[idx_line, :] > 0)
                    tmp_startp_idx_h = tmp_vertex_idx[0][0]
                    tmp_startp_idx_value = temp_smooth_cls[idx_line, tmp_startp_idx_h]
                    tmp_endp_idx_h = tmp_vertex_idx[0][-1]
                    tmp_endp_idx_value = temp_smooth_cls[idx_line, tmp_endp_idx_h]
                    tmp_endp_dir = out_orient[tmp_endp_idx_h, int(tmp_endp_idx_value / 8)]
                    tmp_endp_next_col = tmp_endp_idx_value + (tmp_endp_dir - 5) * 4

                    attached = False
                    for sub_idx_line in range(line_num):
                        if exist_lane_length[sub_idx_line] >= 2:
                            # check the end vertex of existing and begin vertex of current lane line
                            vertex_idx = np.where(smooth_cls_total[sub_idx_line, :] > 0)
                            startp_idx_h = vertex_idx[0][0]
                            startp_idx_value = smooth_cls_total[sub_idx_line, startp_idx_h]
                            endp_idx_h = vertex_idx[0][-1]
                            ednp_idx_value = smooth_cls_total[sub_idx_line, endp_idx_h]
                            current_end_dir = out_orient[endp_idx_h, int(ednp_idx_value / 8)]
                            endp_next_col = ednp_idx_value + (current_end_dir - 5) * 4
                            #  attach to bottom || # attach to top
                            if (0 < (tmp_startp_idx_h - endp_idx_h) < buff_depth and np.abs(endp_next_col - tmp_startp_idx_value) < buff_width) or \
                                (0 < (startp_idx_h - tmp_endp_idx_h) < buff_depth and np.abs(tmp_endp_next_col - startp_idx_value) < buff_width):
                                smooth_cls_total[sub_idx_line, tmp_vertex_idx[0]] = temp_smooth_cls[idx_line, tmp_vertex_idx[0]]
                                exist_lane_length[sub_idx_line] += tmp_lane_length[idx_line]
                                attached = True
                                break

                    if attached == False:  # start a new lane
                        for sub_idx_line in range(line_num):
                            if exist_lane_length[sub_idx_line] < 2:
                                smooth_cls_total[sub_idx_line, tmp_vertex_idx[0]] = temp_smooth_cls[idx_line, tmp_vertex_idx[0]]
                                exist_lane_length[sub_idx_line] = tmp_lane_length[idx_line]
                                break

        # if complete the inner nodes
        if complete_inner_nodes:
            smooth_cls_total = self.interpolate_plyline(smooth_cls_total)

        # return the lines after smoothing
        return smooth_cls_total

    def interpolate_plyline(self, lane_vertex):
        for idx_line in range(self.num_cls):
            ph_idx = np.where(lane_vertex[idx_line, :] > 0)
            if len(ph_idx[0]) > 1:
                start_id = ph_idx[0][0]
                end_id = ph_idx[0][-1]
                current_positive_id = -1
                for v_id in range(start_id, end_id):
                    if lane_vertex[idx_line, v_id] < 0:
                        lane_vertex[idx_line, v_id] = (lane_vertex[idx_line, ph_idx[0][current_positive_id]] + lane_vertex[idx_line, ph_idx[0][current_positive_id + 1]]) * 0.5
                    else:
                        current_positive_id += 1
        return lane_vertex
    def occupancy_filter(self, occu_flag, occu_seg_conf, half_k_size=4):
        f_row, f_col = occu_flag.shape
        occu_flag_copy = occu_flag.copy()
        for r_id in range(f_row):
            for c_id in range(half_k_size, f_col-half_k_size):
                # if more than 2 vertexes are in one buffer zone, keep the one with higher confidence
                # print("occu_flag[r_id, c_id-half_k_size:c_id+half_k_size]", occu_flag[r_id, c_id-half_k_size:c_id+half_k_size])
                if np.sum(occu_flag_copy[r_id, (c_id-half_k_size):(c_id+half_k_size)]) > 1:
                    # find the one with highest confidence
                    local_values = occu_seg_conf[r_id, (c_id-half_k_size):(c_id+half_k_size)]
                    local_idxes = np.where(occu_flag_copy[r_id, (c_id-half_k_size):(c_id+half_k_size)] > 0)[0]
                    # get max value index
                    max_id = local_idxes[0]
                    max_value = local_values[max_id]
                    for id in local_idxes:
                        if local_values[id] > max_value:
                            max_id = id
                            max_value = local_values[max_id]

                    occu_flag_copy[r_id, (c_id-half_k_size):(c_id+half_k_size)] = 0
                    occu_flag_copy[r_id, (c_id - half_k_size + max_id)] = 1.
        return occu_flag_copy

    def cluster_select_topK_pts(self, pts_h, pts_w, cluster_r = 4, select_K=2):
        # cluster centers
        cen_clusters = []
        cen_clusters.append([pts_h[0], pts_w[0]])

        ele_clusters = [[]]

        # clustering
        for pt_h, pt_w in zip(pts_h, pts_w):
            in_exist_cluster = False
            for idx_cen, center in enumerate(cen_clusters):
                dist = torch.sqrt(((center[0] - pt_h)**2 + (center[1] - pt_w)**2).float())
                if dist < cluster_r:
                    in_exist_cluster = True
                    ele_clusters[idx_cen].append([pt_h, pt_w])
                    # renew cluster center:
                    new_center = torch.zeros((2)).cuda()
                    for id in range(len(ele_clusters[idx_cen])):
                        new_center[0] += ele_clusters[idx_cen][id][0]
                        new_center[1] += ele_clusters[idx_cen][id][1]
                    new_center[0] /= len(ele_clusters[idx_cen])
                    new_center[1] /= len(ele_clusters[idx_cen])
                    cen_clusters[idx_cen][0] = new_center[0]
                    cen_clusters[idx_cen][1] = new_center[1]
                    break
            if in_exist_cluster == False:
                new_center = [pt_h, pt_w]
                cen_clusters.append(new_center)
                ele_clusters.append([new_center])

        # select topK by cluster size
        cluster_size = [len(ele) for ele in ele_clusters]
        cluster_size = np.array(cluster_size)
        if len(cluster_size) < select_K:
            cluster_size_sorted = np.argsort(cluster_size)
        else:
            cluster_size_sorted = np.argsort(cluster_size)[:select_K]

        # choose the nearest pt or cluster center?
        center_h = [ cen_clusters[k][0] for k in cluster_size_sorted]
        center_w = [ cen_clusters[k][1] for k in cluster_size_sorted]

        return torch.tensor(center_h), torch.tensor(center_w)

    def get_lane_map_numpy_with_label(self, output, data, is_flip=True, is_img=False, is_get_1_stage_result=False):
        '''
        * in : output feature map
        * out: lane map with class or confidence
        *       per batch
        *       ### Label ###
        *       'cls_exist': (n_lane, 144) / 0(not exist), 1(exist)
        *       'cls_label': (n_lane, 144) / col coordinate at the original resolution
        *       ### Raw Prediction ###
        *       'prop_conf': (batch, n_proposal, 2)
        *       'prop_ext_conf': (Batch, n_proposal, 144) / 0 ~ 1
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

        list_cls_label = []
        list_conf_label = []
        list_conf_label_raw = [] # (1152, 1152), 0, 1
        list_exist_label = []  # (n_lane, 144): -1(not exist), 1 (exist)
        list_coor_label = []   # (n_lane, 144): coordinate(-1, 0~1152)

        list_conf_by_cls = []  # (144, 144): 0, 1 (by argmax)
        list_exist_pred = []  # (n_lane, 144): -1(not exist), 1 (exist)

        list_cls_idx = []     # (144, 144), 0, 1, 2, 3, ...255
        list_cls_coor = []    # (n_lane, 144): coordinate(-1, 0~1152)
        list_cls_coor_smooth = []  # (n_lane, 144): coordinate(-1, 0~1152)
        list_cls_exp_smooth = []  # (n_lane, 144): coordinate(-1, 0~1152)
        list_cls_offset_smooth = []  # (n_lane, 144): coordinate(-1, 0~1152)

        list_endp_pred = []
        list_bi_seg_pred = []

        batch_size, num_prop, num_hv, num_wc = output['prop_cls_conf'].shape
        raw_label = data['label_raw'].cpu()

        for batch_idx in range(batch_size):
            cls_label = data['label'][batch_idx].cpu().numpy()
            conf_label = np.where(cls_label == 255, 0, 1)  # KLane
            exist_labl, coor_label, _, _ = self.get_line_existence_and_cls_wise_maps_per_batch(raw_label[batch_idx], n_cls=self.num_cls, downsample=False)
            exist_labl = exist_labl.cpu().numpy()
            exist_labl[exist_labl==0] = -1
            coor_label = coor_label.cpu().numpy()
            conf_label_raw = np.where(raw_label[batch_idx].numpy() == 255, 0, 1)

            # 144, 144
            prop_conf = output['prop_conf'][batch_idx, :, 1]  # channel-1 is the proposal existing confidence
            prop_exist_conf_v = output['prop_ext_conf'][batch_idx, :, :] # (n_proposal, row_num)
            prop_cls_conf_v = output['prop_cls_conf'][batch_idx, :, :, :]   # (n_proposal, row_num, 20)


            # whether exist the lane in the proposal
            prop_no_exist = torch.where(prop_conf < 0.1)[0]
            # print('proposals confidence: ', prop_conf)
            # print('proposals dont exist any lane: ', prop_no_exist)
            prop_exist_conf_v[prop_no_exist, :] = 0.

            # n_lane, 144
            v_exist_pred = torch.where(prop_exist_conf_v>0.1, 1, -1).cpu().numpy()
            # exist_pred = torch.where(exist_pred>self.cfg.exist_thr, 1, -1).cpu().numpy()
            cls_coor_pred = output['cls'][batch_idx].cpu().numpy() / self.row_size * 1152 + 4
            cls_coor_pred = np.where(v_exist_pred==(-1), -1, cls_coor_pred)
            cls_coor_exp = output['cls_exp'][batch_idx].cpu().numpy() / self.row_size * 1152
            cls_coor_exp = np.where(v_exist_pred==(-1), -1, cls_coor_exp)
            cls_coor_offset = output['cls_offset'][batch_idx].cpu().numpy() / self.row_size * 1152
            cls_coor_offset = np.where(v_exist_pred==(-1), -1, cls_coor_offset)
            cls_coor_pred[np.where(cls_coor_pred<0)] = 0
            cls_coor_pred[np.where(cls_coor_pred > 1151)] = 1151
            cls_coor_exp[np.where(cls_coor_exp < 0)] = 0
            cls_coor_exp[np.where(cls_coor_exp > 1151)] = 1151
            cls_coor_offset[np.where(cls_coor_offset < 0)] = 0
            cls_coor_offset[np.where(cls_coor_offset > 1151)] = 1151

            # print("predicted coordinates: ", cls_coor_pred)
            # segmentation augment
            # for idx_lane in range(52):
            #     for idx_h in range(self.row_size):
            #         col1 = np.floor(cls_coor_pred[idx_lane, idx_h])
            #         col2 = np.floor(cls_coor_exp[idx_lane, idx_h])
            #         col3 = np.floor(cls_coor_offset[idx_lane, idx_h])
            #         if col3 > 1151:
            #             col3=1151
            #         if output['bi_seg'][batch_idx, idx_h*8+3, int(col1)] < 0.08 and exist_pred[idx_lane, idx_h] < 0.1:
            #             cls_coor_pred[idx_lane, idx_h] = -1.
            #         if output['bi_seg'][batch_idx, idx_h*8+3, int(col2)] < 0.08 and exist_pred[idx_lane, idx_h] < 0.1:
            #             cls_coor_exp[idx_lane, idx_h] = -1.
            #         if output['bi_seg'][batch_idx, idx_h*8+3, int(col3)] < 0.08 and exist_pred[idx_lane, idx_h] < 0.1:
            #             cls_coor_offset[idx_lane, idx_h] = -1.

            # smooth n_lane coordinates:
            cls_coor_pred_smooth = self.smooth_cls_line_per_batch(cls_coor_pred, output['orient'][batch_idx].cpu().numpy(), complete_inner_nodes=True, out_seg_conf=output['bi_seg'][batch_idx].cpu().numpy())
            cls_coor_exp_smooth = self.smooth_cls_line_per_batch(cls_coor_exp, output['orient'][batch_idx].cpu().numpy(), complete_inner_nodes=True, out_seg_conf=output['bi_seg'][batch_idx].cpu().numpy())
            cls_coor_offset_smooth = self.smooth_cls_line_per_batch(cls_coor_offset, output['orient'][batch_idx].cpu().numpy(), complete_inner_nodes=True, out_seg_conf=output['bi_seg'][batch_idx].cpu().numpy())

            # add offset to the predicted vertex:
            # cls_coor_offset_smooth = self.add_offset_2_vertex(cls_coor_pred_smooth, output['offset'][batch_idx].cpu().numpy())

            # apply line NMS (Non-Maximum Suppression)
            cls_coor_pred_smooth = self.line_NMS(cls_coor_pred_smooth, output['bi_seg'][batch_idx].cpu().numpy())
            cls_coor_exp_smooth = self.line_NMS(cls_coor_exp_smooth, output['bi_seg'][batch_idx].cpu().numpy())
            cls_coor_offset_smooth = self.line_NMS(cls_coor_offset_smooth, output['bi_seg'][batch_idx].cpu().numpy())
            # for endpoint
            endp_pred_raw = output['endp'][batch_idx].cpu().numpy()

            # for binary segmentation
            # bi_seg_raw = torch.zeros_like(output['bi_seg'][batch_idx, :, :])
            # bi_seg_raw[torch.where(output['bi_seg'][batch_idx, :, :] > 0.1)] = 1
            # bi_seg_raw = bi_seg_raw.cpu().numpy()

            # for semantic segmentation
            bi_seg_raw = torch.squeeze(output['semantic_seg'][batch_idx, :, :])
            # locate = torch.where(bi_seg_raw>0)
            # print("lane seg location: ", locate)
            bi_seg_raw = bi_seg_raw.numpy()


            list_cls_label.append(cls_label)
            list_conf_label.append(conf_label)
            list_conf_label_raw.append(conf_label_raw)
            list_exist_label.append(exist_labl)
            list_coor_label.append(coor_label)

            list_cls_coor.append(cls_coor_pred)
            list_cls_coor_smooth.append(cls_coor_pred_smooth)
            list_cls_exp_smooth.append(cls_coor_exp_smooth)
            list_cls_offset_smooth.append(cls_coor_offset_smooth)

            list_endp_pred.append(endp_pred_raw)
            list_bi_seg_pred.append(bi_seg_raw)


        lane_maps.update({
            'conf_label': list_conf_label,
            'conf_label_raw': list_conf_label_raw,
            'cls_label': list_cls_label,
            'exist_label': list_exist_label,
            'coor_label': list_coor_label,
            'cls_coor_pred':list_cls_coor,
            'cls_coor_pred_smooth':list_cls_coor_smooth,
            'cls_exp_smooth': list_cls_exp_smooth,
            'cls_offset_smooth': list_cls_offset_smooth,
            'endp_by_cls': list_endp_pred,
            'bi_seg': list_bi_seg_pred
        })

        return lane_maps
    def add_offset_2_vertex(self, arr_cls_vertex, offset_map):
        arr_cls_vertex_copy = arr_cls_vertex.copy()
        for idx_lane in range(self.num_cls):
            for idx_h in range(self.row_size):
                if arr_cls_vertex[idx_lane, idx_h] > 0:
                    tmp_col = int((arr_cls_vertex[idx_lane, idx_h] - 4)/8.)
                    arr_cls_vertex_copy[idx_lane, idx_h] = (int((arr_cls_vertex[idx_lane, idx_h] - 4)/8.) + offset_map[idx_h, tmp_col]) * 8.
        return arr_cls_vertex_copy

    def Hausdorf_distance(self, line1, line2):
        dists = np.abs(line1 - line2)
        dists[np.where(line1 < 0)] = -1
        dists[np.where(line2 < 0)] = -1   # calculate the minimum distance for each overlapping vertex
        max_dist = np.max(dists)
        dist_valid = dists[np.where(dists >= 0)[0]]
        mean_dist = np.mean(dist_valid)
        return max_dist, mean_dist
    def line_NMS(self, arr_cls_vertex_in, semantic_map):
        active_id = 0
        num_lane, num_h = arr_cls_vertex_in.shape
        arr_cls_vertex = arr_cls_vertex_in.copy()
        while active_id < num_lane - 1:  # for every lane
            if len(np.where(arr_cls_vertex[active_id, :] > 0)[0]) < 2:
                active_id += 1
                continue
            for idx_l in range(active_id+1, num_lane): # check the Hausdorf distance between lines
                if len(np.where(arr_cls_vertex[idx_l, :] > 0)[0]) < 2:
                    continue
                else:
                    max_dist, mean_dist = self.Hausdorf_distance(arr_cls_vertex[active_id, :], arr_cls_vertex[idx_l, :])
                    if max_dist >= 0 and max_dist < 17 and mean_dist < 17: # two close lines, then merge them
                        for idx_h in range(num_h):
                            if arr_cls_vertex[active_id, idx_h] < 0 and arr_cls_vertex[idx_l, idx_h] < 0:   # two lines have no vertex at this location
                                continue
                            elif arr_cls_vertex[active_id, idx_h] > 0 and arr_cls_vertex[idx_l, idx_h] < 0: # line 1 has vertex; line 2 doesn't
                                continue
                            elif arr_cls_vertex[active_id, idx_h] < 0 and arr_cls_vertex[idx_l, idx_h] > 0:  # line 1 has no vertex; line 2 has vertex here
                                arr_cls_vertex[active_id, idx_h] = arr_cls_vertex[idx_l, idx_h]
                                arr_cls_vertex[idx_l, idx_h] = -1.
                            elif arr_cls_vertex[active_id, idx_h] > 0 and arr_cls_vertex[idx_l, idx_h] > 0:  # line 1 and line 2 both have vertex on this row
                                if semantic_map[idx_h*8+3, int(arr_cls_vertex[active_id, idx_h])] > semantic_map[idx_h*8+3, int(arr_cls_vertex[idx_l, idx_h])] :
                                    arr_cls_vertex[idx_l, idx_h] = -1.
                                else:
                                    arr_cls_vertex[active_id, idx_h] = arr_cls_vertex[idx_l, idx_h]
                                    arr_cls_vertex[idx_l, idx_h] = -1.
            active_id += 1
        return arr_cls_vertex

    def get_rgb_img_from_cls_map(self, cls_map, endp_map=None):
        temp_rgb_img = np.zeros((144, 144, 3), dtype=np.uint8)

        # for lanes
        for j in range(144):
            for i in range(144):
                idx_lane = int(cls_map[j,i])
                temp_rgb_img[j,i,:] = self.cfg.cls_lane_color[idx_lane] \
                                        if not (idx_lane == 255) else (0,0,0)

        # for endpoint
        # if endp_map is not None:
        #     # print("endpoint_map shape: ", endp_map.shape)
        #     for idx_lane in range(self.num_cls):
        #         two_vertex = np.where(endp_map[idx_lane, :, :] > 0)
        #         # print("the endpoint coordinate for class is: ", two_vertex)
        #         if len(two_vertex) > 0 and len(two_vertex[0]) and len(two_vertex[1]):
        #             # print("endpoint coordinate: ", two_vertex)
        #             for id in range(2):
        #                 h_top = 0 if (two_vertex[0][id] - 3) < 0 else (two_vertex[0][id] - 3)
        #                 h_bottom = (144 - 1) if (two_vertex[0][id] + 3) > (144 - 1) else (two_vertex[0][id] + 3)
        #                 w_left = 0 if (two_vertex[1][id] - 3) < 0 else (two_vertex[1][id] - 3)
        #                 w_right = (144 - 1) if (two_vertex[1][id] + 3) > (144 - 1) else (two_vertex[1][id] + 3)
        #                 temp_rgb_img[h_top: h_bottom, w_left: w_right, :] = self.cfg.cls_lane_color[idx_lane]
        #             # temp_rgb_img[two_vertex[0][0] - 3: two_vertex[0][0] + 3, two_vertex[1][0] - 3: two_vertex[1][0] + 3, :] = self.cfg.cls_lane_color[idx_lane]
        #             # temp_rgb_img[two_vertex[0][1] - 3: two_vertex[0][1] + 3, two_vertex[1][1] - 3: two_vertex[1][1] + 3, :] = self.cfg.cls_lane_color[idx_lane]
        return temp_rgb_img

    def get_rgb_img_from_cls_coors(self, cls_coors, endp_map=None):
        temp_rgb_img = np.zeros((1152, 1152, 3), dtype=np.uint8)

        # for lanes: n_lane * 144
        for idx_lane in range(self.num_cls):
            row_ids = np.arange(3, 1152, 8)
            for row in range(self.row_size):
                if(cls_coors[idx_lane, row] > 0):
                    row_org = row_ids[row]
                    col = int(cls_coors[idx_lane, row])
                    temp_rgb_img[row_org, col, :] = self.cfg.cls_lane_color[idx_lane]

        return temp_rgb_img

    def weighted_smooth_l1_loss(logits, target, mask=None):
        loss = F.smooth_l1_loss(logits, target, reduction='none')
        if mask is not None:
            loss = loss * mask
        return loss.mean()

    def init_weights(self, m, pretrained=None):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     logger = get_root_logger()  # old one : mmdetection
        #     # logger = Runner()     # replaced by the logger in this repo
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
        # elif pretrained is None:
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError('pretrained must be a str or None')

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
            # gt_endp_init = (data['initp'][batch_idx] + data['init_offset'][batch_idx]).cpu().numpy() * 8
            # gt_endp_end = (data['endp'][batch_idx] + data['end_offset'][batch_idx]).cpu().numpy() * 8
            gt_endp_init = (data['initp'][batch_idx]).cpu().numpy()
            gt_endp_end = (data['endp'][batch_idx]).cpu().numpy()
            gt_endp_coors = [[gt_endp_init[lane_id], gt_endp_end[lane_id]] for lane_id in range(self.num_cls)]
            raw_image_gray = self.get_gt_endp_on_raw_image(gt_endp_coors, raw_image_gray)
            raw_image_gray_coor = raw_image_gray.copy()
            raw_image_gray_coor_smooth = raw_image_gray.copy()
            raw_image_gray_bi_seg = raw_image_gray.copy()
            raw_image_gray_coor_exp = raw_image_gray.copy()
            raw_image_gray_coor_offset = raw_image_gray.copy()

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
            pred_lane_coors_offset = np.zeros((prop_size, self.row_size, 2))
            pred_lane_coors_offset[:, :, 0] = np.arange(3, 1152, 8)
            pred_lane_coors_offset[:, :, 1] = output['lane_maps']['cls_offset_smooth'][batch_idx]

            # upsample endpoint
            discrete_endp = output['lane_maps']['endp_by_cls'][batch_idx]
            endp_coors = np.where(discrete_endp == 1 )
            endp_coors = np.array(endp_coors)
            # if self.cfg.heads.endp_mode == 'Regr':
            #     pass
            # else:
            # endp_coors = endp_coors * 8 + 4
            # endp_coors = np.array(endp_coors)

            # binary segmentation result
            bi_seg_map = output['lane_maps']['bi_seg'][batch_idx]
            for semantic_id in [1, 2]:
                bi_seg_coors = np.where(bi_seg_map == semantic_id)
                bi_seg_coors = np.array(bi_seg_coors)

                # draw on image
                if bi_seg_coors.shape[1] > 0:
                    raw_image_gray_bi_seg = self.get_bi_seg_on_image(bi_seg_coors, raw_image_gray_bi_seg, semantic_id=semantic_id)

            if np.sum(output['lane_maps']['endp_by_cls'][batch_idx]) > 0 :
                raw_image_gray = self.get_endp_on_raw_image(endp_coors, raw_image_gray)
                raw_image_gray_coor = self.get_endp_on_raw_image(endp_coors, raw_image_gray_coor)
                raw_image_gray_coor_smooth = self.get_endp_on_raw_image(endp_coors, raw_image_gray_coor_smooth)

            for lane_id in range(prop_size):
                # print("endp size: ", len(endp_coors[lane_id][0]))
                # raw_image_gray = self.get_lane_on_raw_image(lane_coors[lane_id], lane_id, raw_image_gray)


                # for predicted lane coordinates:
                pred_lane = pred_lane_coors[lane_id, :, :]
                pred_lane = pred_lane[np.where(pred_lane[:, 1] > 0)]
                # print("predicted lane: ", pred_lane)
                raw_image_gray_coor = self.get_lane_on_raw_image_coordinates(pred_lane, lane_id, raw_image_gray_coor)

                # for predicted lane coordinates:
                pred_lane_smooth = pred_lane_coors_smooth[lane_id, :, :]
                pred_lane_smooth = pred_lane_smooth[np.where(pred_lane_smooth[:, 1] > 0)]
                # print("predicted lane: ", pred_lane)
                raw_image_gray_coor_smooth = self.get_lane_on_raw_image_coordinates(pred_lane_smooth, lane_id,
                                                                                 raw_image_gray_coor_smooth)

                # for predicted lane expectation:
                pred_lane_exp = pred_lane_coors_exp[lane_id, :, :]
                pred_lane_exp = pred_lane_exp[np.where(pred_lane_exp[:, 1] > 0)]
                raw_image_gray_coor_exp = self.get_lane_on_raw_image_coordinates(pred_lane_exp, lane_id,
                                                                                    raw_image_gray_coor_exp)

                # for predicted lane with offset:
                pred_lane_offset = pred_lane_coors_offset[lane_id, :, :]
                pred_lane_offset = pred_lane_offset[np.where(pred_lane_offset[:, 1] > 0)]
                raw_image_gray_coor_offset = self.get_lane_on_raw_image_coordinates(pred_lane_offset, lane_id,
                                                                                 raw_image_gray_coor_offset)

            list_lanes_on_img.append(raw_image_gray)
            list_org_lanes_on_img.append(raw_image_gray_coor)
            list_org_lanes_on_img_smooth.append(raw_image_gray_coor_smooth)
            list_org_lanes_smooth_vertex.append(pred_lane_coors_smooth)
            list_org_lanes_binary_seg.append(raw_image_gray_bi_seg)
            list_org_lanes_on_img_exp.append(raw_image_gray_coor_exp)
            list_org_lanes_on_img_offset.append(raw_image_gray_coor_offset)

            lane_maps.update({
                'pred_lanes_on_image': list_lanes_on_img,
                'pred_org_lanes_on_image': list_org_lanes_on_img,
                'pred_smooth_lanes_on_image': list_org_lanes_on_img_smooth,
                'pred_smooth_lane_vertex': list_org_lanes_smooth_vertex,
                'pred_bi_seg_on_image': list_org_lanes_binary_seg,
                'pred_exp_lanes_on_image':list_org_lanes_on_img_exp,
                'pred_offset_lanes_on_image': list_org_lanes_on_img_offset
            })

        return lane_maps

    def get_lane_on_raw_image(self, discrete_lane_coors, lane_id, raw_image):
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for p_idx in range(len(discrete_lane_coors[0]) - 1) :
            color = self.cfg.cls_lane_color[lane_id] \
                if not (lane_id == 255) else random_color
            pt1 = (discrete_lane_coors[1][p_idx], discrete_lane_coors[0][p_idx])
            pt2 = (discrete_lane_coors[1][p_idx + 1], discrete_lane_coors[0][p_idx + 1])
            cv2.line(raw_image, pt1, pt2, color=random_color, thickness=1)
        return raw_image

    def get_lane_on_raw_image_coordinates(self, discrete_lane_coors, lane_id, raw_image):
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255) )
        for p_idx in range(len(discrete_lane_coors) - 1) :  # shape: N, 2
            color = self.cfg.cls_lane_color[lane_id] \
                if not (lane_id < 255) else random_color
            pt1 = (int(discrete_lane_coors[p_idx][1]), int(discrete_lane_coors[p_idx][0]))
            pt2 = (int(discrete_lane_coors[p_idx + 1][1]), int(discrete_lane_coors[p_idx + 1][0]))
            cv2.line(raw_image, pt1, pt2, color=color, thickness=1)
        return raw_image

    def get_endp_on_raw_image(self, discrete_endp_coors, raw_image):
        # for p_idx in range(len(discrete_endp_coors[0]) - 1):
        lane_id = 0
        if (lane_id < 255):
            color = self.cfg.cls_lane_color[lane_id]
        else:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for idx in range(discrete_endp_coors.shape[1]):
            pt1 = (int(discrete_endp_coors[1][idx]), int(discrete_endp_coors[0][idx]))
            cv2.circle(raw_image, center=pt1, radius=7, color=color)
        return raw_image

    def get_gt_endp_on_raw_image(self, gt_endp_coors, raw_image):
        for lane_id in range(len(gt_endp_coors)):
            if (lane_id < 255):
                color = self.cfg.cls_lane_color[lane_id]
            else:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # print("gt coordinate: ", gt_endp_coors[lane_id])
            pt1 = (int(gt_endp_coors[lane_id][0][1]), int(gt_endp_coors[lane_id][0][0]))
            pt2 = (int(gt_endp_coors[lane_id][1][1]), int(gt_endp_coors[lane_id][1][0]))
            cv2.circle(raw_image, center=pt1, radius=5, color=color, thickness=cv2.FILLED)
            cv2.circle(raw_image, center=pt2, radius=5, color=color, thickness=cv2.FILLED)
        return raw_image

    def get_bi_seg_on_image(self, bi_seg_coors, raw_image, semantic_id=None):
        for idx in range(bi_seg_coors.shape[1]):
            if semantic_id is None:
                color = (255, 255, 255)
            else:
                if semantic_id == 1:
                    color = (0, 0, 255)
                elif semantic_id == 2:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
            # print("gt coordinate: ", gt_endp_coors[lane_id])
            pt1 = (int(bi_seg_coors[1][idx]), int(bi_seg_coors[0][idx]))
            # cv2.circle(raw_image, center=pt1, radius=1, color=color, thickness=cv2.FILLED)
            raw_image[pt1[1], pt1[0]] = color
        return raw_image

    def clip_lines(self, arr_cls, arr_endp):
        for idx_b in range(self.b_size):
            for idx_c in range(self.num_cls):
                two_vertex = np.where(arr_endp[idx_b, idx_c, :, :] > 0)
                if len(two_vertex) > 0 and len(two_vertex[0]) > 0 and len(two_vertex[1]) > 0:
                    endp_h1 = two_vertex[0][0]
                    endp_h2 = two_vertex[0][1]
                    arr_cls[idx_b, idx_c,0:endp_h1, :] = 0
                    arr_cls[idx_b, idx_c, endp_h2:, :] = 0
                    arr_cls[idx_b, idx_c, endp_h1, two_vertex[1][0]] = 1  # end point 1
                    arr_cls[idx_b, idx_c, endp_h2, two_vertex[1][1]] = 1  # end point 2

        return arr_cls