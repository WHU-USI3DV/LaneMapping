'''
* Thanks to AVELab, KAIST. All rights reserved.
* Modified by Xiaoxin (mixiaoxin@whu.edu.cn)
'''
import os.path as osp
import os
from re import M

import numpy
import numpy as np
import cv2
import torch
from glob import glob
import pickle
import laspy
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import skimage
import torchvision
from mmdet3d.structures import BasePoints
# from mmcv import DataContainer
from mmengine.structures import BaseDataElement
# from mmengine.dataset.base_dataset import Compose
from mmdet3d.datasets import get_loading_pipeline
from mmdet3d.datasets.transforms import Pack3DDetInputs

try:
    from baseline.datasets.registry import DATASETS
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from baseline.datasets.registry import DATASETS

@DATASETS.register_module
class LaserLaneProposal(Dataset):
    def __init__(self, data_root,  data_split_file, mode="valid", cfg=None):
        assert mode in {"train", "valid", "test", "single", "all", "infer_only"}
        sub_img_path = 'cropped_tiff'
        sub_label_path = 'labels'   # lidar bev annotations
        image_path = osp.join(data_root, sub_img_path)
        seq_path = osp.join(data_root, sub_label_path, 'sparse_seq')
        mask_path = osp.join(data_root, sub_label_path, 'sparse_semantic')
        instance_path = osp.join(data_root, sub_label_path, 'sparse_instance')
        orientation_path = osp.join(data_root, sub_label_path, 'sparse_orient')
        endp_path = osp.join(data_root, sub_label_path, 'sparse_endp')
        #

        image_list, seq_list, mask_list, instance_list, ori_list, endp_list, image_stem_list = \
            load_datadir(data_root, data_split_file, image_path, seq_path, mask_path, instance_path, \
                                                                 orientation_path, endp_path, mode)
        self.data_root = data_root
        self.data_split_file = data_split_file
        self.seq_len = len(seq_list)
        self.image_list = image_list
        self.seq_list = seq_list
        self.mask_list = mask_list
        self.instance_list = instance_list
        self.ori_list = ori_list
        self.endp_list = endp_list
        self.image_stem_list = image_stem_list
        self.mode = mode
        self.cfg = cfg


    def __len__(self):
        r"""
        :return: data length
        """
        return self.seq_len

    def __getitem__(self, idx):
        sample = dict()
        image_name = self.image_stem_list[idx]
        sample['image_name'] = image_name[0:11]
        sample['proj'] = self.load_img(idx=idx)
        
        if self.mode in {"train", "valid", "test", "single", "all"}:
            sample_prop = self.format_gt_column_proposal(idx)
            sample.update(sample_prop)
        
        return sample

    def load_img(self, idx):
        image_path = self.image_list[idx]
        img = Image.open(image_path)
        img = np.array(img, dtype=np.uint8)
        img = torchvision.transforms.functional.to_tensor(img).float()
        assert img.shape[1] == img.shape[2]
        
        # BEV
        if img.shape[0] > 3:
            img = img[0:3, :, : ]
        if self.cfg.dataset_color_augment:
                img = self.img_transform(img)
    
        return img
        
         
        
    def format_gt_column_proposal(self, idx):
        meta = dict()
        meta['mask'], meta['instance'], meta['instance_raw'], meta['ori'], meta['endp_map'] = load_label_image(
            self.mask_list[idx], self.instance_list[idx], ori_path=self.ori_list[idx], endp_path=self.endp_list[idx],
            downsample_instance=True, downsample_ratio=self.cfg.gt_downsample_ratio)
        _, _, meta['initp'], meta['endp'], _, _, meta['semantic'] = load_seq(self.seq_list[idx], downsample_vertex=False)
  
        # Labels on BEV
        if len(meta['initp']) != len(meta['endp']):
            print('init_point shape is {}; end_point shape is {}'.format(len(meta['initp']), len(meta['endp'])))
            print('end_point is: ', meta['endp'])

        meta['instance'] = np.where(meta['instance'] > self.cfg.number_lanes, 0, meta['instance'])
        meta['instance_raw'] = np.where(meta['instance_raw'] > self.cfg.number_lanes, 0, meta['instance_raw'])
        meta['ori'] = np.where(meta['instance_raw'] == 0, 0, meta['ori'])
        meta['mask'] = np.where(meta['instance_raw'] == 0, 0, meta['mask'])
        # refine the label: background == 255; lane instance from 0 - ~
        meta['instance'] = np.where(meta['instance'] == 0, 255, meta['instance']-1)
        meta['instance_raw'] = np.where(meta['instance_raw'] == 0, 255, meta['instance_raw'] - 1)

        sample = dict()
        sample['label'] = meta['instance']
        sample['label_raw'] = meta['instance_raw']
        sample['ori'] = meta['ori']
        sample['mask'] = meta['mask']
        sample['endp_map'] = meta['endp_map']

        # align the end points and semantics length:
        sample['initp'] = np.zeros((self.cfg.number_lanes, 2), dtype=np.float64)
        sample['endp'] = np.zeros((self.cfg.number_lanes, 2), dtype=np.float64)
        sample['semantic'] = np.zeros((self.cfg.number_lanes), dtype=np.float64)
        if(len(meta['initp']) > self.cfg.number_lanes):
            sample['initp'] = np.array(meta['initp'][:self.cfg.number_lanes], dtype=np.float64)
            sample['endp'] = np.array(meta['endp'][:self.cfg.number_lanes], dtype=np.float64)
            sample['semantic'] = np.array(meta['semantic'][:self.cfg.number_lanes], dtype=np.float64)
        else:
            sample['initp'][:len(meta['initp'])]= np.array(meta['initp'], dtype=np.float64)
            sample['endp'][:len(meta['endp'])] = np.array(meta['endp'], dtype=np.float64)
            sample['semantic'][:len(meta['semantic'])] = np.array(meta['semantic'], dtype=np.float64)

        # visualization:
        # view_instance_endpoint(sample['label_raw'], sample['initp'], sample['endp'])
        ###############################################################################################################
        # From the image sample, get the column proposal ground truth:
        # Aim: gt_proposal = torch.zeros((self.b_size, proposal_size, 2))
        #         gt_exist = torch.zeros((self.b_size, proposal_size, self.row_size))
        #         gt_coors = torch.zeros((self.b_size, proposal_size, self.row_size))
        #         gt_offset = torch.zeros((self.b_size, proposal_size, self.row_size, (self.prop_fea_width)))
        #         gt_offset_mask = torch.zeros((self.b_size, proposal_size, self.row_size, (self.prop_fea_width)))
        #         gt_bi_seg = torch.zeros((self.b_size, proposal_size, self.row_size *8, (self.prop_fea_width) *8))
        #         lb_lc_orient = torch.zeros((self.b_size, self.row_size, delf.column_size))
        #         lb_lc_endp = torch.zeros((self.b_size, self.row_size*8, delf.column_size*8))
        lb_lc_ext, lb_lc_cls, lb_lc_offset, lb_lc_offset_mask, lb_lc_endp, lb_lc_orient, bi_org_lane_label, semantic_org_lane_label = \
            self.get_lane_exist_and_cls_wise_and_endpoints_maps(torch.tensor(sample['label_raw']),
                                                                endp_map=sample['endp'],
                                                                orient_label=sample['ori'],
                                                                line_semantic=sample['semantic'],
                                                                merge_connect_lines=True,
                                                                init_pts=sample['initp'], terminal_pts=sample['endp'],
                                                                is_ret_list=False)

        # 1. caluculate the minimumn mean distance from the each proposal to the gt, keep this distance and GT id
        # distance size: (b, n_proposal)
        # GT id: (b, n_proposal)
        proposal_size = self.cfg.heads.num_prop  # 72
        col_index = self.cfg.heads.prop_width * torch.arange(proposal_size, dtype=torch.float32)  # 0, 2, 4, 6 ,,,,
        dist_prop_line = col_index.repeat(self.cfg.heads.row_size, self.cfg.number_lanes, 1)
        dist_prop_line = dist_prop_line.permute(2, 0, 1)  # [72, 144, 9]: for every proposal has its own initial value: 0, 2, 4,,,
        dist_prop_line_valid = torch.ones_like(dist_prop_line)
        prop_cls = lb_lc_cls.repeat(proposal_size, 1, 1)  # [72, 9, 144]
        # begin: constraint the GT for proposal buffer
        for p_id in range(proposal_size):
            left_border = self.cfg.heads.prop_width * p_id - (self.cfg.heads.prop_half_buff)
            right_border = self.cfg.heads.prop_width * p_id + self.cfg.heads.prop_half_buff + self.cfg.heads.prop_width
            outside_l = torch.where((prop_cls[p_id, ...] < left_border) | (prop_cls[p_id, ...] > right_border))
            prop_cls[p_id, ...][outside_l] = -1
        # end: constraint the GT for proposal buffer

        prop_cls = prop_cls.permute(0, 2, 1)  # [72, 144, 9]
        invalid_cls_loc = torch.where(prop_cls < 0)
        dist_prop_line -= prop_cls
        dist_prop_line[invalid_cls_loc] = 0.  # distance from invalid vertexes is set as 0. For conenience to calculating the mean invalid distance
        dist_prop_line_valid[invalid_cls_loc] = 0  # other vertexes (1) are valid, contribute nothing to mean distance

        dist_prop_line = torch.abs(dist_prop_line)
        dist_prop_line = torch.sum(dist_prop_line, dim=1)  # [72, 9]
        line_valid_sum = torch.sum(dist_prop_line_valid, dim=1)  # [72, 9]
        line_valid_sum[torch.where(line_valid_sum < 1)] = 1  # avoid Nan divide
        dist_prop_line = dist_prop_line / line_valid_sum  # average distance to each gt lane
        dist_prop_line[torch.where(dist_prop_line == 0.)] = 143.  # distance from the proposal to the empty line instance is 143.

        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 0)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value
        # expand the offset map, bi_seg_map, endp_map
        zero_pad = torch.nn.ZeroPad2d(padding=(self.cfg.heads.prop_half_buff, self.cfg.heads.prop_half_buff, 0, 0))
        zero_pad_raw = torch.nn.ZeroPad2d(padding=(int(self.cfg.heads.prop_half_buff*8), int(self.cfg.heads.prop_half_buff*8), 0, 0))
        lb_lc_offset = zero_pad(lb_lc_offset)  # [9, 144, 144 + 8 + 9]
        lb_lc_offset_mask = zero_pad(lb_lc_offset_mask)  # # [9, 144, 144 + 8 + 8]
        bi_org_lane_label = zero_pad_raw(bi_org_lane_label)  # [9, 1152, 1152 + 8*8 + 8*8]


        # for every proposal, the min distance to the GT-line
        dist_prop_line_min = torch.amin(dist_prop_line, dim=-1)  # [72, 9]  --> (batch, proposal_size)
        # for every proposal, the corresponding GT-line ID to each proposal
        dist_prop_line_id = torch.argmin(dist_prop_line, dim=-1)  # [72, 9]  --> (batch, proposal_size)

        # objective_loss = torch.sum(dist_prop_line)
        prop_fea_width = int(self.cfg.heads.prop_width + self.cfg.heads.prop_half_buff*2)
        gt_proposal = torch.zeros((proposal_size, 2))
        gt_exist = torch.zeros((proposal_size, self.cfg.heads.row_size))
        gt_coors = torch.zeros((proposal_size, self.cfg.heads.row_size))
        gt_offset = torch.zeros((proposal_size, self.cfg.heads.row_size, prop_fea_width))
        gt_offset_mask = torch.zeros((proposal_size, self.cfg.heads.row_size, prop_fea_width))
        gt_bi_seg = torch.zeros((proposal_size, int(self.cfg.heads.row_size * self.cfg.gt_downsample_ratio), int(prop_fea_width * self.cfg.gt_downsample_ratio)))

        for id_p in range(proposal_size):
            gt_exist[id_p, :] = lb_lc_ext[dist_prop_line_id[id_p], :]
            gt_coors[id_p, :] = lb_lc_cls[dist_prop_line_id[id_p], :] - (self.cfg.heads.prop_width * id_p - self.cfg.heads.prop_half_buff)  # absolute coordinate
            prop_min_id = self.cfg.heads.prop_width * id_p
            prop_max_id = self.cfg.heads.prop_width * id_p + prop_fea_width
            org_min_id = self.cfg.gt_downsample_ratio * prop_min_id
            org_max_id = self.cfg.gt_downsample_ratio * prop_max_id
            gt_offset[id_p, :, :] = lb_lc_offset[dist_prop_line_id[id_p], :, prop_min_id:prop_max_id]
            gt_offset_mask[id_p, :, :] = lb_lc_offset_mask[dist_prop_line_id[id_p], :, prop_min_id:prop_max_id]
            gt_bi_seg[id_p, :, :] = bi_org_lane_label[dist_prop_line_id[id_p], :, org_min_id: org_max_id]
        lb_lc_cls_raw = lb_lc_cls.clone()
        lb_lc_cls_raw[torch.where(lb_lc_cls_raw > -1.)] *= self.cfg.gt_downsample_ratio
        ###############################################################################################################
        sample_propsal = dict()
        if self.mode is not 'train':  # data augmentation
            sample_propsal['initp'] = sample['initp']  # only for evaluation
            sample_propsal['endp'] = sample['endp']    # only for evaluation
            sample_propsal['mask'] = sample['mask']     # only for evaluation
            
        sample_propsal['label_raw'] = sample['label_raw']
        sample_propsal['semantic_label_raw'] = torch.tensor(sample['mask'])
        sample_propsal['endp_map'] = torch.tensor(sample['endp_map'])
        sample_propsal['lc_orient'] = lb_lc_orient
        sample_propsal['lc_coor_raw'] = lb_lc_cls_raw
        
        sample_propsal['prop_bi_seg'] = gt_bi_seg
        sample_propsal['prop_obj'] = gt_proposal
        sample_propsal['prop_ext'] = gt_exist
        sample_propsal['prop_coor'] = gt_coors
        sample_propsal['prop_offset'] = gt_offset
        sample_propsal['prop_offset_mask'] = gt_offset_mask

    
        return sample_propsal

        # add data augmentation: brightness, contrast, hue
    def img_transform(self, img):
        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                torchvision.transforms.ConvertImageDtype(torch.float),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )        
        out_img = transform_train(img)
        return out_img


    def get_lane_exist_and_cls_wise_and_endpoints_maps(self,
                                                       lanes_label_raw,
                                                       endp_map=None,
                                                       orient_label=None,
                                                       line_semantic=None,
                                                       merge_connect_lines=False,
                                                       init_pts=None, terminal_pts=None,
                                                       is_ret_list=True):
        # label_tensor = self.line_label_formatting(lanes_label_down, is_flip=self.flip_label) # channel0 = line number, channel1 = confidence
        img_h = self.cfg.heads.row_size  # 144
        img_w = self.cfg.heads.row_size  # 144
        n_cls = self.cfg.number_lanes

        line_label = self.line_label_formatting(lanes_label_raw, is_flip=self.cfg.flip_label)
        _, org_img_h, org_img_w = line_label.shape  # 2, 1152, 1152


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
        lb_cls_raw = line_label[0, :, :]  # 0~(num_cls-1): cls, num_cls: background
        ret_exist = torch.zeros((n_cls, img_h))  # (num_cls, 144)
        ret_maps = torch.zeros((n_cls, img_h))
        ret_offset_maps = torch.zeros((n_cls, img_h, img_w))
        ret_offset_mask = torch.zeros((n_cls, img_h, img_w))
        ret_endpoint_maps = torch.tensor(endp_map)
        ret_orient_maps = torch.zeros((img_h, img_w))
        ret_bi_seg = self.binary_label_formatting(lanes_label_raw, is_flip=self.cfg.flip_label)   #[1152, 1152], binary segmentation
        ret_semantic_seg = torch.zeros_like(ret_bi_seg)

        # for semantic segmentation
        for c_id in range(self.cfg.number_lanes):
            if line_semantic[c_id] > 0:
                ppp = torch.where(ret_bi_seg[c_id, :, :] > 0)
                ret_semantic_seg[c_id, ppp[0], ppp[1]] = float(line_semantic[c_id])

        ret_exist, ret_maps, ret_offset_maps, ret_offset_mask, ret_orient_maps = \
            self.get_line_existence_and_cls_wise_maps_per_batch(lb_cls_raw,
                                                                n_cls=self.cfg.number_lanes,
                                                                raw_orient_map=orient_label,
                                                                line_semantic=line_semantic)

        if merge_connect_lines:
            for lane_id1 in range(self.cfg.number_lanes):
                end_p1 = terminal_pts[lane_id1, :]
                if (end_p1[0]) > 0 and (end_p1[1] > 0):  # lane_id1 exists
                    for lane_id2 in range(self.cfg.number_lanes):
                        if lane_id2 == lane_id1:  # lane_id2 is not lane_id1
                            continue
                        start_p2 = init_pts[lane_id2, :]
                        # lane_id2 exist, the start point of lane_id2 is close to the terminate point of lane_id1, then merge
                        if (start_p2[0] > 0) and (start_p2[1] > 0) and (abs(end_p1[0] - start_p2[0]) < 2) and (
                                abs(end_p1[1] - start_p2[1]) < 2):
                            ext_row_ids = torch.where(ret_exist[lane_id2, :] > 0)[0]
                            # merge existence
                            ret_exist[lane_id1, ext_row_ids] = ret_exist[lane_id2, ext_row_ids]

                            # merge coordinate
                            ret_maps[lane_id1, ext_row_ids] = ret_maps[lane_id2, ext_row_ids]
                            ret_offset_maps[lane_id1, ext_row_ids, :] = ret_offset_maps[lane_id2, ext_row_ids, :]
                            ret_offset_mask[lane_id1, ext_row_ids, :] = ret_offset_mask[lane_id2, ext_row_ids, :]

                            # merge binary segmentation label
                            pixels = torch.where(ret_bi_seg[lane_id2, :, :] > 0)
                            ret_bi_seg[lane_id1, pixels[0], pixels[1]] = 1

                            # merge semantic segmentation label
                            ret_semantic_seg[lane_id1, pixels[0], pixels[1]] = line_semantic[lane_id2]
                            ret_exist[lane_id2, ext_row_ids] = 0
                            ret_maps[lane_id2, ext_row_ids] = -1
                            ret_offset_maps[lane_id2, ext_row_ids, :] = 0
                            ret_offset_mask[lane_id2, ext_row_ids, :] = 0
                            init_pts[lane_id2, :] = 0
                            terminal_pts[lane_id2, :] = 0
                            ret_bi_seg[lane_id2, :, :] = 0
                            ret_semantic_seg[lane_id2, pixels[0], pixels[1]] = 0


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
        return ret_exist, ret_maps, ret_offset_maps, ret_offset_mask, ret_endpoint_maps, ret_orient_maps, \
            ret_bi_seg, ret_semantic_seg

    def line_label_formatting(self, raw_label, is_flip=True):
        # Output image: top-left of the image is farthest-left
        label_h, label_w = raw_label.shape  # numpy.Size([1152, 1152])
        line_label = torch.zeros((2, label_h, label_w), dtype=torch.int)

        if is_flip:
            line_label[0, ...] = torch.flip(raw_label, (0, 1))
        else:
            line_label[0, ...] = raw_label
        line_label[0, ...][torch.where(line_label[0, ...] == 255)] = self.cfg.number_lanes  # label id
        line_label[1, ...][torch.where(line_label[0, ...] < self.cfg.number_lanes)] = 1  # existence
        return line_label

    def binary_label_formatting(self, raw_label, is_flip=False):
        # Output image: top-left of the image is farthest-left
        label_h, label_w = raw_label.shape  # np.Size([1152, 1152])
        bi_semantic_label = torch.zeros((self.cfg.number_lanes, label_h, label_w))  # torch.Size([4, n_lane, 1152, 1152])
        for lane_id in range(self.cfg.number_lanes):
            coor_idx = torch.where(raw_label == lane_id)
            bi_semantic_label[lane_id, coor_idx[0], coor_idx[1]] = 1
        if is_flip:
            bi_semantic_label = torch.flip(bi_semantic_label, (1, 2))

        return bi_semantic_label

    def get_line_existence_and_cls_wise_maps_per_batch(self, lb_cls, n_cls=6, img_h=144, img_w=144, downsample=True,
                                                       raw_orient_map=None, line_semantic=None):
        # print(lb_cls.shape) # torch.Size([144, 144])
        cls_maps_raw = torch.zeros((n_cls, img_h * 8))
        cls_maps = torch.zeros((n_cls, img_h))

        line_ext = torch.zeros((n_cls, img_h))
        orient_map = torch.zeros((img_h, img_w))

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

            if raw_orient_map is not None:
                orient_buff_r = 3
                down_row = (torch.where(cls_maps[idx_cls, :] > 0.))[0]
                if (len(down_row) < 2):
                    continue
                pixel_num = len(down_row)
                down_col = cls_maps[idx_cls, down_row].long()
                down_col_left = down_col - orient_buff_r
                down_col_left = torch.where(down_col_left < 0, 0, down_col_left)
                down_col_right = down_col + orient_buff_r
                down_col_right = torch.where(down_col_right > (img_w - 1), img_w - 1, down_col_right)
                up_row = down_row * 8 + 3
                up_col = (cls_maps[idx_cls, down_row] * 8).long()
                for idx_p in range(pixel_num):
                    orient_map[down_row[idx_p], down_col_left[idx_p]:down_col_right[idx_p]] = raw_orient_map[up_row[idx_p], up_col[idx_p]]
                    # orient_map[down_row[idx_p], down_col[idx_p]] = raw_orient_map[up_row[idx_p], up_col[idx_p]]
                    # orient_map[down_row[idx_p], down_col_left[idx_p]] = raw_orient_map[up_row[idx_p], up_col[idx_p]]
                    # orient_map[down_row[idx_p], down_col_right[idx_p]] = raw_orient_map[up_row[idx_p], up_col[idx_p]]
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

        if raw_orient_map is not None:
            return line_ext, cls_maps, cls_offset_maps, cls_offset_mask, orient_map
        else:
            return line_ext, cls_maps, cls_offset_maps, cls_offset_mask



def load_datadir(data_root, data_split_file, image_path, seq_path, mask_path, instance_path, ori_path, endp_path, mode):
    # with open(osp.join(data_root, 'data_split-shuffle.json'), 'r') as jf:   # data_split-shuffle
    with open(osp.join(data_root, data_split_file), 'r') as jf:   # data_split-shuffle
        json_list = json.load(jf)
    train_list = list(json_list['train'])
    test_list = list(json_list['test'])  # all data: pretrain;   # test/val: test
    val_list = list(json_list['valid'])
    single_list = list(json_list['single'])
    all_list = list(json_list['pretrain'])

    if mode == 'single':
        json_list = [x + '.json' for x in single_list]
    elif mode == 'valid':
        json_list = [x + '.json' for x in val_list][:150]
    elif mode == 'test':
        random.shuffle(test_list)
        json_list = [x + '.json' for x in test_list]
    elif mode == 'all' or mode == 'infer_only':
        json_list = [x + '.json' for x in all_list]
    else:
        json_list = [x + '.json' for x in train_list]

    seq_list = []
    image_list = []
    mask_list = []
    instance_list = []
    ori_dir = []
    endp_dir = []
    image_stem_list = []

    for jsonf in json_list:
        image_list.append(os.path.join(image_path, jsonf[:-4] + 'png'))
        seq_list.append(os.path.join(seq_path, jsonf))
        mask_list.append(os.path.join(mask_path, jsonf[:-4] + 'png'))
        instance_list.append(os.path.join(instance_path, jsonf[:-4] + 'png'))
        ori_dir.append(os.path.join(ori_path, jsonf[:-4] + 'png'))
        endp_dir.append(os.path.join(endp_path, jsonf[:-4]+'png'))
        image_stem_list.append(jsonf[:-4])
    return image_list, seq_list, mask_list, instance_list, ori_dir, endp_dir, image_stem_list


def load_seq(seq_path, cfg=None, downsample_vertex=True, downsample_ratio=8):
    r'''
    Load the dense sequence of the current image. It may contains the vertices of multiple boundary instances.
    '''
    with open(seq_path) as json_file:
        load_json = json.load(json_file)
        data_json = load_json

    seq_lens = []
    init_points = []
    end_points = []
    init_offsets = []
    end_offsets = []
    semantics = []

    # print("data json: ", len(data_json))
    for area in data_json:
        # print("area: ", area)
        seq_lens.append(len(area['seq']))
        init_points.append(area['init_vertex'])
        end_points.append(area['end_vertex'])
        semantics.append(area['semantic'])


    # if not cfg.gt_init_vertex and cfg.test:
    #     with open(os.path.join(cfg.init_vertex_dir, seq_path[-16:]), 'r') as jf:
    #         init_points = json.load(jf)
    # print("seq path: ", seq_path)
    # print("seq lens", seq_lens==[])

    seq = np.zeros((len(seq_lens), max(seq_lens), 2))
    for idx, area in enumerate(data_json):
        if seq_lens[idx] == 0:
            continue
        seq[idx, :seq_lens[idx]] = [x[0:2] for x in area['seq']]
    # seq = torch.FloatTensor(seq)
    # print("init points before downsampe: ", init_points)
    if downsample_vertex:
        seq /= downsample_ratio
        init_points = [[int(point[0]/downsample_ratio), int(point[1]/downsample_ratio)] for point in init_points]
        end_points = [[int(point[0]/downsample_ratio), int(point[1]/downsample_ratio)] for point in end_points]
        init_offsets = [[point[0]/downsample_ratio - int(point[0]/downsample_ratio), point[1]/downsample_ratio - int(point[1]/downsample_ratio)] for point in init_points]
        end_offsets = [[point[0] / downsample_ratio - int(point[0] / downsample_ratio), point[1] / downsample_ratio - int(point[1] / downsample_ratio)] for point in
                        end_points]


    return seq, seq_lens, init_points, end_points, init_offsets, end_offsets, semantics


def load_label_image(mask_path, instance_path, ori_path=None, endp_path=None, downsample_instance=True, downsample_ratio=8):
    
    # mask = np.array(Image.open(mask_path))[:,:,0]
    mask = np.array(Image.open(mask_path))  # for 1 channel
    # mask = mask / 128  # for 2 classes, solid is 1, dashed is 2
    mask[np.where(mask==128)] = 1
    mask[np.where(mask == 255)] = 2
    # kernel = skimage.morphology.rectangle(ncols=3, nrows=1)
    # mask = skimage.morphology.dilation(mask, kernel)
    # cv2.imshow('mask before dilation', mask)
    # cv2.imshow('mask after dilation', mask_dilat)
    # cv2.waitKey(0)

    instance = np.array(Image.open(instance_path)) # instance.shape: [1, h, w]
    instance_down = np.array(instance.shape)
    if downsample_instance:
        instance_down = skimage.measure.block_reduce(instance, block_size=downsample_ratio, func=np.max)

    endp_map = np.zeros_like(mask, dtype=np.float32)
    if endp_path is not None:
        endp_map = np.array(Image.open(endp_path), dtype=np.float32)
        endp_map /= 255.

    ori_map = np.zeros_like(mask, dtype=np.float32)
    if ori_path is not None:
        # ori = np.array(Image.open(ori_path))[:,:,0]
        ori_map = np.array(Image.open(ori_path))  # for 1 channel

    return mask, instance_down, instance, ori_map, endp_map

def read_las(filepath):
    las = laspy.read(filepath)
    positions = np.stack((las.x, las.y, las.z), axis=1)
    # colors = np.stack((las.red, las.green, las.blue), axis=1)
    # to load gps_time but not available 
    intensity = np.expand_dims((las.intensity), axis=1)
    # intensity2 = intensity.reshape((-1, 1)) 
    # to normalize intensity
    inten_min = 800.0
    inten_max = 33000.0
    intensity = np.clip(intensity, a_min=inten_min, a_max=inten_max)
    intensity = (intensity - inten_min) / inten_max
    
    lidar_pts = torch.tensor(np.hstack((positions, intensity)))
    if lidar_pts.size(0) < 5:
        print('lidar pts: ', lidar_pts)
        print(filepath)
        exit()   
    return lidar_pts # N 4 (x, y, z, intensity)

def view_instance_endpoint(raw_image, instance, init_pts, end_pts, cfg=None):
    # to visualize downsampled instance image fed into network
    # instance[np.where(instance < 255)] += 200
    # instance[np.where(instance == 255)] = 0

    # to visualize raw image
    # instance[np.where(instance > 0)] += 200

    h, w = instance.shape
    print("h is {}, w is {}".format(h, w))
    # BGR Format to OpenCV
    cls_lane_color = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (42, 42, 128),  # add by xiaoxin
        (158, 168, 3),  # add by xiaoxin
        (240, 32, 160),  # add by xiaoxin
        (84, 46, 8),  # add by xiaoxin
        (255, 97, 0),  # add by xiaoxin
        (100, 255, 0)  # add by xiaoxin
    ]
    instance_img = np.zeros((h, w, 3))
    raw_img = raw_image.permute(1, 2, 0).numpy()
    print("raw image shape: ", raw_img.shape)

    for lane_id in range(1, 15):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        instance_img[np.where(instance==lane_id)] = color
        raw_img[np.where(instance == lane_id)] = color



    # print('init_pts,shape: ', init_pts.shape)
    for lane_id, pt in enumerate(init_pts):
        pt *= 8
        h_top = 0 if  (int(pt[0]) - 3) < 0 else (int(pt[0]) - 3)
        h_bottom = (h-1) if (int(pt[0]) + 3) > (h -1) else (int(pt[0]) + 3)
        w_left = 0 if  (int(pt[1]) - 3) < 0 else (int(pt[1]) - 3)
        w_right = (w-1) if (int(pt[1]) + 3) > (w -1) else (int(pt[1]) + 3)
        instance[h_top : h_bottom, w_left : w_right] = 255
        instance_img[h_top:h_bottom, w_left:w_right, :] = cls_lane_color[lane_id]
        raw_img[h_top:h_bottom, w_left:w_right, :] = cls_lane_color[lane_id]
    for lane_id, pt in enumerate(end_pts):
        pt *= 8
        h_top = 0 if (int(pt[0]) - 3) < 0 else (int(pt[0]) - 3)
        h_bottom = (h - 1) if (int(pt[0]) + 3) > (h - 1) else (int(pt[0]) + 3)
        w_left = 0 if (int(pt[1]) - 3) < 0 else (int(pt[1]) - 3)
        w_right = (w - 1) if (int(pt[1]) + 3) > (w - 1) else (int(pt[1]) + 3)
        instance[h_top: h_bottom, w_left: w_right] = 105
        instance_img[h_top:h_bottom, w_left:w_right, :] = cls_lane_color[lane_id]
        raw_img[h_top:h_bottom, w_left:w_right, :] = cls_lane_color[lane_id]

    cv2.imshow('instance_endpoint', instance_img)
    cv2.imshow('raw image with instance_endpoint', raw_img)
    cv2.waitKey(0)





if __name__=='__main__':
    image_path = "/data/mxx/data/LaserLane/All-ordered/cropped_tiff/190712_0519.png"
    seq_path = "/data/mxx/data/LaserLane/All-ordered/labels/sparse_seq/190712_0519.json"
    semantic_path = "/data/mxx/data/LaserLane/All-ordered/labels/sparse_semantic/190712_0519.png"
    instance_path = "/data/mxx/data/LaserLane/All-ordered/labels/sparse_instance/190712_0519.png"


    raw_img, semantic_img,_, instance_img = load_image(image_path, semantic_path, instance_path)
    _, _, init_pts, end_pts, _, _, _ = load_seq(seq_path, cfg=None, downsample_vertex=False)
    view_instance_endpoint(raw_img, instance_img, init_pts, end_pts)


