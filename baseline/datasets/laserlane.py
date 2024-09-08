'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
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
import open3d as o3d
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import skimage
import torchvision

try:
    from baseline.datasets.registry import DATASETS
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from baseline.datasets.registry import DATASETS

@DATASETS.register_module
class LaserLane(Dataset):
    def __init__(self, data_root,  mode="valid", cfg=None):
        assert mode in {"train", "valid", "test", "single", "all"}
        sub_img_path = 'cropped_tiff'
        sub_label_path = 'labels'
        image_path = osp.join(data_root, sub_img_path)
        seq_path = osp.join(data_root, sub_label_path, 'sparse_seq')
        mask_path = osp.join(data_root, sub_label_path, 'sparse_semantic')
        instance_path = osp.join(data_root, sub_label_path, 'sparse_instance')
        orientation_path = osp.join(data_root, sub_label_path, 'sparse_orient')
        endp_path = osp.join(data_root, sub_label_path, 'sparse_endp')
        #
        # if mode:
        #     mode = 'test'
        image_list, seq_list, mask_list, instance_list, ori_list, endp_list, image_stem_list = load_datadir(data_root, image_path, \
                                                                 seq_path, mask_path, instance_path, \
                                                                 orientation_path, endp_path, mode)
        self.data_root = data_root
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
        image_name = self.image_stem_list[idx]
        meta = dict()

        meta['image_name'] = image_name[0:11]
        sample = dict()

        sample['proj'], meta['mask'], meta['instance'], meta['instance_raw'], meta['ori'], meta['endp_map'] = load_image(
            self.image_list[idx], self.mask_list[idx], self.instance_list[idx], ori_path=self.ori_list[idx], endp_path=self.endp_list[idx])
        _, _, meta['initp'], meta['endp'], _, _, meta['semantic'] = load_seq(self.seq_list[idx],
                                                                                              downsample_vertex=False)

        if self.mode == 'train':
            pass
        else:
            sample['image_name'] = meta['image_name']


        if len(meta['initp']) != len(meta['endp']):
            print('init_point shape is {}; end_point shape is {}'.format(len(meta['initp']), len(meta['endp'])))
            print('end_point is: ', meta['endp'])

        # print("instance id: ", meta['instance'][np.where(meta['instance'] > 0)])

        meta['instance'] = np.where(meta['instance'] > self.cfg.number_lanes, 0, meta['instance'])
        meta['instance_raw'] = np.where(meta['instance_raw'] > self.cfg.number_lanes, 0, meta['instance_raw'])
        meta['ori'] = np.where(meta['instance_raw'] == 0, 0, meta['ori'])
        meta['mask'] = np.where(meta['instance_raw'] == 0, 0, meta['mask'])
        # refine the label: background == 255; lane instance from 0 - ~
        meta['instance'] = np.where(meta['instance'] == 0, 255, meta['instance']-1)
        meta['instance_raw'] = np.where(meta['instance_raw'] == 0, 255, meta['instance_raw'] - 1)

        if sample['proj'].shape[0] > 3:
            sample['proj'] = sample['proj'][0:3, :, : ]

        sample['label'] = meta['instance']
        sample['label_raw'] = meta['instance_raw']
        sample['ori'] = meta['ori']
        sample['mask'] = meta['mask']
        sample['endp_map'] = meta['endp_map']

        # align the end points length:
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
        return sample

def load_datadir(data_root, image_path, seq_path, mask_path, instance_path, ori_path, endp_path, mode):
    with open(osp.join(data_root, 'data_split-shuffle.json'), 'r') as jf:   # data_split-shuffle
        json_list = json.load(jf)
    train_list = list(json_list['train'])
    test_list = list(json_list['test'])  # all data: pretrain;   # test/val: test
    val_list = list(json_list['valid'])
    single_list = list(json_list['single'])
    pretrain_list = list(json_list['pretrain'])

    if mode == 'single':
        json_list = [x + '.json' for x in single_list]
    elif mode == 'valid':
        json_list = [x + '.json' for x in val_list][:150]
    elif mode == 'test':
        random.shuffle(test_list)
        json_list = [x + '.json' for x in test_list]
    elif mode == 'all':
        json_list = [x + '.json' for x in pretrain_list]
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


def load_seq(seq_path, cfg=None, downsample_vertex=True):
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
        seq /= 8
        init_points = [[int(point[0]/8), int(point[1]/8)] for point in init_points]
        end_points = [[int(point[0]/8), int(point[1]/8)] for point in end_points]
        init_offsets = [[point[0]/8 - int(point[0]/8), point[1]/8 - int(point[1]/8)] for point in init_points]
        end_offsets = [[point[0] / 8 - int(point[0] / 8), point[1] / 8 - int(point[1] / 8)] for point in
                        end_points]


    return seq, seq_lens, init_points, end_points, init_offsets, end_offsets, semantics


def load_image(image_path, mask_path, instance_path, ori_path=None, endp_path=None, downsample_instance=True):
    img = Image.open(image_path)
    img = np.array(img, dtype=np.uint8)
    img = torchvision.transforms.functional.to_tensor(img).float()

    assert img.shape[1] == img.shape[2]
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
        instance_down = skimage.measure.block_reduce(instance, block_size=8, func=np.max)

    endp_map = np.zeros_like(mask, dtype=np.float32)
    if endp_path is not None:
        endp_map = np.array(Image.open(endp_path), dtype=np.float32)
        endp_map /= 255.

    ori_map = np.zeros_like(mask, dtype=np.float32)
    if ori_path is not None:
        # ori = np.array(Image.open(ori_path))[:,:,0]
        ori_map = np.array(Image.open(ori_path))  # for 1 channel

    return img, mask, instance_down, instance, ori_map, endp_map


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


