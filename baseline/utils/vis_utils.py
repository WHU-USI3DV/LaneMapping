'''
* author: Xiaoxin Mi
* e-mail: mixiaoxin@whu.edu.cn
'''
import os
import sys
import pickle
import numpy as np
import cv2
import random
import torch
import open3d as o3d


__all__ = ['get_bi_seg_on_image', 'get_endp_on_raw_image',\
             'get_metrics_conf_cls', 'get_lane_on_raw_image_coordinates', \
           'get_semantic_lane_on_raw_image_coordinates', 'get_endp_on_raw_image', \
           'get_gt_endp_on_raw_image']

def get_bi_seg_on_image(bi_seg_coors, raw_image, semantic_id=None):
    for idx in range(len(bi_seg_coors[0])):
        if semantic_id is None:
            color = (255, 255, 255)
        else:
            if semantic_id == 1:
                color = (255, 0, 0)
            elif semantic_id == 2:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
        # print("gt coordinate: ", gt_endp_coors[lane_id])
        pt1 = (int(bi_seg_coors[1][idx]), int(bi_seg_coors[0][idx]))
        # cv2.circle(raw_image, center=pt1, radius=1, color=color, thickness=cv2.FILLED)
        raw_image[pt1[1], pt1[0]] = color
    return raw_image

def get_endp_on_raw_image(discrete_endp_coors, raw_image):
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    for idx in range(len(discrete_endp_coors[0])):
        pt1 = (int(discrete_endp_coors[1][idx]), int(discrete_endp_coors[0][idx]))
        cv2.circle(raw_image, center=pt1, radius=11, color=color, thickness=cv2.FILLED)
    return raw_image

def get_lane_on_raw_image_coordinates(discrete_lane_coors, lane_id, raw_image, color=None):
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255) )
    for p_idx in range(len(discrete_lane_coors) - 1) :  # shape: N, 2
        pt1 = (int(discrete_lane_coors[p_idx][1]), int(discrete_lane_coors[p_idx][0]))
        pt2 = (int(discrete_lane_coors[p_idx + 1][1]), int(discrete_lane_coors[p_idx + 1][0]))
        cv2.line(raw_image, pt1, pt2, color=color, thickness=2)
    return raw_image


def get_semantic_lane_on_raw_image_coordinates(discrete_lane_coors, lane_semantic, raw_image):
    if lane_semantic == 1:
        color = (255, 0, 0)
    elif lane_semantic == 2:
        color = (0, 0, 255)
    else:
        color = (255, 255, 255)
    for p_idx in range(len(discrete_lane_coors) - 1) :  # shape: N, 2
        if abs(discrete_lane_coors[p_idx][0] - discrete_lane_coors[p_idx+1][0]) > 40:  # more than one semantics on the line
            continue
        pt1 = (int(discrete_lane_coors[p_idx][1]), int(discrete_lane_coors[p_idx][0]))
        pt2 = (int(discrete_lane_coors[p_idx + 1][1]), int(discrete_lane_coors[p_idx + 1][0]))
        cv2.line(raw_image, pt1, pt2, color=color, thickness=2)
    return raw_image

def get_endp_on_raw_image(discrete_endp_coors, raw_image):
    # for p_idx in range(len(discrete_endp_coors[0]) - 1):
    color = (0, 0, 250)
    for idx in range(discrete_endp_coors.shape[1]):
        pt1 = (int(discrete_endp_coors[1][idx]), int(discrete_endp_coors[0][idx]))
        cv2.circle(raw_image, center=pt1, radius=7, color=color)
    return raw_image

def get_gt_endp_on_raw_image(gt_endp_coors, raw_image):
    for lane_id in range(len(gt_endp_coors)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # print("gt coordinate: ", gt_endp_coors[lane_id])
        pt1 = (int(gt_endp_coors[lane_id][0][1]), int(gt_endp_coors[lane_id][0][0]))
        pt2 = (int(gt_endp_coors[lane_id][1][1]), int(gt_endp_coors[lane_id][1][0]))
        cv2.circle(raw_image, center=pt1, radius=5, color=color, thickness=cv2.FILLED)
        cv2.circle(raw_image, center=pt2, radius=5, color=color, thickness=cv2.FILLED)
    return raw_image
########################################################################
def get_metrics_conf_cls(list_metric):
    metric_conf = np.array(list_metric[:2])
    metric_cls = np.array(list_metric[2:])
    return np.max(metric_conf), np.max(metric_cls)


# RGB颜色转换为HSL颜色
def rgb2hsl(rgb):
    rgb_normal = [[[rgb[0] / 255, rgb[1] / 255, rgb[2] / 255]]]
    hls = cv2.cvtColor(np.array(rgb_normal, dtype=np.float32), cv2.COLOR_RGB2HLS)
    return hls[0][0][0], hls[0][0][2], hls[0][0][1]  # hls to hsl


# HSL颜色转换为RGB颜色
def hsl2rgb(hsl):
    hls = [[[hsl[0], hsl[2], hsl[1]]]]  # hsl to hls
    rgb_normal = cv2.cvtColor(np.array(hls, dtype=np.float32), cv2.COLOR_HLS2RGB)
    return int(rgb_normal[0][0][0] * 255), int(rgb_normal[0][0][1] * 255), int(rgb_normal[0][0][2] * 255)


# HSL渐变色
def get_multi_colors_by_hsl(begin_color, end_color, color_count):
    if color_count < 2:
        return []

    colors = []
    hsl1 = rgb2hsl(begin_color)
    hsl2 = rgb2hsl(end_color)
    steps = [(hsl2[i] - hsl1[i]) / (color_count - 1) for i in range(3)]
    for color_index in range(color_count):
        hsl = [hsl1[i] + steps[i] * color_index for i in range(3)]
        colors.append(hsl2rgb(hsl))

    return colors
