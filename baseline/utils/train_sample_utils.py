'''
@Author: Xiaoxin Mi
'''

import torch
import numpy as np
import math


__all__ = ['gaussian', 'get_endpoint_maps_per_batch', ]
def gaussian(x_id, y_id, H, W, sigma=5):
    ## Function that creates the heatmaps ##
    channel = [math.exp(-((x - x_id) ** 2 + (y - y_id) ** 2) / (2 * sigma ** 2)) for x in range(W) for y in
               range(H)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W))
    return channel

def get_endpoint_maps_per_batch(lb_initpoints, lb_endpoints, lb_initoffs=None, lb_endoffs=None,
                                n_cls=6, img_h=144, img_w=144, is_flip=False, merge_endp_map=False):
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
                heatmap1 = gaussian(int(lb_initpoints[idx_cls][0]), int(lb_initpoints[idx_cls][1]), img_h,
                                         img_w, sigma=kernel_size / 2)
            # for end point: Gaussian kernel roi
            if lb_endpoints[idx_cls][0] > clip_width and lb_endpoints[idx_cls][0] < (img_h - clip_width) and \
                    lb_endpoints[idx_cls][1] > clip_width and lb_endpoints[idx_cls][1] < (img_w - clip_width):
                heatmap2 = gaussian(int(lb_endpoints[idx_cls][0]), int(lb_endpoints[idx_cls][1]), img_h, img_w,
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
        endpoint_maps = torch.amax(endpoint_maps, dim=0, keepdim=True)  # .astype(torch.float32)
        # endpoint_maps[endpoint_maps>0.1] = 1
        # endpoint_maps[endpoint_maps <= 0.1] = 0
    return endpoint_maps, endpoint_offs




###############################################################
# 步骤二： 根据sequence生成多种数据以加速训练: instance image, mask image, orientation image
