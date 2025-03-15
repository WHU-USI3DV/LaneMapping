import os
import cv2
import numpy as np
import tqdm
import json
import math
from multiprocessing import Pool
from functools import partial
import time
import random
import heapq


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load_seq(seq_path):
    r'''
    Load the dense sequence of the current image. It may contain the vertices of multiple boundary instances.
    '''
    with open(seq_path) as json_file:
        load_json = json.load(json_file)
        data_json = load_json

    seq_lens = []
    init_points = []
    end_points = []
    seq_semantic = []
    seq_instance = []

    for area in data_json:
        # print("area: ", area)
        seq_lens.append(len(area['seq']))
        init_points.append(area['init_vertex'])
        end_points.append(area['end_vertex'])
        seq_semantic.append(area['semantic'])
        seq_instance.append(area['instance'])

    seq = np.zeros((len(seq_lens), max(seq_lens), 2))
    for idx, area in enumerate(data_json):
        seq[idx, :seq_lens[idx]] = [x[0:2] for x in area['seq']]
    # seq = torch.FloatTensor(seq)
    # print("init points before downsampe: ", init_points)
    return seq, seq_lens, seq_semantic, seq_instance, init_points, end_points

def save_seq(seqs, seq_lens, seqs_semantic, seqs_instance, seqs_orient, seqs_filename):
    lines_labeled = []
    for id, seq_len in enumerate(seq_lens):
        line = {}
        line['semantic'] = seqs_semantic[id]
        line['instance'] = seqs_instance[id]
        line['seq_len'] = seq_len
        line['seq'] = seqs[id, :seq_len,:]
        line['init_vertex'] = seqs[id, 0, :]
        line['end_vertex'] = seqs[id,seq_len-1,:]
        line['seq_orient'] = seqs_orient[id, :seq_len]
        lines_labeled.append(line)

    with open(seqs_filename, "w") as f:
        json.dump(lines_labeled, f, cls=NpEncoder)
        print("Finish save sequence in ", seqs_filename)
        f.close()

def cal_seq_orientation(seqs, seq_lens):
    n_line, n_verx, _ = seqs.shape
    new_seqs_orient = np.zeros((n_line, n_verx), dtype=int)
    for id, line in enumerate(seqs):
        for v_id in range(seq_lens[id] - 1):
            pt0 = line[v_id]    # [row, col]
            pt1 = line[v_id+1]  # [row, col]
            vec = pt1 - pt0
            vec /= np.sqrt(vec[0]**2 + vec[1]**2) # [row, col] ~ [y, x]
            if np.abs(vec[1]) < 0.25:
                new_seqs_orient[id, v_id] = 5
            if 0.25 <= vec[1] < 0.6:
                new_seqs_orient[id, v_id] = 6
            if 0.6 <= vec[1] < 0.78:
                new_seqs_orient[id, v_id] = 7
            if 0.78 <= vec[1] < 0.86:
                new_seqs_orient[id, v_id] = 8
            if 0.86 <= vec[1] < 0.92:
                new_seqs_orient[id, v_id] = 9
            if vec[1] > 0.92:
                new_seqs_orient[id, v_id] = 10
            if -0.6 <= vec[1] < -0.25:
                new_seqs_orient[id, v_id] = 4
            if -0.78 <= vec[1] < -0.6:
                new_seqs_orient[id, v_id] = 3
            if -0.86 <= vec[1] < -0.78:
                new_seqs_orient[id, v_id] = 2
            if -0.92 <= vec[1] < -0.86:
                new_seqs_orient[id, v_id] = 1
            if vec[1] < -0.92:
                new_seqs_orient[id, v_id] = 0
    return new_seqs_orient

def sort_select_seq(seqs, seq_lens, seqs_semantic, seqs_instance, topK=4, col_range=(0, 1152), row_range=None):
    n_seq, n_verx, _ = seqs.shape
    # calculate mid point
    mid_pt = np.zeros((len(seq_lens), 2))
    for id, seq in enumerate(seqs):
        mid_pt[id] = (seq[0] + seq[seq_lens[id] - 1]) * 0.5

    # select
    if col_range is not None:
        valid = [col_range[0] <= pt[1] <= col_range[1] for pt in mid_pt]
        valid=np.where(valid)[0]
        seqs = seqs[valid, :, :]
        seqs_semantic = [seqs_semantic[idx] for idx in valid]
        seqs_instance = [seqs_instance[idx] for idx in valid]
        seq_lens = [seq_lens[idx] for idx in valid]

    if row_range is not None:
        valid = np.where(row_range[0] <= mid_pt[:, 0] <= row_range[1])
        valid = np.where(valid)[0]
        seqs = seqs[valid, :, :]
        seqs_semantic = [seqs_semantic[idx] for idx in valid]
        seqs_instance = [seqs_instance[idx] for idx in valid]
        seq_lens = [seq_lens[idx] for idx in valid]

    # filter the short ones
    if True:
        valid = []
        for id, seq in enumerate(seqs):
            if np.abs(seq[0,0] - seq[seq_lens[id]-1, 0]) > 10:
                valid.append(id)
        seqs = seqs[valid, :, :]
        seqs_semantic = [seqs_semantic[idx] for idx in valid]
        seqs_instance = [seqs_instance[idx] for idx in valid]
        seq_lens = [seq_lens[idx] for idx in valid]


    # select TopK
    if len(seqs) > topK:
        seqs_ins = np.array(seqs_instance)
        seqs_ins_idx = np.argsort(seqs_ins)
        valid_id = seqs_ins_idx[:topK]
        seqs=seqs[valid_id, :, :]
        seqs_semantic=[seqs_semantic[idx] for idx in valid_id]
        seqs_instance=[seqs_instance[idx] for idx in valid_id]
        seq_lens=[seq_lens[idx] for idx in valid_id]

    # check the seq increase from lower row to higher row
    for id, seq in enumerate(seqs):
        if seq[0, 0] > seq[seq_lens[id] - 1, 0]:
            seqs[id, :, :] = seq[::-1, :]
            seqs[id, :seq_lens[id], :] = seqs[id,(n_verx - seq_lens[id]):, :]
            seqs[id, seq_lens[id]:, : ] = 0


    # resort instance ID from left to right
    start_pt_col = seqs[:, 0, 1]
    terminal_pt_col = np.array([seqs[id,l-1,1]  for id, l in enumerate(seq_lens)])

    new_instance_id = np.lexsort((terminal_pt_col, start_pt_col))
    new_seqs_semantic = [seqs_semantic[id] for id in new_instance_id]
    new_seqs = seqs[new_instance_id]
    new_seq_lens = [seq_lens[id] for id in new_instance_id]
    new_seqs_instance = list(range(1, len(new_instance_id) + 1))

    # calculate orientation
    new_seqs_orient = cal_seq_orientation(new_seqs, new_seq_lens)
    # n_line, n_verx, _ = seqs.shape
    # new_seqs_orient = np.zeros((n_line, n_verx), dtype=int)
    # for id, line in enumerate(new_seqs):
    #     for v_id in range(new_seq_lens[id] - 1):
    #         pt0 = line[v_id]    # [row, col]
    #         pt1 = line[v_id+1]  # [row, col]
    #         vec = pt1 - pt0
    #         vec /= np.sqrt(vec[0]**2 + vec[1]**2) # [row, col] ~ [y, x]
    #         if np.abs(vec[1]) < 0.25:
    #             new_seqs_orient[id, v_id] = 5
    #         if 0.25 <= vec[1] < 0.6:
    #             new_seqs_orient[id, v_id] = 6
    #         if 0.6 <= vec[1] < 0.78:
    #             new_seqs_orient[id, v_id] = 7
    #         if 0.78 <= vec[1] < 0.86:
    #             new_seqs_orient[id, v_id] = 8
    #         if 0.86 <= vec[1] < 0.92:
    #             new_seqs_orient[id, v_id] = 9
    #         if vec[1] > 0.92:
    #             new_seqs_orient[id, v_id] = 10
    #         if -0.6 <= vec[1] < -0.25:
    #             new_seqs_orient[id, v_id] = 4
    #         if -0.78 <= vec[1] < -0.6:
    #             new_seqs_orient[id, v_id] = 3
    #         if -0.86 <= vec[1] < -0.78:
    #             new_seqs_orient[id, v_id] = 2
    #         if -0.92 <= vec[1] < -0.86:
    #             new_seqs_orient[id, v_id] = 1
    #         if vec[1] < -0.92:
    #             new_seqs_orient[id, v_id] = 0

    #         # angle_inv = np.arccos(vec[1])
    #         # new_seqs_orient[id, v_id] = int(18 * angle_inv / np.pi)

    return new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient

def sort_seq_anchor(seqs, seq_lens, seqs_semantic, seqs_instance, seqs_orient, col_range=(384, 768),
                    num_range=3, num_lane_per_range=3):
    # 1. calculate the subrange
    per_range = (col_range[1] - col_range[0]) / num_range
    sub_col_ranges = []
    start_range = col_range[0]
    for idx_range in range(num_range):
        this_col_range = (start_range, start_range+per_range)
        sub_col_ranges.append(this_col_range)
        start_range += per_range

    # 2. regenerate the instance id and corresponding infos
    n_seq, n_verx, _ = seqs.shape
    # calculate mid point
    mid_pt = np.zeros((len(seq_lens), 2))
    for id, seq in enumerate(seqs):
        mid_pt[id] = (seq[0] + seq[seq_lens[id] - 1]) * 0.5

    new_seqs = np.zeros((num_range*num_lane_per_range, n_verx, 2))
    new_seq_lens = np.zeros((num_range*num_lane_per_range), dtype=int)
    new_seqs_semantic = np.zeros((num_range*num_lane_per_range), dtype=int)
    new_seqs_instance = np.arange(1, num_range*num_lane_per_range+1)
    new_seqs_orient = np.zeros((num_range*num_lane_per_range, n_verx))
    for idx_range, sub_range in enumerate(sub_col_ranges):
        tmp_local_inst_id = 1

        for idx_inst, seq_len in enumerate(seq_lens):
            if (mid_pt[idx_inst, 1] > sub_range[0]) and (mid_pt[idx_inst, 1] < sub_range[1]) and \
                    (tmp_local_inst_id < (num_lane_per_range+1)):
                this_instance_id = idx_range * num_lane_per_range + tmp_local_inst_id
                new_seqs[this_instance_id-1, :, :] = seqs[idx_inst, :, :]
                new_seq_lens[this_instance_id-1] = seq_len
                new_seqs_semantic[this_instance_id-1] = seqs_semantic[idx_inst]
                new_seqs_instance[this_instance_id-1] = this_instance_id
                new_seqs_orient[this_instance_id-1, :] = seqs_orient[idx_inst, :]
                tmp_local_inst_id += 1


    # 3. return
    return new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient

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
    endpoint_maps = np.zeros((n_cls, img_h, img_w))
    endpoint_offs = np.zeros((n_cls, 2, img_h, img_w), dtype=np.float32)

    # for endpoint
    kernel_size = 4
    clip_width = kernel_size * 5
    if with_gaussian_kernel:
        for idx_cls in range(n_cls):
            if (np.abs(lb_endpoints[idx_cls][0] - lb_initpoints[idx_cls][0]) < EPS) and (
                    np.abs(lb_endpoints[idx_cls][1] - lb_initpoints[idx_cls][1]) < EPS):
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
            endpoint_maps[idx_cls, :, :] = two_maps
    # make sure the endpoint is eaqual to 1
    for idx_cls in range(n_cls):
        if (np.abs(lb_endpoints[idx_cls][0] - lb_initpoints[idx_cls][0]) < EPS) and (
                np.abs(lb_endpoints[idx_cls][1] - lb_initpoints[idx_cls][1]) < EPS):
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
        endpoint_maps = np.flip(np.flip(endpoint_maps, 1), 2)
        endpoint_offs = np.flip(np.flip(np.flip(endpoint_offs, 0), 1), 2)
    if merge_endp_map:
        endpoint_maps = np.amax(endpoint_maps, axis=0, keepdims=False)  # .astype(torch.float32)
        # endpoint_maps[endpoint_maps>0.1] = 1
        # endpoint_maps[endpoint_maps <= 0.1] = 0
    return endpoint_maps, endpoint_offs

def write_instance_orientation_seq(new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient,
                                   seqs_filename, semantic_filename, instance_filename, orient_filename, endp_filename):
    n_line, n_vertex, n_coor = new_seqs.shape
    label_semantic = np.zeros((1152, 1152), dtype=np.uint8)
    label_instance = np.zeros((1152, 1152), dtype=np.uint8)
    label_orient = np.zeros((1152, 1152), dtype=np.uint8)

    seq_start_vertexes = np.zeros((n_line, 2))
    seq_terminal_vertexes = np.zeros((n_line, 2))
    for id, seq_len in enumerate(new_seq_lens):
        tmp_semantic = new_seqs_semantic[id]
        if(new_seqs_semantic[id]==1):
            tmp_semantic = 128
        else:
            tmp_semantic = 255

        tmp_instance = int(new_seqs_instance[id])
        seq_start_vertexes[id, :] = new_seqs[id, 0, :]
        seq_terminal_vertexes[id, :] = new_seqs[id,seq_len-1,:]

        for id_vert in range(seq_len-1):
            pt0 = new_seqs[id, id_vert, :]
            pt1 = new_seqs[id, id_vert+1, :]
            pt0 = pt0[::-1]
            pt1 = pt1[::-1]
            pt0 = tuple(map(int, pt0))
            pt1 = tuple(map(int, pt1))

            # write semantic
            # print("pt 0: ", pt0)
            # print("pt 1: ", pt1)
            cv2.line(label_semantic, pt0, pt1, tmp_semantic)
            # write instance
            cv2.line(label_instance, pt0, pt1, tmp_instance)
            # write orientation
            tmp_orient = int(new_seqs_orient[id, id_vert])
            # print("temp orientation id: ", tmp_orient)
            cv2.line(label_orient, pt0, pt1, tmp_orient)
    # get endpoints map
    label_endp_map = np.zeros((1152, 1152))
    if n_line > 0:  # to avoid a NULL matrix in endpoints map generation
        label_endp_map, _ = get_endpoint_maps_per_batch(seq_start_vertexes, seq_terminal_vertexes, n_cls=n_line, img_h=1152, \
                                                 img_w=1152, is_flip=False, merge_endp_map=True)
    label_endp_map *= 255 # to write and show
    # write labeled images and seqs:
    img_quality = [cv2.IMWRITE_PNG_COMPRESSION, 100]
    cv2.imwrite(semantic_filename, label_semantic, img_quality)
    cv2.imwrite(instance_filename, label_instance, img_quality)
    cv2.imwrite(orient_filename, label_orient, img_quality)
    cv2.imwrite(endp_filename, label_endp_map, img_quality)
    save_seq(new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient, seqs_filename)

def process_single_file(old_seq_filename, new_seq_dir, new_semantic_dir, new_instance_dir, new_orient_dir, new_endp_dir,
                        topK=20, col_range=(100, 1000), row_range=None):
    print("old filename", old_seq_filename)
    with open(old_seq_filename) as json_file:
        load_json = json.load(json_file)
        if load_json==None:
            return
    parent_path, filesem = os.path.split(old_seq_filename)
    filetem, _ = os.path.splitext(filesem)
    new_seqs_filename = os.path.join(new_seq_dir, filetem + ".json")
    new_semantic_filename = os.path.join(new_semantic_dir, filetem + ".png")
    new_instance_filename = os.path.join(new_instance_dir, filetem + ".png")
    new_orient_filename = os.path.join(new_orient_dir, filetem + ".png")
    new_endp_filename = os.path.join(new_endp_dir, filetem + ".png")

    seqs, seq_lens, seqs_semantic, seqs_instance, _, _ = load_seq(old_seq_filename)
    new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient = sort_select_seq(
        seqs, seq_lens, seqs_semantic, seqs_instance, topK=topK, col_range=col_range, row_range=row_range)
    # generate lane annotation according to different lane range
    # anchor based
    # new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient = sort_seq_anchor(
    #     new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient,
    #     col_range=col_range, num_range=3, num_lane_per_range=3)
    write_instance_orientation_seq(new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient,
                                   new_seqs_filename, new_semantic_filename, new_instance_filename, new_orient_filename,
                                   new_endp_filename)

def test_one_file(old_seq_filename):
    parent_path, filesem = os.path.split(old_seq_filename)
    filetem, _ = os.path.splitext(filesem)
    new_seqs_filename = os.path.join(parent_path, filetem + "new_seq.json")
    new_semantic_filename = os.path.join(parent_path, filetem + "new_semantic.png")
    new_instance_filename = os.path.join(parent_path, filetem + "new_instance.png")
    new_orient_filename = os.path.join(parent_path, filetem + "new_orient.png")
    new_endp_filename = os.path.join(parent_path, filetem + "new_endp.png")

    with open(old_seq_filename) as json_file:
        load_json = json.load(json_file)
        if load_json==None:
            return
    seqs, seq_lens, seqs_semantic, seqs_instance, _, _ = load_seq(old_seq_filename)
    new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient = sort_select_seq(
        seqs, seq_lens, seqs_semantic, seqs_instance, topK=15, col_range=(384, 768), row_range=None)
    # generate lane annotation according to different lane range
    new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient = sort_seq_anchor(
        new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient,
        col_range=(384, 768), num_range=3, num_lane_per_range=3)
    write_instance_orientation_seq(new_seqs, new_seq_lens, new_seqs_semantic, new_seqs_instance, new_seqs_orient,
                                   new_seqs_filename, new_semantic_filename, new_instance_filename, new_orient_filename,
                                   new_endp_filename)


def multiprocessing_seqs_files(old_seq_filenames, new_seq_dir, new_semantic_dir, new_instance_dir, new_orient_dir, new_endp_dir):
    # for old_file in old_seq_filenames:
    #     process_single_file(old_file, new_seq_dir=new_seq_dir,
    #                                           new_semantic_dir=new_semantic_dir, new_instance_dir=new_instance_dir,
    #                                           new_orient_dir=new_orient_dir, new_endp_dir=new_endp_dir)

    num_process = 12
    with Pool(processes=num_process) as p:
        max_iter = len(old_seq_filenames)
        with tqdm.tqdm(total=max_iter) as pbar:
            for _ in p.imap_unordered(partial(process_single_file, new_seq_dir=new_seq_dir,
                                              new_semantic_dir=new_semantic_dir, new_instance_dir=new_instance_dir,
                                              new_orient_dir=new_orient_dir, new_endp_dir=new_endp_dir), old_seq_filenames, chunksize=num_process):
                pbar.update()



if __name__=='__main__':
    # old_seq_filename = "/home/mxx/Desktop/test_seq_oper/181013_0190.json"
    # test_one_file(old_seq_filename)

    # multiprocessing
    # old_seq_dir = "./LaserLane/All/labels/annotation_seq"  #"/workspace/All/labels/annotation_seq"
    # old_seq_dir = "/data/mxx/data/LaserLane/Test-Area-2/labels/annotation_seq"   # for evaluation data-1
    # old_seq_dir = "./data/LaserLane/All/labels_inside_lidar_range/annotation_seq"   # for inside_lidar_range
    old_seq_dir = "./data/LaserLane/Test-Area-WuhanDonghugaoxin/labels/annotation_seq"   # for test Wuhan data
    
    parent_dir, _ = os.path.split(old_seq_dir)
    os.chmod(parent_dir, mode=0o777)   # give the permission
    new_seq_dir = os.path.join(parent_dir, "sparse_seq")
    new_semantic_dir = os.path.join(parent_dir, "sparse_semantic")
    new_instance_dir = os.path.join(parent_dir, "sparse_instance")
    new_orient_dir = os.path.join(parent_dir, "sparse_orient")
    new_endp_dir = os.path.join(parent_dir, "sparse_endp")

    if not os.path.exists(new_seq_dir):
        os.makedirs(new_seq_dir)
    if not os.path.exists(new_semantic_dir):
        os.makedirs(new_semantic_dir)
    if not os.path.exists(new_instance_dir):
        os.makedirs(new_instance_dir)
    if not os.path.exists(new_orient_dir):
        os.makedirs(new_orient_dir)
    if not os.path.exists(new_endp_dir):
        os.makedirs(new_endp_dir)

    all_old_seqfiles = []
    for root, dirs, files in os.walk(old_seq_dir):
        for filepath in files:
            abs_filepath = os.path.join(root, filepath)
            if os.stat(abs_filepath).st_size == 0:
                print("empty filepath: ", abs_filepath)
                continue
            all_old_seqfiles.append(abs_filepath)

    multiprocessing_seqs_files(all_old_seqfiles, new_seq_dir, new_semantic_dir, new_instance_dir, new_orient_dir, new_endp_dir)



