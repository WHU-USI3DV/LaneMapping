'''
* Copyright (c) Xiaoxin Mi (mixiaoxin@whu.edu.cn). All rights reserved.
* author: Xiaoxin Mi
* e-mail: mixiaoxin@whu.edu.cn, mixiaoxin.goodluck@gmail.com
'''
import os.path
import cv2
import numpy as np
from PIL import Image
import tqdm
from multiprocessing import Pool
from functools import partial
from io_utils import load_lane_seq, load_pc_2_img_transform_paras, save_seqs_json, save_seqs_txt

EPS = 1e-6
'''
@ Brief: multiply two quanternions
quan_1 = [w1, x1, y1 , z1]
quan_2 = [w2, x2, y2 , z2]

'''
def multiplyQuanternion(quan_1, quan_2):
    assert len(quan_1) == len(quan_2) == 4
    quan_out = np.zeros(4)
    quan_out[0] = quan_1[0]*quan_2[0] - quan_1[1]*quan_2[1] - quan_1[2]*quan_2[2] - quan_1[3]*quan_2[3]
    quan_out[1] = quan_1[0]*quan_2[1] + quan_1[1]*quan_2[0] + quan_1[2]*quan_2[3] - quan_1[3]*quan_2[2]
    quan_out[2] = quan_1[0]*quan_2[2] - quan_1[1]*quan_2[3] + quan_1[2]*quan_2[0] + quan_1[3]*quan_2[1]
    quan_out[3] = quan_1[0]*quan_2[3] + quan_1[1]*quan_2[2] - quan_1[2]*quan_2[1] + quan_1[3]*quan_2[0]

    return quan_out

'''
@brief: roated_v = quan * v * inv(quan)
if quan = [w, x, y ,z]
||quan|| = sqrt(w^2 + x^2 + y^2 + z^2)
inv(quan) = [w, -x, -y, -z] / ||quan|| 
'''
def rotateByQuanternion3D(quan, vec_in):
    quan_norm = np.sqrt(np.sum(np.square(quan)))
    assert len(quan) == 4 and quan_norm > EPS
    q_vec_in = np.zeros(4)
    q_vec_in[1] = vec_in[0]
    q_vec_in[2] = vec_in[1]
    q_vec_in[3] = vec_in[2]

    quan_inv = quan.copy()
    quan_inv[1:4] *= -1.
    quan_inv /= quan_norm
    inter_q = multiplyQuanternion(quan, q_vec_in)
    vec_out = multiplyQuanternion(inter_q, quan_inv)
    vec_out = vec_out[1:]

    return vec_out

def translateCoordinates(vec_in, offset):
    vec_in += offset
    return vec_in

def LeastSuqare(X, Y):
    # Y = wX + b
    # calculate w
    N = len(Y)
    p = N*sum(X*Y) - sum(X)*sum(Y)
    q = N*sum(X*X) - sum(X)*sum(X)
    if abs(q) < EPS:
        w = 0.
    else:
        w = p/q

    # calculate b
    b = sum(Y - w*X)/N

    return w, b
def modify_empty_pixel_elevation(img, roi_pts=None, roi_pts_length=None):
    # check the empty pixels:
    img_h, img_w, img_c = img.shape
    if roi_pts is None:
        pts_loc = np.where(np.sum(img, axis=2) < 1)

        # for empty pixels, search the nearest non-empty pixel by dilation rectangle
        for pt_id in range(len(pts_loc[0])):
            pt_h = pts_loc[0][pt_id]
            pt_w = pts_loc[1][pt_id]
            step = 1
            roi_value = 0
            while roi_value < 1:
                min_h = max(pt_h-step, 0)
                max_h = min(pt_h+step, img_h)
                min_w = max(pt_w-step, 0)
                max_w = min(pt_w+step, img_w)
                roi_value = np.sum(img[min_h:max_h, min_w:max_w, :])
                if roi_value > 0:
                    valid_num = len(np.where(np.sum(img[min_h:max_h, min_w:max_w, :], axis=2) > 0)[0])
                    roi_ele_value = np.sum(img[min_h:max_h, min_w:max_w, 1]) / valid_num
                    img[pt_h, pt_w, 1] = roi_ele_value
                else:
                    step += 1
    else:
        # modify the interested pixels
        n_lane, n_pts, _ = roi_pts.shape
        for l_id in range(n_lane):
            this_line_length = roi_pts_length[l_id]
            for pt_id in range(this_line_length):
                pt_h = int(roi_pts[l_id, pt_id, 0])
                pt_w = int(roi_pts[l_id, pt_id, 1])
                if ((pt_h == 0.) and (pt_w == 0.)) or (np.sum(img[pt_h, pt_w, :]) > 1):
                    continue
                else:
                    step = 1
                    roi_value = 0
                    while roi_value < 1:
                        min_h = max(pt_h - step, 0)
                        max_h = min(pt_h + step, img_h)
                        min_w = max(pt_w - step, 0)
                        max_w = min(pt_w + step, img_w)
                        roi_value = np.sum(img[min_h:max_h, min_w:max_w, :])
                        if roi_value > 0:
                            valid_num = len(np.where(np.sum(img[min_h:max_h, min_w:max_w, :], axis=2) > 0)[0])
                            roi_ele_value = np.sum(img[min_h:max_h, min_w:max_w, 1]) / valid_num
                            img[pt_h, pt_w, 1] = roi_ele_value
                        else:
                            step += 1

    return img


def transform_coordinate_from_img_2_pc(params, img_seqs, img_seq_lens, bev_img):
    n_line, max_line_len, _ = img_seqs.shape
    # 1) image scale is related to image resolution
    # 2) image offset is related to point cloud offset
    seqs_3d = np.zeros((n_line, max_line_len, 3))
    ########################
    # This operation is not in demand. I do it here because I observe that reversely projected points in original points coordinates system are not at the same locations as they are on predicted bev images.
    # img_seqs += 4
    ########################
    seqs_3d[:, :, 0] = (img_seqs[:, :, 0]) * params['img_reso'][0]
    seqs_3d[:, :, 1] = (img_seqs[:, :, 1]) * params['img_reso'][1]
    seqs_3d[:, :, 0] = seqs_3d[:, :, 0] + params['bev_img_offset'][0]
    seqs_3d[:, :, 1] = seqs_3d[:, :, 1] + params['bev_img_offset'][1]
    # 3) G-channel is related to elevation
    bev_img = np.array(bev_img)
    # cv2.imshow('before elevation modify', bev_img)
    # refine the elevation values of BEV images: if it is empty, add the nearest average elevation
    # print("locations: ", np.where(np.sum(bev_img, axis=2) < 1))
    bev_img = modify_empty_pixel_elevation(bev_img, img_seqs, img_seq_lens)
    # cv2.imshow('after elevation modify', bev_img2)
    # print("locations2: ", np.where(np.sum(bev_img, axis=2) < 1))
    # cv2.waitKey(0)

    seqs_3d[:, :, 2] = bev_img[img_seqs[:, :, 0].astype(int), img_seqs[:, :, 1].astype(int), 1] * params['ele_reso'] + params['local_min_ele']
    # seqs_3d[:, :, 2] = bev_img[img_seqs[:, :, 0].astype(int), img_seqs[:, :, 1].astype(int), 1] * 0.05 + params['local_min_ele']  # real elevation_reso = 0.05

    # smooth elevation value here
    for l_id in range(n_line):
        ele_values = np.array(seqs_3d[l_id, 0:img_seq_lens[l_id], 2])
        ele_idxes = np.arange(img_seq_lens[l_id])
        w, b = LeastSuqare(ele_idxes, ele_values)
        new_ele_values = w*ele_idxes + b
        seqs_3d[l_id,0:img_seq_lens[l_id], 2] = new_ele_values


    # 4) reverse rotation from bev image to point cloud, then plus the translation
    quan_translate = params['las_rotation_trans_quan'][0:3]
    quan_translate = np.array(quan_translate)
    quan = params['las_rotation_trans_quan'][3:]
    quan = np.array(quan)
    for idx_line in range(n_line):
        for idx_v in range(max_line_len):
            # print("before rotation elevation: ", seqs_3d[idx_line, idx_v, :])
            seqs_3d[idx_line, idx_v, :] = rotateByQuanternion3D(quan, seqs_3d[idx_line, idx_v, :])
            # print("after rotation elevation: ", seqs_3d[idx_line, idx_v, :])
            seqs_3d[idx_line, idx_v, :] += quan_translate

    # 5) add initial point cloud offset
    las_read_offset = params['las_read_offset']
    las_read_offset = np.array(las_read_offset)
    seqs_3d += las_read_offset
    #####################
    # to show, add extra elevation offset
    # seqs_3d[:, :, 2] += 0.25
    ####################

    return seqs_3d

def transform_coordinate_from_img_2_pc_single(img_seqfile_path, bev_img_path, pc_img_params_path,
                                              pc_seqfile_path, pc_seqfile_txt_path):
    # print("filename: ", img_seqfile_path, bev_img_path, pc_img_params_path, pc_seqfile_path, pc_seqfile_txt_path)
    # 1. load necessary files
    img_seqs, img_seq_lens, img_seq_init_points, img_seq_end_points = load_lane_seq(img_seqfile_path)
    if len(img_seqs) < 1:
        return
    pc_img_params = load_pc_2_img_transform_paras(pc_img_params_path)
    bev_img = Image.open(bev_img_path)

    # 2. transform coordinates
    pc_seqs = transform_coordinate_from_img_2_pc(pc_img_params, img_seqs, img_seq_lens, bev_img)

    # 3. Modify coordinates:
    # 3.1 elevation smoothness

    # 3.2 xoy coordinates smoothness

    pc_seqs_valid = []
    n_line, _, _ = pc_seqs.shape
    for idx_line in range(n_line):
        sub_valid_seq = pc_seqs[idx_line, :img_seq_lens[idx_line], :]
        sub_line={}
        sub_line["seq"] = sub_valid_seq
        sub_line["seq_len"] = img_seq_lens[idx_line]
        sub_line["init_vertex"] = sub_valid_seq[0, :]
        sub_line["end_vertex"] = sub_valid_seq[img_seq_lens[idx_line]-1, :]
        pc_seqs_valid.append(sub_line)

    # 3. save 3d seq
    save_seqs_json(pc_seqs_valid, pc_seqfile_path)
    save_seqs_txt(pc_seqs_valid, pc_seqfile_txt_path)


def multiprocessing_seqs_files(img_seqfile_dir, bev_img_dir, pc_img_params_dir):
    parent_dir, _ = os.path.split(img_seqfile_dir)
    out_pc_seq_json_dir = os.path.join(parent_dir, "out_pc_seq_json_dir")
    out_pc_seq_txt_dir = os.path.join(parent_dir, "out_pc_seq_txt_dir")
    if not os.path.exists(out_pc_seq_json_dir):
        os.makedirs(out_pc_seq_json_dir)
    if not os.path.exists(out_pc_seq_txt_dir):
        os.makedirs(out_pc_seq_txt_dir)

    all_img_seqfile_path = []
    all_bev_img_path = []
    all_pc_img_params_path = []
    all_pc_seqfile_json_path = []
    all_pc_seqfile_txt_path = []
    for root, dirs, files in os.walk(img_seqfile_dir):
        for filepath in files:
            img_seqfile_path = os.path.join(root, filepath)
            if os.stat(img_seqfile_path).st_size == 0 :
                print("empty filepath: ", img_seqfile_path)
                continue
            (filepath_stem, filepath_ext) = os.path.splitext(filepath)
            if filepath_ext != '.json':
                continue

            bev_img_path = os.path.join(bev_img_dir, filepath_stem+'.png')
            pc_img_params_path = os.path.join(pc_img_params_dir, filepath_stem+'.txt')
            pc_seqfile_json_path = os.path.join(out_pc_seq_json_dir, filepath_stem+".json")
            pc_seqfile_txt_path = os.path.join(out_pc_seq_txt_dir, filepath_stem+".txt")

            all_img_seqfile_path.append(img_seqfile_path)
            all_bev_img_path.append(bev_img_path)
            all_pc_img_params_path.append(pc_img_params_path)
            all_pc_seqfile_json_path.append(pc_seqfile_json_path)
            all_pc_seqfile_txt_path.append(pc_seqfile_txt_path)

    all_argument_pairs = zip(all_img_seqfile_path, all_bev_img_path, all_pc_img_params_path, all_pc_seqfile_json_path,
                             all_pc_seqfile_txt_path)

    num_process = 12
    with Pool(processes=num_process) as p:
        max_iter = len(all_img_seqfile_path)
        with tqdm.tqdm(total=max_iter) as pbar:
            for _ in p.starmap(partial(transform_coordinate_from_img_2_pc_single,), all_argument_pairs,  chunksize=num_process):
                pbar.update()
    pass

def test_single_file():
    img_seqfile_path = "/home/mxx/Desktop/test-img2las/181013_0209.json"
    bev_img_path = "/home/mxx/Desktop/test-img2las/181013_0209.png"
    pc_img_params_path = "/home/mxx/Desktop/test-img2las/181013_0209.txt"
    parent_dir, _ = os.path.split(img_seqfile_path)
    pc_seqfile_path = os.path.join(parent_dir, "point_cloud_seq.json")
    pc_seqfile_txt_path = os.path.join(parent_dir, "point_cloud_seq.txt")

    transform_coordinate_from_img_2_pc_single(img_seqfile_path, bev_img_path, pc_img_params_path,
                                              pc_seqfile_path, pc_seqfile_txt_path)

if __name__ == "__main__":
    # multiprocessing
    img_seqfile_dir = '/home/mxx/mxxcode/LaneMapping/logs/vis/LaserLaneProposal'
    bev_img_dir = '/mnt/data/LaneMapping/LaserLane/Test-Area-Nanjing/cropped_tiff'
    pc_img_params_dir = '/mnt/data/LaneMapping/LaserLane/Test-Area-Nanjing/cropped_tiff_param'
    multiprocessing_seqs_files(img_seqfile_dir, bev_img_dir, pc_img_params_dir)

    # test single file
    # test_single_file()
