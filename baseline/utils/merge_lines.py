'''
* Copyright (c) Xiaoxin Mi (mixiaoxin@whu.edu.cn). All rights reserved.
* author: Xiaoxin Mi
* e-mail: mixiaoxin@whu.edu.cn, mixiaoxin.goodluck@gmail.com
'''
import os.path

import numpy as np
from PIL import Image
import tqdm
from multiprocessing import Pool
from functools import partial
from io_utils import load_lane_seq, load_pc_2_img_transform_paras, save_seqs_json, save_seqs_list

EPS = 1e-6

def calculate_neatest_dist_id(pt, seq, proj=False):
    delta_x = seq[:, 0] - pt[0]
    delta_y = seq[:, 1] - pt[1]
    delta_xy_square = np.square(delta_x) + np.square(delta_y)
    min_xy_id = np.argmin(delta_xy_square)
    min_xy = np.sqrt(delta_xy_square[min_xy_id])

    if proj:
        seq_orint = calculate_principal_easy(seq)
        local_origin = seq[min_xy_id, :]
        local_origin_pt_vec = pt - local_origin
        perpend_vec = np.cross(seq_orint, local_origin_pt_vec)
        min_xy = np.sqrt(perpend_vec[0]*perpend_vec[0] + perpend_vec[1]*perpend_vec[1] + perpend_vec[2]*perpend_vec[2])

    return min_xy, min_xy_id


# brief:
# seq_query is the base sequence, seq_key is the sequence to be ckecked.
def calculate_average_distance_in_overlap_area(seq_query, seq_key):
    num_ver, _ = seq_key.shape
    aver_dist = 0.
    max_dist = 0.
    for id_v in range(num_ver):
        min_dist, _ = calculate_neatest_dist_id(seq_key[id_v], seq_query)
        aver_dist += min_dist
        if min_dist > max_dist:
            max_dist = min_dist
    return aver_dist / num_ver, max_dist


def calculate_principal_strict(seq):
    seq = np.array(seq)
    assert len(seq) >= 2
    center_pt = np.mean(seq, axis=0)
    seq_tranlate = seq - center_pt
    seq_tranlate[:,2] = 0   # set the z-value as zero
    seq_mat = np.dot(seq_tranlate.transpose(), seq_tranlate)
    e_vals, e_vecs = np.linalg.eig(seq_mat)
    sorted_index = np.argsort(e_vals)
    return e_vecs[:, sorted_index[-1]]

def calculate_principal_easy(seq):
    orients = seq[-1, :] - seq[0, :]
    orients[2] = 0
    orients_norm = np.sqrt(np.sum(np.square(orients)))
    orients_normalized = orients / (orients_norm + EPS)

    return orients_normalized

def merge_2_seqs(seq_base, seq_new):
    # calculate the principal direction of the base sequence
    prin_dir = calculate_principal_strict(seq_base)

    # keep the right direction
    easy_prin_dir = calculate_principal_easy(seq_base)
    if np.dot(prin_dir, easy_prin_dir) < 0:
        prin_dir *= -1

    # calculate the projection distance on the principal direction of two seqs
    base_proj_dist = [seq.dot(prin_dir) for seq in seq_base]
    new_proj_dist = [seq.dot(prin_dir) for seq in seq_new]

    # merge to the one, and label the merged vertex index, return the merged sequence and the starting vertex
    base_dist_max = base_proj_dist[-1]
    new_dist_min = new_proj_dist[0]
    base_overlap = np.where(base_proj_dist > new_dist_min)
    new_overlap = np.where(new_proj_dist < base_dist_max)

    # for the overlapping vertexes
    for new_overlap_idx in new_overlap[0]:
        # for base_overlap_idx in range(max(0, base_overlap[0][0]-1), seq_base.shape[0] - 1):
        for base_overlap_idx in base_overlap[0]:
            if new_proj_dist[new_overlap_idx] < base_proj_dist[base_overlap_idx ]:
                seq_base = np.insert(seq_base, base_overlap_idx, seq_new[new_overlap_idx], axis=0)
                base_proj_dist = np.insert(base_proj_dist, base_overlap_idx, new_proj_dist[new_overlap_idx])
                break

    if len(new_overlap[0]) < 1:
        base_overlap = [[seq_base.shape[0]]]
        seq_base = np.insert(seq_base, seq_base.shape[0], seq_new, axis=0)
    else:
        seq_base = np.insert(seq_base, seq_base.shape[0], seq_new[(new_overlap[0][-1] + 1):, ], axis=0)
    return seq_base, base_overlap[0][0]

def merge_2_reversed_seqs(seq_base, seq_new):
    # calculate the principal direction of the base sequence
    prin_dir = calculate_principal_strict(seq_base)

    # keep the right direction
    easy_prin_dir = calculate_principal_easy(seq_base)
    if np.dot(prin_dir, easy_prin_dir) < 0:
        prin_dir *= -1

    # calculate the projection distance on the principal direction of two seqs
    base_proj_dist = [seq.dot(prin_dir) for seq in seq_base]
    new_proj_dist = [seq.dot(prin_dir) for seq in seq_new]

    # merge to the one, and label the merged vertex index, return the merged sequence and the starting vertex
    base_dist_max = base_proj_dist[-1]
    base_dist_min = base_proj_dist[0]
    new_dist_min = new_proj_dist[0]
    new_forward = np.where(new_proj_dist > base_dist_max)
    new_backward = np.where(new_proj_dist < base_dist_min)

    for forward_idx in range(len(new_forward[0])):
        temp_idx = new_forward[0][-1-forward_idx]
        seq_base = np.insert(seq_base, len(seq_base), seq_new[temp_idx], axis=0)
        base_proj_dist = np.insert(base_proj_dist, len(base_proj_dist), new_proj_dist[temp_idx])

    for backward_idx in range(len(new_backward[0])):
        temp_idx = new_backward[0][backward_idx]
        seq_base = np.insert(seq_base, 0, seq_new[temp_idx], axis=0)
        base_proj_dist = np.insert(base_proj_dist, 0, new_proj_dist[temp_idx])

    return seq_base
def downsample_seqs(seq_base, dist_min=0.6):
    # downsample points in the seq by the distance:
    seq_base_next = seq_base.copy()
    seq_base_next[:-1,:] = seq_base_next[1:,:]
    seq_base_dist = seq_base_next - seq_base
    seq_base_dist[:, 2] = 0
    seq_base_dist = np.sqrt(np.sum(np.square(seq_base_dist), axis=1))

    accumute_dist = 0.
    downsampled_seq = np.array([seq_base[0, :]])
    for idx, idx_dist in enumerate(seq_base_dist):
        accumute_dist += idx_dist
        if accumute_dist > dist_min:
            downsampled_seq = np.concatenate((downsampled_seq, [seq_base[idx, :]]), axis=0)
            accumute_dist = 0.
        elif idx == (len(seq_base)-1):  # add the last vertex
            if seq_base_dist[idx-1] < 0.05 or idx == 0 or accumute_dist < 0.05:
                continue
            else:
                downsampled_seq = np.concatenate((downsampled_seq, [seq_base[idx, :]]), axis=0)
    return downsampled_seq

def fit_spline(seq_base):
    pass

def cal_local_orient(seq):
    # The last 5 vertexes are used to calculate the local orientation.
    # When the length of the sequence is shorter than 5, all vertexes are taken into considerations.
    if len(seq) > 5:
        orient = calculate_principal_easy(seq[-5:,:])
    else:
        orient = calculate_principal_easy(seq)
    return orient
def merge_lines(seq_filenames):
    check_merge_end = True
    # 1. sort filenames in increasing order
    sorted_seq_filenames = sorted(seq_filenames)

    # 2. read sequences and merge
    # take the sequences in the first file as baselines
    merged_seqs = []
    active_seqs_np, active_seq_lens, active_init_points, active_end_points = load_lane_seq(sorted_seq_filenames[0], dim_coor=3)
    active_seqs = [seq[:active_seq_lens[id]] for id, seq in enumerate(active_seqs_np)]
    active_seqs_roi_id = [0] * len(active_seqs)   # [0, 0, 0, 0, 0, 0, 0]
    active_seqs_orient = np.zeros((len(active_seqs), 3))
    for l_id in range(len(active_seqs)):
        active_seqs_orient[l_id, :] = cal_local_orient(active_seqs[l_id])
    print("active seq roi id: ", active_seqs_roi_id)

    for idx_block in range(1, len(sorted_seq_filenames)):
        temp_seqs_np, temp_seq_lens, temp_init_points, temp_end_points = load_lane_seq(sorted_seq_filenames[idx_block], dim_coor=3)
        temp_seqs = [seq[:temp_seq_lens[id]] for id, seq in enumerate(temp_seqs_np)]
        flag_active = [0] * (len(active_seqs))
        temp_seqs_orient = np.zeros((len(temp_seqs), 3))
        for l_id in range(len(temp_seqs)):
            temp_seqs_orient[l_id, :] = cal_local_orient(temp_seqs[l_id])

        for idx_tmp, tmp_s in enumerate(temp_seqs):
            # calculate the nearest distance from the temp lane start vertex to the active seqs
            temp_min_active_id = -1
            temp_min_active_vertex = -1
            temp_min_xy = 10
            temp_cos_angle = 1.0
            for idx_act, act_s in enumerate(active_seqs):
                # distance from the points to the active seqs
                # print("active_s: ", act_s.shape, int(active_seqs_roi_id[idx_act]))
                # for every temporary line, calculate the nearest active sequence
                a_xy, a_vertex_id = calculate_neatest_dist_id(temp_init_points[idx_tmp], act_s[int(active_seqs_roi_id[idx_act]):,:], proj=True)

                if a_xy < temp_min_xy:
                    temp_min_active_id = idx_act
                    temp_min_xy = a_xy
                    temp_min_active_vertex = a_vertex_id + active_seqs_roi_id[idx_act]

            if temp_min_xy < 0.5: # merge current seqs to the nearest active seqs
                # calculate the angle between two sequences:
                temp_act_cos_angle = temp_seqs_orient[idx_tmp].dot(active_seqs_orient[temp_min_active_id])
                b_xy, _ = calculate_neatest_dist_id(active_seqs[temp_min_active_id][-1, :], tmp_s, proj=True)
                print("active orient: ", active_seqs_orient[temp_min_active_id])
                print("cos angle: ", temp_act_cos_angle)

                # case 1:
                if check_merge_end and (b_xy < 0.5) and (temp_act_cos_angle > 0.7):  # exact orientation merge
                    # then calculate the endpoints of the merging part
                    print("MERGE1")
                    current_merged_seq, cuurent_merged_pt = merge_2_seqs(active_seqs[temp_min_active_id][active_seqs_roi_id[temp_min_active_id]:,:], tmp_s)
                    active_seqs[temp_min_active_id] = np.concatenate((active_seqs[temp_min_active_id][:active_seqs_roi_id[temp_min_active_id]],
                                                                      current_merged_seq), axis=0)
                    active_seqs_roi_id[temp_min_active_id] += cuurent_merged_pt
                    active_seq_lens[temp_min_active_id] = len(active_seqs[temp_min_active_id])
                    active_end_points[temp_min_active_id] = active_seqs[temp_min_active_id][-1, :]
                    flag_active[temp_min_active_id] = 1
                    # renew the orient
                    tmp_new_orient = cal_local_orient(active_seqs[temp_min_active_id])
                    active_seqs_orient[temp_min_active_id, :] = tmp_new_orient
                # case 2:
                elif check_merge_end and (b_xy < 0.5) and (temp_act_cos_angle < -0.7):  # reverse orientation merge
                    # calculate the overlapping part
                    print("MERGE2")
                    active_seqs[temp_min_active_id] = merge_2_reversed_seqs(active_seqs[temp_min_active_id], tmp_s)
                    # active_seqs_roi_id[temp_min_active_id] # keep the same as the last roi start pt
                    active_seq_lens[temp_min_active_id] = len(active_seqs[temp_min_active_id])
                    active_end_points[temp_min_active_id] = active_seqs[temp_min_active_id][-1, :]
                    active_init_points[temp_min_active_id] = active_seqs[temp_min_active_id][0, :]
                    flag_active[temp_min_active_id] = 1
                    # renew the orient
                    tmp_new_orient = cal_local_orient(active_seqs[temp_min_active_id])
                    active_seqs_orient[temp_min_active_id, :] = tmp_new_orient

                elif (not check_merge_end):
                    print("MERGE3")
                    # print("before appending ", active_seqs[temp_min_active_id].shape)
                    active_seqs[temp_min_active_id]=np.concatenate((active_seqs[temp_min_active_id], tmp_s), axis=0)
                    active_seqs_roi_id[temp_min_active_id] = len(active_seqs[temp_min_active_id]) - active_seq_lens[temp_min_active_id]
                    active_seq_lens[temp_min_active_id] += temp_seq_lens[idx_tmp]
                    active_end_points[temp_min_active_id] = temp_end_points[idx_tmp]
                    flag_active[temp_min_active_id] = 1
                    active_seqs_orient[temp_min_active_id, :] = temp_seqs_orient[idx_tmp, :]
                    # print("after appending ", active_seqs[temp_min_active_id].shape)
                else:
                    print("MERGE4")
                    active_seqs.append(tmp_s)
                    active_seq_lens.append(temp_seq_lens[idx_tmp])
                    active_init_points.append(temp_init_points[idx_tmp])
                    active_end_points.append(temp_end_points[idx_tmp])
                    active_seqs_roi_id.append(0)
                    active_seqs_orient = np.append(active_seqs_orient, [temp_seqs_orient[idx_tmp, :]], axis=0)
                    flag_active.append(1)
            else:
                print("MERGE5")
                active_seqs.append(tmp_s)
                active_seq_lens.append(temp_seq_lens[idx_tmp])
                active_init_points.append(temp_init_points[idx_tmp])
                active_end_points.append(temp_end_points[idx_tmp])
                active_seqs_roi_id.append(0)
                active_seqs_orient = np.append(active_seqs_orient, [temp_seqs_orient[idx_tmp, :]], axis=0)
                flag_active.append(1)

        for id, flag in enumerate(flag_active):
            if flag < 0.5: # is not active
                if len(active_seqs[id]) < 3:
                    active_seqs.pop(id)
                else:
                    merged_seqs.append(active_seqs.pop(id))
                active_seq_lens.pop(id)
                active_init_points.pop(id)
                active_end_points.pop(id)
                active_seqs_roi_id.pop(id)
                flag_active.pop(id)
                active_seqs_orient = np.delete(active_seqs_orient, id, axis=0)

    for id, seq in enumerate(active_seqs):
        if len(seq) < 3:
            continue
        else:
            merged_seqs.append(seq)

    # 3. return merged sequences
    return merged_seqs

if __name__=="__main__":
    # seqfiles_dir = "/home/mxx/Desktop/test_merge_seqs"  # for test
    seqfiles_dir = "./logs/vis/out_pc_seq_json_dir"
    all_seq_json_path = []
    for root, dirs, files in os.walk(seqfiles_dir):
        for filepath in files:
            seqfile_path = os.path.join(root, filepath)
            if os.stat(seqfile_path).st_size == 0:
                print("empty filepath: ", seqfile_path)
                continue
            (filepath_stem, filepath_ext) = os.path.splitext(filepath)
            if filepath_ext != '.json':
                continue
            all_seq_json_path.append(seqfile_path)

    merged_seqs = merge_lines(all_seq_json_path)

    # downsample the sequences
    downsampled_merged_seqs = [downsample_seqs(seq) for seq in merged_seqs]

    filename_merged_seq = os.path.join(seqfiles_dir, "merged.txt")
    filename_downsample_seq = os.path.join(seqfiles_dir, "merged_downsample.txt")
    save_seqs_list(merged_seqs, filename_merged_seq)
    save_seqs_list(downsampled_merged_seqs, filename_downsample_seq)