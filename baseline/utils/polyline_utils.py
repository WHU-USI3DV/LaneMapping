# @ Author: Xiaoxin Mi
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors


def Hausdorf_distance(line1, line2):
    dists = np.abs(line1 - line2)
    dists[np.where(line1 < 0)] = -1
    dists[np.where(line2 < 0)] = -1  # calculate the minimum distance for each overlapping vertex
    max_dist = np.max(dists)
    if max_dist < 0.:
        mean_dist = -1.
        min_dist = -1.
    else:
        dist_valid = dists[np.where(dists >= 0)[0]]
        mean_dist = np.mean(dist_valid)
        min_dist = np.min(dist_valid)
    return min_dist, max_dist, mean_dist


def lines_align(line1, line2):
    '''
    Param_in: line-1: column values for line1
    param_in: line-2: column calues for line2
    
    return: line-1 , line-2 (line-1 is always at the left of line-2)
    '''
    dists = np.abs(line1 - line2)
    dists[np.where(line1 < 0)] = -1
    dists[np.where(line2 < 0)] = -1  # calculate the minimum distance for each overlapping vertex
    over_lap_rows = np.where(dists >= 0.00001)[0]
    # print('overlap_rows: ', over_lap_rows)
    for row_id in over_lap_rows:
        if line2[row_id] < line1[row_id]:
            tmp_value = line1[row_id]
            line1[row_id] = line2[row_id]
            line2[row_id] = tmp_value
        if abs(line1[row_id] - line2[row_id]) < 2.0:
            if abs(line1[row_id] - line1[row_id-1]) < abs(line2[row_id] - line2[row_id-1]) and line1[row_id-1] > 0 and line2[row_id-1]>0:
                line2[row_id] = -1  # refine this vertex
            else:
                line1[row_id] = -1
        
    return line1, line2

def line_slop(lines):
    slops = np.zeros(len(lines))
    for id in range(len(lines)):
        valid_vs = np.where(lines[id, :] > 0.)[0]
        if len(valid_vs>1):
            delta_h = (valid_vs[-1] - valid_vs[0])
            delta_w = lines[id, valid_vs[-1]] - lines[id, valid_vs[0]]
            slops[id] = delta_w / delta_h
    return slops

def polyline_NMS2(arr_cls_vertex, semantic_map):
    '''

    Args:
        arr_cls_vertex_in: predicted polylines
        semantic_map: predicted semantic map

        if the mean overlapping distance < 20 and maximum overlapping distance < 50, then we need to check the vertex location,
        to make sure that two polylines are not cross each other.

    Returns:
        polylines after Non-Maximum Suppression
    '''
    mean_dist_thre = 10
    active_id = 0
    num_lane, num_h = arr_cls_vertex.shape
    # arr_slops = line_slop(arr_cls_vertex)
    opt_point_2_point = True  # This performs better
    opt_line_align = True

    while active_id < num_lane - 1:  # for every lane
        if len(np.where(arr_cls_vertex[active_id, :] > 0)[0]) < 2:
            # arr_cls_vertex[active_id, :] = -1.
            active_id += 1
            continue
        for idx_l in range(active_id + 1, num_lane):  # check the Hausdorf distance between lines
            if len(np.where(arr_cls_vertex[idx_l, :] > 0)[0]) < 2:
                continue
            else:
                min_dist, _, mean_dist = Hausdorf_distance(arr_cls_vertex[active_id, :], arr_cls_vertex[idx_l, :])
                last_active_v = None
                last_check_v = None
                if (min_dist >= 0.) and (min_dist < mean_dist_thre):  # two close lines, then merge them
                    if opt_line_align:
                        arr_cls_vertex[active_id, :], arr_cls_vertex[idx_l, :] = lines_align(arr_cls_vertex[active_id, :], arr_cls_vertex[idx_l, :])
                    if opt_point_2_point:
                        for idx_h in range(num_h):
                            if arr_cls_vertex[active_id, idx_h] < 0 and arr_cls_vertex[idx_l, idx_h] < 0:  # two lines have no vertex at this location
                                continue
                            elif arr_cls_vertex[active_id, idx_h] > 0 and arr_cls_vertex[idx_l, idx_h] < 0:  # line 1 has vertex; line 2 doesn't
                                continue
                            elif arr_cls_vertex[active_id, idx_h] < 0 and arr_cls_vertex[idx_l, idx_h] > 0:  # line 1 has no vertex; line 2 has vertex here
                                if last_active_v is None:
                                    arr_cls_vertex[active_id, idx_h] = arr_cls_vertex[idx_l, idx_h]
                                    arr_cls_vertex[idx_l, idx_h] = -1.
                                    last_active_v = arr_cls_vertex[active_id, idx_h]
                                else:
                                    if abs(last_active_v - arr_cls_vertex[idx_l, idx_h]) < mean_dist_thre:
                                        arr_cls_vertex[active_id, idx_h] = arr_cls_vertex[idx_l, idx_h]
                                        arr_cls_vertex[idx_l, idx_h] = -1.
                                        last_active_v = arr_cls_vertex[active_id, idx_h]
                                    else:
                                        last_check_v = arr_cls_vertex[idx_l, idx_h]
                            elif arr_cls_vertex[active_id, idx_h] > 0 and arr_cls_vertex[idx_l, idx_h] > 0:  # line 1 and line 2 both have vertex on this row
                                if np.abs( arr_cls_vertex[idx_l, idx_h] - arr_cls_vertex[active_id, idx_h]) < mean_dist_thre:
                                    if semantic_map[idx_h * 8 + 3, int(arr_cls_vertex[active_id, idx_h])] > semantic_map[
                                        idx_h * 8 + 3, int(arr_cls_vertex[idx_l, idx_h])]:
                                        high_conf_v = arr_cls_vertex[active_id, idx_h]
                                    else:
                                        high_conf_v = arr_cls_vertex[idx_l, idx_h]
                                    # then:
                                    if (last_active_v is None) and (last_check_v is None):
                                        arr_cls_vertex[active_id, idx_h] = high_conf_v
                                        arr_cls_vertex[idx_l, idx_h] = -1.
                                        last_active_v = arr_cls_vertex[active_id, idx_h]
                                    elif (active_id is not None) and abs(last_active_v - high_conf_v) < mean_dist_thre:
                                        arr_cls_vertex[active_id, idx_h] = high_conf_v
                                        arr_cls_vertex[idx_l, idx_h] = -1.
                                        last_active_v = arr_cls_vertex[active_id, idx_h]
                                    else:
                                        arr_cls_vertex[active_id, idx_h] = -1.
                                        arr_cls_vertex[idx_l, idx_h] = high_conf_v
                                        last_check_v = arr_cls_vertex[idx_l, idx_h]
                                else:
                                    if (last_active_v is None) and (last_check_v is None):
                                        if arr_cls_vertex[active_id, idx_h] > arr_cls_vertex[idx_l, idx_h]:  # keep active_id is on the left
                                            ttt = arr_cls_vertex[idx_l, idx_h]
                                            arr_cls_vertex[idx_l, idx_h] = arr_cls_vertex[active_id, idx_h]
                                            arr_cls_vertex[active_id, idx_h] = ttt
                                            last_check_v = arr_cls_vertex[idx_l, idx_h]
                                            last_active_v = arr_cls_vertex[active_id, idx_h]
                
        active_id += 1
    arr_cls_vertex = interpolate_plyline(arr_cls_vertex)

    # remove close polyline
    active_id = 0
    while active_id < num_lane - 1:  # for every lane
        active_v_num = len(np.where(arr_cls_vertex[active_id, :] > 0)[0])
        if active_v_num < 2:
            arr_cls_vertex[active_id, :] = -1.
            active_id += 1
            continue

        for idx_l in range(active_id + 1, num_lane):  # check the Hausdorf distance between lines
            idx_v_num = len(np.where(arr_cls_vertex[idx_l, :] > 0)[0])
            if idx_v_num < 2:
                arr_cls_vertex[idx_l, :] = -1.
                continue
            else:
                min_dist, max_dist, mean_dist = Hausdorf_distance(arr_cls_vertex[active_id, :], arr_cls_vertex[idx_l, :])
                if (max_dist >= 0.) and (max_dist < mean_dist_thre*1.5 or mean_dist < mean_dist_thre*0.8):  # two close lines, delete the shorter one:
                    if active_v_num < idx_v_num:
                        arr_cls_vertex[active_id, :] = -1.
                    else:
                        arr_cls_vertex[idx_l, :] = -1.
        active_id += 1
    return arr_cls_vertex


def sort_lines_from_left_to_right(lane_vertex):
    num_l, num_v = lane_vertex.shape
    first_v_valus = np.zeros(num_l) + 1152
    # sort by the first vertexes' column coordinate
    for l_id in range(num_l):
        first_v = np.where(lane_vertex[l_id, :] >= 0)
        if len(first_v[0]) > 0:
            first_v_valus[l_id] = lane_vertex[l_id, first_v[0][0]]
    sort_id = np.argsort(first_v_valus)
    lane_vertex_sorted = lane_vertex[sort_id, :]

    return lane_vertex_sorted

def interpolate_plyline(lane_vertex):
    num_l, num_v = lane_vertex.shape
    for idx_line in range(num_l):
        ph_idx = np.where(lane_vertex[idx_line, :] > 0.0001)
        if len(ph_idx[0]) > 1:
            start_id = ph_idx[0][0]
            end_id = ph_idx[0][-1]
            current_positive_id = -1
            for v_id in range(start_id, end_id):
                if lane_vertex[idx_line, v_id] < 0.0001:
                    tmp_ratio = (1.0 * v_id - ph_idx[0][current_positive_id]) / (
                                ph_idx[0][current_positive_id + 1] - ph_idx[0][current_positive_id])
                    lane_vertex[idx_line, v_id] = (1 - tmp_ratio) * lane_vertex[
                        idx_line, ph_idx[0][current_positive_id]] + (
                                                  lane_vertex[idx_line, ph_idx[0][current_positive_id + 1]]) * tmp_ratio
                    # lane_vertex[idx_line, v_id] = (lane_vertex[idx_line, ph_idx[0][current_positive_id]] + lane_vertex[idx_line, ph_idx[0][current_positive_id + 1]]) * 0.5
                else:
                    current_positive_id += 1
    return lane_vertex

def occupancy_filter(occu_flag, occu_seg_conf, half_k_size=4):
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

def smooth_cls_line_per_batch(out_cls, out_orient, complete_inner_nodes=False, out_seg_conf=None):
    # traverse every lane from the first lane:
    buff_width = 6
    buff_depth = 24
    line_num, vertex_num = out_cls.shape
    # sort lines:
    smooth_cls = sort_lines_from_left_to_right(out_cls)
      
    # smooth_cls_total = smooth_cls
    smooth_cls_total = np.zeros_like(out_cls) - 1
    exist_lane_length = np.zeros(line_num)
    flag_in0 = np.zeros((vertex_num, 1152))  # detected points location
    for idx_line in range(line_num):
        ph_idx = np.where(out_cls[idx_line, :] > 0)
        flag_in0[ph_idx[0], (out_cls[idx_line, ph_idx[0]]).astype(int)] = 1
    
    if out_seg_conf is not None:
        flag_in = occupancy_filter(flag_in0, out_seg_conf[3:1152:8, :], half_k_size=4)
    else:
        flag_in = flag_in0

    # print("flag_in sumL2 ", np.sum(flag_in))
    # print("diff: ", np.where(flag_in != flag_in0))
    while flag_in.sum() > 2 and exist_lane_length.min() < 2:  # free vertex & empty output lane id
        temp_smooth_cls = np.zeros_like(out_cls) - 1  # initial vertex
        tmp_lane_length = np.zeros(line_num)  # length of each lane
        for idx_line in range(line_num):
            flag_start = False
            last_h = 0
            idx_h = 0  # vertex row id
            active_lane_id = idx_line

            last_c = 0
            delta_last_d = 0
            h_step = 1  # row step
            while idx_h < vertex_num:
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
                        last_c = current_col
                        active_lane_id = idx_line
                    idx_h += 1  # move to next row
                    h_step = 1
                else:  # vertex string
                    # next_pred_col = current_col + (current_dir - 5) * 4
                    next_pred_col = current_col  #+ (current_dir - 5) * 4
                    if tmp_lane_length[idx_line] > 1:
                        delta_last_d = (current_col - last_c) / h_step
                        next_pred_col = current_col + delta_last_d  
                    near_dist = 1152
                    near_id = line_num
                    near_h = idx_h
                    # width traverse for searching
                    for sub_idx_line in range(line_num):  # list the candidate, choose the closest one
                        if smooth_cls[sub_idx_line, idx_h] > 0 and flag_in[
                            idx_h, int(smooth_cls[sub_idx_line, idx_h])] > 0:
                            tmp_dist = np.abs(next_pred_col - smooth_cls[sub_idx_line, idx_h])
                            # delta_d = smooth_cls[sub_idx_line, idx_h] - current_col
                            # tmp_dist = np.abs(delta_d - delta_last_d)
                            if tmp_dist < near_dist:
                                near_dist = tmp_dist
                                near_id = sub_idx_line
                                near_h = idx_h
                    # depth traverse for searching
                    for next_h_idx in range(idx_h + 1, vertex_num):
                        if (next_h_idx - idx_h) > buff_depth:
                            break
                        if smooth_cls[active_lane_id, next_h_idx] > 0 and flag_in[
                            next_h_idx, int(smooth_cls[active_lane_id, next_h_idx])] > 0:
                            tmp_dist = np.abs(next_pred_col - smooth_cls[active_lane_id, next_h_idx])
                            # delta_d = smooth_cls[active_lane_id, next_h_idx] - current_col
                            # tmp_dist = np.abs(delta_d - delta_last_d)
                            if tmp_dist < near_dist:
                                near_dist = tmp_dist
                                near_id = active_lane_id
                                near_h = next_h_idx
                            break  # finish as soon as searched the first vertex

                    if near_dist < buff_width:  # succeed in finding next vertex
                        temp_smooth_cls[idx_line, near_h] = smooth_cls[near_id, near_h]
                        tmp_lane_length[idx_line] += 1
                        # renew the coordinates
                        last_c = current_col
                        current_col = smooth_cls[near_id, near_h]
                        current_dir = out_orient[near_h, int(current_col / 8)]
                        next_pred_col = current_col + (current_dir - 5) * 4
                        flag_in[near_h, int(smooth_cls[near_id, near_h])] = 0  # this vertex is occupied
                        h_step = near_h - last_h
                        last_h = near_h
                        idx_h = near_h + 1
                        active_lane_id = near_id
                    else:  # fail in finding the next vertex
                        temp_smooth_cls[idx_line, idx_h] = -1  # no vertex
                        # we find no next vertex, then stop extending this line.
                        # break
                        idx_h += 1
                        h_step += 1
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
                # tmp_endp_next_col = tmp_endp_idx_value + (tmp_endp_dir - 5) * 4
                tmp_endp_next_col = tmp_endp_idx_value + (tmp_endp_idx_value - temp_smooth_cls[idx_line, tmp_vertex_idx[0][-2]])

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
                        # endp_next_col = ednp_idx_value + (current_end_dir - 5) * 4
                        endp_next_col = ednp_idx_value + (ednp_idx_value - smooth_cls_total[sub_idx_line, vertex_idx[0][-2]])
                        #  attach to bottom || # attach to top
                        if (0 < (tmp_startp_idx_h - endp_idx_h) < buff_depth and np.abs(
                                endp_next_col - tmp_startp_idx_value) < buff_width) or \
                                (0 < (startp_idx_h - tmp_endp_idx_h) < buff_depth and np.abs(
                                    tmp_endp_next_col - startp_idx_value) < buff_width):
                            smooth_cls_total[sub_idx_line, tmp_vertex_idx[0]] = temp_smooth_cls[
                                idx_line, tmp_vertex_idx[0]]
                            exist_lane_length[sub_idx_line] += tmp_lane_length[idx_line]
                            attached = True
                            break

                if attached == False:  # start a new lane
                    for sub_idx_line in range(line_num):
                        if exist_lane_length[sub_idx_line] < 2:
                            smooth_cls_total[sub_idx_line, tmp_vertex_idx[0]] = temp_smooth_cls[
                                idx_line, tmp_vertex_idx[0]]
                            exist_lane_length[sub_idx_line] = tmp_lane_length[idx_line]
                            break

    # if complete the inner nodes
    
    if complete_inner_nodes:
        smooth_cls_total = interpolate_plyline(smooth_cls_total)
    
    # return the lines after smoothing
    smooth_cls_total = sort_lines_from_left_to_right(smooth_cls_total)
    
    # if two lines are close enough, then keep the two polylines parallel but not intersect
    # smooth_cls_total1 = modify_topology(smooth_cls_total)
    
    return smooth_cls_total

'''
uniform polyline's semantics on one image;
after modification, the vertexes semantics on one polyline are identical.
'''
def hat_window(sequence, ):
    pass

def polyline_uniform_semantics(ply_vertexes, endp_map, r_buff=12):
    '''
    Args:
        ply_vertexes: the vertexes of each polyline [N, M, 2],
                      N number of polylines,
                      M number of vertexes on a polyline,
                      2:  column coordinates and semantics for each vertex
        endp_map: [H, W], where there is an endpoint, the value there is 1; otherwise, the value is 0

        Brief: For each polyline, the function checks its vertexes' semantics sequentially.
        Confirm the semantic change only if there is a detected endpoint nearby.
        On the other hand, Confirm the endpoint only if there is a detected polyline nearby.
    Returns:
        ply_vertexes: where semantics of the vertexes on a polyline keep identical.

    '''
    n_line, n_v, n_att = ply_vertexes.shape
    (h_endp, w_endp) = np.where(endp_map > 0)
    v_endp = np.concatenate((h_endp.reshape((len(h_endp), 1)), w_endp.reshape((len(w_endp), 1))), axis=1)
    n_endp, _ = v_endp.shape
    knearest_endp = NearestNeighbors(algorithm='kd_tree').fit(v_endp)

    endp_checked = [False]*n_endp
    for line_id in range(n_line):
        polyline_v_id = np.where(ply_vertexes[line_id, :, 0] > 0.)[0]
        if(len(polyline_v_id) > 1):
            last_v_semantic = ply_vertexes[line_id, polyline_v_id[0], 1]
            for v_id in polyline_v_id[1:]:
                if last_v_semantic != ply_vertexes[line_id, v_id, 1]:
                    # ckeck the endpoint
                    current_v = [8*v_id + 3, ply_vertexes[line_id, v_id, 0]]
                    n_dist, n_idx = knearest_endp.kneighbors([current_v], n_neighbors=1)
                    post_idx = min(v_id + 10, polyline_v_id[-1], n_v)
                    mean_semantic_post = np.mean(ply_vertexes[line_id, v_id:post_idx, 1])

                    if (n_dist[0, 0] < r_buff) or np.abs(mean_semantic_post - ply_vertexes[line_id, v_id, 1]) < 0.001:  # which means semantics is supposed to be smooth
                        # confirm semantic change
                        last_v_semantic = ply_vertexes[line_id, v_id, 1]
                        endp_checked[n_idx[0, 0]] = True
                    else:
                        ply_vertexes[line_id, v_id, 1] = last_v_semantic
    # print("old endpoint number: ", n_endp)
    # print("old endp: ", np.sum(endp_map))
    # print("new endpoint number: ", len(np.where(np.array(endp_checked) == True)[0]))
    # check endp:
    if len(np.where(np.array(endp_checked) == True)[0]) < n_endp:
        f_p_endp_id = np.where(np.array(endp_checked) == False)[0]
        endp_map[v_endp[f_p_endp_id, 0], v_endp[f_p_endp_id, 1]] = 0
    # print("new endp: ", np.sum(endp_map))
    return ply_vertexes, endp_map


def polyline_uniform_semantics_by_statistics(ply_vertexes, endp_map=None, r_buff=12):
    '''
    Args:
        ply_vertexes: the vertexes of each polyline [N, M, 2],
                      N number of polylines,
                      M number of vertexes on a polyline,
                      2:  column coordinates and semantics for each vertex
        endp_map: [H, W], where there is an endpoint, the value there is 1; otherwise, the value is 0

        Brief: For each polyline, the function checks its vertexes' semantics sequentially.
        Confirm the semantic change only if there is a detected endpoint nearby.
        On the other hand, Confirm the endpoint only if there is a detected polyline nearby.
    Returns:
        ply_vertexes: where semantics of the vertexes on a polyline keep identical.

    '''
    max_void = r_buff
    n_line, n_v, n_att = ply_vertexes.shape
    if endp_map is not None:
        # print("endp_map: ", endp_map)
        (h_endp, w_endp) = np.where(endp_map > 0)
        v_endp = np.concatenate((h_endp.reshape((len(h_endp), 1)), w_endp.reshape((len(w_endp), 1))), axis=1)
        n_endp, _ = v_endp.shape
        endp_kdtree = NearestNeighbors(algorithm='kd_tree').fit(v_endp)
        # print("endp.shape: ", v_endp.shape)

    all_vertexes = None
    for line_id in range(n_line):
        polyline_v_id = np.where(ply_vertexes[line_id, :, 0] > 0.)[0]
        if(len(polyline_v_id) > 1):
            # get the vertex:
            line_vs = np.zeros((n_v, 2))
            line_vs[:, 0] = np.arange(3, 1152, 8)
            line_vs[:, 1] = ply_vertexes[line_id, :, 0]
            if all_vertexes is None:
                all_vertexes = line_vs[polyline_v_id, :]
            else:
                all_vertexes = np.append(all_vertexes, line_vs[polyline_v_id, :], axis=0)

            # statistics of semantics
            semantic_count = np.zeros((1, 2), dtype=int)
            semantic_count[0, 0] = ply_vertexes[line_id, 0, 1]  # the semantics of the first vertex
            semantic_count[0, 1] = 1   # the accumulating count of the current semantic
            current_semantics = ply_vertexes[line_id, 0, 1]
            for v_id in range(1, n_v):
                if current_semantics == ply_vertexes[line_id, v_id, 1]:
                    semantic_count[-1, 1] += 1
                else:
                    semantic_count = np.append(semantic_count, [[ply_vertexes[line_id, v_id, 1], 1]], axis=0)
                    current_semantics = ply_vertexes[line_id, v_id, 1]

            # smooth the semantics:
            void_size = 5
            while void_size < max_void:
                s_idx = 1
                while (s_idx < (len(semantic_count)-1)):
                    if (semantic_count[s_idx - 1, 0] > 0) and\
                        (semantic_count[s_idx - 1, 0] != semantic_count[s_idx, 0]) and \
                        (semantic_count[s_idx + 1, 0] == semantic_count[s_idx - 1, 0]) and \
                        (semantic_count[s_idx, 1] < void_size) and \
                        (semantic_count[s_idx - 1, 1] - semantic_count[s_idx, 1] >= 0) and \
                        (semantic_count[s_idx + 1, 1] - semantic_count[s_idx, 1] >= 0):
                        semantic_count[s_idx - 1, 1] += (semantic_count[s_idx, 1] + semantic_count[s_idx+1, 1])
                        semantic_count = np.delete(semantic_count, s_idx, axis=0)
                        semantic_count = np.delete(semantic_count, s_idx, axis=0) # now, s_idx is the original (s_idx+1)
                        s_idx = 1
                    else:
                        s_idx += 1
                void_size += 3

            # update semantics of vertexes:
            start_id = 0
            for s_idx in range(len(semantic_count)):
                end_id = int(start_id+semantic_count[s_idx, 1])
                ply_vertexes[line_id, start_id:end_id, 1] = semantic_count[s_idx, 0]
                start_id = end_id

            if endp_map is not None:
                ##########################################################
                # the relationship between lines and endpoints
                kdtree_line = NearestNeighbors(algorithm='kd_tree').fit(line_vs[polyline_v_id, :])
                # 1. if there is only one semantics on the line, no endpoint is supposed to be in the middle of line
                max_semantic = 0
                max_semantic_count = 0
                for s_id in range(len(semantic_count)):
                    if (semantic_count[s_id, 0] > 0) and (semantic_count[s_id, 1] > max_semantic_count):
                        max_semantic = semantic_count[s_id, 0]
                        max_semantic_count = semantic_count[s_id, 1]
                if max_semantic_count > 130:
                    nearest_dist, nearest_idx = kdtree_line.radius_neighbors(v_endp, radius=8)
                    for id in range(len(nearest_idx)): # corresponding to the endpoint index
                        if len(nearest_idx[id]) > 0:
                            endp_map[v_endp[id, 0], v_endp[id, 1]] = 0.

                # 2. if there are more than 1 semantics on the line, endpoints are supposed to be at the semantics changing location and terminal locations.

                # 3. if the terminal points on the middle of the image, there is supposed to be a endpoint here,
                # if (line_vs[polyline_v_id[0], 0] > 200) and (line_vs[polyline_v_id[0], 0] < 952) and (len(polyline_v_id) > 35):
                #     nearest_dist, nearest_idx = endp_kdtree.kneighbors(line_vs[polyline_v_id[0], :].reshape([1,2]), n_neighbors=1)
                #     if(nearest_dist < max_void):
                #         if line_vs[polyline_v_id[0], 0] < v_endp[nearest_idx[0], 0]:
                #             new_start_idx = int((v_endp[nearest_idx[0], 0] - 3) / 8.)
                #             ply_vertexes[line_id, 0:new_start_idx, 0] = -1.  # coordinates
                #             ply_vertexes[line_id, 0:new_start_idx, 1] = 0    # semantics
                #         else:
                #             new_start_idx = int((v_endp[nearest_idx[0], 0] - 3) / 8.)
                #             ply_vertexes[line_id, new_start_idx:polyline_v_id[0], 0] = ply_vertexes[line_id, polyline_v_id[0], 0]  # coordinates
                #             ply_vertexes[line_id, new_start_idx:polyline_v_id[0], 1] = ply_vertexes[line_id, polyline_v_id[0], 1]  # semantics
                #     else: # new endpoint
                #         endp_map[int(line_vs[polyline_v_id[0], 0]), int(line_vs[polyline_v_id[0], 1])] = 1.
                #
                # if (line_vs[polyline_v_id[-1], 0] > 200) and (line_vs[polyline_v_id[-1], 0] < 952) and (len(polyline_v_id) > 35):
                #     nearest_dist, nearest_idx = endp_kdtree.kneighbors(line_vs[polyline_v_id[-1], :].reshape(1,2), n_neighbors=1)
                #     if (nearest_dist < max_void):
                #         if line_vs[polyline_v_id[-1], 0] < v_endp[nearest_idx[0], 0]:
                #             new_end_idx = int((v_endp[nearest_idx[0], 0] - 3) / 8.)
                #             ply_vertexes[line_id, polyline_v_id[-1]:new_end_idx, 0] = ply_vertexes[
                #                 line_id, polyline_v_id[-1], 0]  # coordinates
                #             ply_vertexes[line_id, polyline_v_id[-1]:new_end_idx, 1] = ply_vertexes[
                #                 line_id, polyline_v_id[-1], 1]  # semantics
                #         else:
                #             new_end_idx = int((v_endp[nearest_idx[0], 0] - 3) / 8.)
                #             ply_vertexes[line_id, new_end_idx:polyline_v_id[-1], 0] = -1.  # coordinates
                #             ply_vertexes[line_id, new_end_idx:polyline_v_id[-1], 1] = 0  # semantics
                #     else:  # new endpoint
                #         endp_map[int(line_vs[polyline_v_id[-1], 0]), int(line_vs[polyline_v_id[-1], 1])] = 1.
                ######################################################### 
            


    if endp_map is not None:
        if all_vertexes is not None:
            knearest_all_vertexes = NearestNeighbors(algorithm='kd_tree').fit(all_vertexes)
            for endp_id in range(len(v_endp)):
                n_dist, n_idx = knearest_all_vertexes.kneighbors([v_endp[endp_id, :]], n_neighbors=1)
                if n_dist[0] > 10:
                    endp_map[v_endp[endp_id, 0], v_endp[endp_id, 1]] = 0
    # print("new endp: ", np.sum(endp_map))
    return ply_vertexes, endp_map


def remove_short_polyline(ply_vertexes, min_v_count=12):
    '''
    Args:
        ply_vertexes: the vertexes of each polyline [N, M, 2],
                      N number of polylines,
                      M number of vertexes on a polyline,
                      2:  column coordinates and semantics for each vertex
    Returns:
        ply_vertexes: where number of the vertexes on a polyline is more than @min_v_count.

    '''
    n_line, n_v, n_att = ply_vertexes.shape
    for line_id in range(n_line):
        polyline_v_id = np.where(ply_vertexes[line_id, :, 0] > 0.)[0]
        if(len(polyline_v_id) < min_v_count):
            # get the vertex:
            ply_vertexes[line_id, :, 0] = -1.
            ply_vertexes[line_id, :, 1] = 0.

    return ply_vertexes

def renew_semantic_map(ply_vertexes):
    '''
    Args:
        ply_vertexes: the vertexes of each polyline [N, M, 2],
                      N number of polylines,
                      M number of vertexes on a polyline,
                      2:  column coordinates and semantics for each vertex
    Returns:
        semantic_map: is generated from ply_vertexes.
    '''
    semantic_map = np.zeros((1152, 1152))
    num_lane, num_vertex, _ = ply_vertexes.shape
    for id_lane in range(num_lane):
            for p_idx in range(num_vertex - 1) :  # shape: N, 2
                pt1_y = int(ply_vertexes[id_lane][p_idx][0])
                pt2_y = int(ply_vertexes[id_lane][p_idx + 1][0])
                if (pt1_y < 0) or (pt2_y < 0):
                    continue
                else: 
                    pt1 = (pt1_y, int(p_idx*8 +3))
                    pt2 = (pt2_y, int((p_idx + 1) *8 +3))
                    if (int(ply_vertexes[id_lane][p_idx][1]) == 2) or (int(ply_vertexes[id_lane][p_idx+1][1]) == 2):
                        color = 2  # dashed line
                    else:
                        color = 1  # solid line

                    cv2.line(semantic_map, pt1, pt2, color=color, thickness=1)
                    
    return semantic_map
