'''
* author: Xiaoxin Mi
* e-mail: mixiaoxin@whu.edu.cn
'''
import numpy as np
import cv2
from skimage.morphology import skeletonize
import scipy
from PIL import Image

EPS = 1e-16

def calc_measures_buffer(arr_label, arr_pred, buff_radius = 2):
    arr_pred_img = upsample_pred(arr_pred)
    TP, FP, FN, TN = calc_measures_conf_buffer(arr_label, arr_pred_img, buff_radius=buff_radius)

    # print(TP, FP, FN, TN)

    accuracy = (TP+TN)/(TP+TN+FP+FN+EPS)
    precision = TP/(TP+FP+EPS)
    recall = TP/(TP+FN+EPS)
    f1 = (2*TP/(2*TP+FP+FN+EPS))

    return accuracy, precision, recall, f1, TP, FP, FN

def calc_measures(arr_label, arr_pred, mode = 'conf', is_wo_offset = False):
    if mode == 'conf':
        if is_wo_offset:
            TP, FP, FN, TN = calc_measures_conf_wo_offset(arr_label, arr_pred)
        else:
            TP, FP, FN, TN = calc_measures_conf(arr_label, arr_pred)
    else:
        if is_wo_offset:
            TP, FP, FN, TN = calc_measures_cls_wo_offset(arr_label, arr_pred)
        else:
            TP, FP, FN, TN = calc_measures_cls(arr_label, arr_pred)

    # print(TP, FP, FN, TN)

    accuracy = (TP+TN)/(TP+TN+FP+FN+EPS)
    precision = TP/(TP+FP+EPS)
    recall = TP/(TP+FN+EPS)
    f1 = (2*TP/(2*TP+FP+FN+EPS))

    return accuracy, precision, recall, f1, TP, FP, FN

def cal_coor_measures(arr_label, arr_pred, mode = 'conf', offset_thre = 8):
    if mode == 'conf':
        TP, num_seg_pts, DG, num_gt_pts = calc_coor_measures_conf_metric2(arr_label, arr_pred, buff_radius=offset_thre)
    else:  #
        TP, FP, FN = calc_coor_measures_cls(arr_label, arr_pred, offset_thre = offset_thre)

    # print(TP, FP, FN, TN)

    # accuracy = (TP+TN)/(TP+TN+FP+FN+EPS)
    # all_positive = len(np.where(arr_label > 0)[0])
    # precision = TP/(TP+FP+EPS)
    # recall = TP/(TP+FN+EPS)
    # f1 = (2*TP/(2*TP+FP+FN+EPS))
    acc = TP / (num_seg_pts + EPS)
    recall = DG / (num_gt_pts + EPS)
    f1 = 2.0 * acc * recall / (acc + recall + EPS)

    return acc, recall, f1, TP, num_seg_pts, DG, num_gt_pts

def calc_coor_measures_conf_metric1(arr_label, arr_pred, offset_thre = 8):
    TP = 0
    FP = 0
    FN = 0
    TP2 = 0
    n_cls, n_anchor = arr_label.shape
    Pos_Pred = np.zeros((n_cls, n_anchor))
    Pos_Pred[arr_pred > 0] = 1
    Pos_count = len(np.where(arr_pred>0)[0])

    for pred_idx_cls in range(n_cls):
        tmp_dist = np.zeros((n_cls, n_anchor))
        for gt_idx_cls in range(n_cls):
            tmp_dist[gt_idx_cls, :] = np.abs(arr_pred[pred_idx_cls, :] - arr_label[gt_idx_cls,:])
        tmp_dist_min_id = np.argmin(tmp_dist, axis=0)
        tmp_dist_min_val = np.amin(tmp_dist, axis=0)
        tmp_pos_count = len(np.where(arr_pred[pred_idx_cls]>0)[0])

        TP_locations = (arr_pred[pred_idx_cls, :] > 0) & (tmp_dist_min_val < offset_thre) & \
                       (arr_label[tmp_dist_min_id, list(np.arange(0, n_anchor, 1))] > 0)
        tmp_TP_count = len(np.where(TP_locations)[0])
        TP += tmp_TP_count
        FP_locations = (arr_pred[pred_idx_cls, :] > 0) & ((tmp_dist_min_val >= offset_thre) | \
                                                          (arr_label[tmp_dist_min_id, list(np.arange(0, n_anchor, 1))] < 0))
        tmp_FP_count = len(np.where( FP_locations)[0])
        FP += len(np.where( FP_locations)[0])
        # print("TP is: {}, FP is {}, Positive is {} ".format(tmp_TP_count, tmp_FP_count, tmp_pos_count))
    for gt_idx_cls in range(n_cls):
        tmp_dist = np.zeros((n_cls, n_anchor))
        for pred_idx_cls in range(n_cls):
            tmp_dist[pred_idx_cls, :] = np.abs(arr_pred[pred_idx_cls, :] - arr_label[gt_idx_cls,:])
        tmp_dist_min_id = np.argmin(tmp_dist, axis=0)
        tmp_dist_min_val = np.amin(tmp_dist, axis=0)

        TP_locations2 = (arr_label[gt_idx_cls, :] > 0) & (tmp_dist_min_val < offset_thre) & \
                    (arr_pred[tmp_dist_min_id, list(np.arange(0, n_anchor, 1))] > 0 )
        tmp_TP_count2 = len(np.where(TP_locations2)[0])
        TP2 += tmp_TP_count2

        FN += len(np.where((arr_label[pred_idx_cls, :] > 0) & ((tmp_dist_min_val >= offset_thre) | \
                                                                 (arr_pred[tmp_dist_min_id, list(np.arange(0, n_anchor, 1))] < 0)) )[0])

    # print("TP1 is {}, TP2 is {}".format(TP, TP2))

    return TP, FP, FN

def calc_coor_measures_conf_metric2(arr_label, arr_pred, buff_radius=2):
    '''
    * in : arr_label (list(np.array), float, 9*(144)) (0, 1152), -1
    * in : arr_pred (list(np.array), float, 9*(144))
    * out: accuracy, precision, recall, f1
    '''
    # assert len(arr_label) == len(arr_pred)
    num_lane = len(arr_label)
    num_pred = len(arr_pred)
    H = len(arr_label[0])
    W = 1152

    all_positive = len(np.where(arr_label>0)[0])

    OCCUPIED = 1.
    NOT_OCCUPIED = 0.

    TP = 0
    FP = 0
    TP_2 = 0
    FN = 0
    # prediction to gt
    for lane_id in range(0, num_pred):
        for row_id in range(0, H):
            pred_enhanced = 0
            label_enhanced = 0
            if arr_pred[lane_id, row_id] > 0:
                col_min = max(0, arr_pred[lane_id, row_id] - buff_radius)
                col_max = min(W-1, arr_pred[lane_id, row_id] + buff_radius)
                for gt_lane_id in range(0, num_lane):
                    if arr_label[gt_lane_id, row_id] > col_min and arr_label[gt_lane_id, row_id] < col_max:
                        pred_enhanced = OCCUPIED
                        break
                if pred_enhanced == OCCUPIED:
                    TP += 1
                else:
                    FP += 1
    # gt to prediction
    for lane_id in range(0, num_lane):
        for row_id in range(0, H):
            label_enhanced = 0
            if arr_label[lane_id, row_id] > 0:
                col_min = max(0, arr_label[lane_id, row_id] - buff_radius)
                col_max = min(W - 1, arr_label[lane_id, row_id] + buff_radius)
                for pred_lane_id in range(0, num_pred):
                    if arr_pred[pred_lane_id, row_id] > col_min and arr_pred[pred_lane_id, row_id] < col_max:
                        label_enhanced = OCCUPIED
                        break
                if label_enhanced == OCCUPIED:
                    TP_2 += 1
                else:
                    FN += 1
    TN = all_positive - TP - FP - FN

    return TP, (TP + FP), TP_2, (TP_2 + FN)

def calc_coor_measures_cls(arr_label, arr_pred, offset_thre = 8):
    dist = np.abs(arr_label - arr_pred)
    TP = len(np.where((dist < offset_thre) & (arr_label > 0) & (arr_pred > 0))[0])
    FP = len(np.where(((dist >= offset_thre) | (arr_label < 0)) & (arr_pred > 0))[0])
    FN = len(np.where(((dist >= offset_thre) | (arr_pred < 0)) & (arr_label > 0))[0])
    positive_count = len(np.where(arr_pred > 0)[0])

    # print("TP={}, FP={}, FN={}, Positive_count={}".format(TP, FP, FN, positive_count))
    return TP, FP, FN

def calc_measures_conf(arr_label, arr_pred):
    '''
    * in : arr_label (np.array, float, 144x144)
    * in : arr_pred (np.array, float, 144x144)
    * out: accuracy, precision, recall, f1
    '''

    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    OCCUPIED = 1.
    NOT_OCCUPIED = 0.

    # F1 = TP/(TP+0.5*(FP+FN))
    TP = 0
    FP = 0
    FN = 0
    for row in range(1,143):
        for col in range(1,143):
            label = 0
            pred = 0
            pred_enhanced = 0
            label_enhanced = 0
            
            if ((temp_pred[row, col] == temp_label[row,col]) or
                (temp_pred[row, col+1] == temp_label[row,col]) or
                (temp_pred[row, col-1] == temp_label[row,col]) or
                (temp_pred[row-1, col+1] == temp_label[row,col]) or
                (temp_pred[row-1, col-1] == temp_label[row,col]) or
                (temp_pred[row+1, col-1] == temp_label[row,col]) or
                (temp_pred[row+1, col+1] == temp_label[row,col]) or
                (temp_pred[row-1, col] == temp_label[row,col]) or
                (temp_pred[row+1, col] == temp_label[row,col])
            ):
                pred_enhanced = OCCUPIED
            
            if ((temp_label[row, col] == temp_pred[row,col]) or
                (temp_label[row, col+1] == temp_pred[row,col]) or
                (temp_label[row, col-1] == temp_pred[row,col]) or
                (temp_label[row-1, col-1] == temp_pred[row,col]) or
                (temp_label[row-1, col+1] == temp_pred[row,col]) or
                (temp_label[row+1, col-1] == temp_pred[row,col]) or
                (temp_label[row+1, col+1] == temp_pred[row,col]) or
                (temp_label[row-1, col] == temp_pred[row,col]) or
                (temp_label[row+1, col] == temp_pred[row,col])
            ):
                label_enhanced = OCCUPIED
            
            label = temp_label[row,col]
            pred = temp_pred[row,col]
            
            if (label == OCCUPIED):
                if(pred_enhanced == OCCUPIED):
                    TP += 1
                else:
                    FN += 1

            if (pred == OCCUPIED):
                if(label_enhanced == NOT_OCCUPIED):
                    FP += 1

    TN = 144*144 - TP - FP - FN
    
    return TP, FP, FN, TN

def calc_measures_conf_wo_offset(arr_label, arr_pred):
    '''
    * in : arr_label (np.array, float, 144x144)
    * in : arr_pred (np.array, float, 144x144)
    * out: f1-score
    '''

    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    OCCUPIED = 1.
    NOT_OCCUPIED = 0.

    # F1 = TP/(TP+0.5*(FP+FN))
    TP = 0
    FP = 0
    FN = 0
    for row in range(144):
        for col in range(144):
            label = 0
            pred = 0
            pred_enhanced = 0
            label_enhanced = 0
            
            if (temp_pred[row, col] == temp_label[row,col]):
                pred_enhanced = OCCUPIED
            
            if (temp_label[row, col] == temp_pred[row,col]):
                label_enhanced = OCCUPIED
            
            label = temp_label[row,col]
            pred = temp_pred[row,col]
            
            if (label == OCCUPIED):
                if(pred_enhanced == OCCUPIED):
                    TP += 1
                else:
                    FN += 1

            if (pred == OCCUPIED):
                if(label_enhanced == NOT_OCCUPIED):
                    FP += 1
    
    TN = 144*144 - TP - FP - FN

    return TP, FP, FN, TN


def upsample_pred(pred_lane_coors):
    n_lane = len(pred_lane_coors)
    raw_img = np.zeros((1152, 1152))
    for lane_id in range(n_lane):
        for p_idx in range(len(pred_lane_coors[lane_id]) - 1):  # shape: N, 2
            pt1 = (int(pred_lane_coors[p_idx][1]), int(pred_lane_coors[p_idx][0]))
            pt2 = (int(pred_lane_coors[p_idx + 1][1]), int(pred_lane_coors[p_idx + 1][0]))
            cv2.line(raw_img, pt1, pt2, color=1, thickness=1)
    return raw_img
def calc_measures_conf_buffer(arr_label, arr_pred, buff_radius=2):
    '''
    * in : arr_label (np.array, float, 1152x1152)
    * in : arr_pred (np.array, float, 1152x1152)
    * out: accuracy, precision, recall, f1
    '''
    assert arr_label.shape == arr_pred.shape
    H, W = arr_label.shape
    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    OCCUPIED = 1.
    NOT_OCCUPIED = 0.

    TP = 0
    FP = 0
    TP_2 = 0
    FN = 0
    for row in range(0, H):
        for col in range(0, W):
            col_min = max(0, col - buff_radius)
            col_max = min(W-1, col + buff_radius)

            pred_enhanced = 0
            label_enhanced = 0
            if temp_pred[row, col] == 1:
                for c_traverse in range(col_min, col_max+1):
                    if (temp_pred[row, col] == temp_label[row, c_traverse]):
                        pred_enhanced = OCCUPIED
                        break
                if pred_enhanced == OCCUPIED:
                    TP += 1
                else:
                    FP += 1

            if temp_label[row, col] == 1:
                for c_traverse in range(col_min, col_max + 1):
                    if (temp_label[row, col] == temp_pred[row, c_traverse]):
                        label_enhanced = OCCUPIED
                        break
                if label_enhanced == OCCUPIED:
                    TP_2 += 1
                else:
                    FN += 1
    TN = H * W - TP - FP - FN

    return TP, FP, FN, TN


def calc_measures_cls(arr_label, arr_pred):
    '''
    * in : arr_label (np.array, float, 144x144)
    * in : arr_pred (np.array, float, 144x144)
    * out: f1-score
    '''

    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    # F1 = TP/(TP+0.5*(FP+FN))
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # TP, FP, FN, #TN
    for j in range(1,143):
        for i in range(1,143):
            if not (temp_label[j,i] == 255): # Lane
                is_tp = False
                for jj in range(-1,2):
                    for ii in range(-1,2):
                        is_tp = is_tp or (temp_pred[j+jj,i+ii] == temp_label[j,i])
                if is_tp:
                    TP += 1
                else:
                    FN += 1
            else: # Not Lane
                if not (temp_pred[j,i] == 255):
                    FP += 1
                else:
                    TN += 1
    
    return TP, FP, FN, TN

def calc_measures_cls_wo_offset(arr_label, arr_pred):
    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    # F1 = TP/(TP+0.5*(FP+FN))
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # TP, FP, FN, #TN
    for j in range(144):
        for i in range(144):
            if not (temp_label[j,i] == 255): # Lane
                if temp_pred[j,i] == temp_label[j,i]:
                    TP += 1
                else:
                    FN += 1
            else: # Not Lane
                if not (temp_pred[j,i] == 255):
                    FP += 1
                else:
                    TN += 1
    
    return TP, FP, FN, TN

def tuple2list(t):
    return [[t[0][x], t[1][x]] for x in range(len(t[0]))]

'''
This function is to evaluate the line/polyline segmentation results.
'''
def eval_metric_line_segmentor(seg_result, mask, bi_seg=True, semantics=2, buff=10):  # The number of semantics here is without the background
    '''
    Evaluate the predicted image by F1 score during evaluation
    '''
    # print("seg result shape herehere", seg_result.dtype)
    
    all_TPs = 0  # The total number of the segmented true positives, corresponding to the all segmented pts
    all_DGs = 0  # The total number of the segmented ture positives, cooresponding to the all ground truth pts
    all_seg_pts = 0
    all_gt_pts = 0
    graph_acc = 0.
    graph_recall = 0.

    if bi_seg:
        skel = skeletonize(seg_result.astype(np.int8), method='lee')  # akeletonize function is applied on a binary image
        gt_points = tuple2list(np.where(mask != 0))
        graph_points = tuple2list(np.where(skel != 0))
        if (len(gt_points) > 0):
            gt_tree = scipy.spatial.cKDTree(gt_points)
            for c_i, thre in enumerate([buff]):
                if (len(graph_points) > 0):
                    graph_tree = scipy.spatial.cKDTree(graph_points)
                    graph_dds, _ = graph_tree.query(gt_points, k=1)
                    gt_acc_dds, gt_acc_iis = gt_tree.query(graph_points, k=1)
                    all_TPs = len([x for x in gt_acc_dds if x < thre])
                    all_seg_pts = len(gt_acc_dds)
                    all_DGs = len([x for x in graph_dds if x < thre])
                    all_gt_pts = len(graph_dds)
                else:
                    all_gt_pts = len(gt_points)
        else:
            all_TPs += 0
            all_seg_pts += len(graph_points)
    else:
        for semantic_id in range(semantics):
            semantic_id += 1  # the background is 0; other semantics starts from 1.
            before_skel = np.zeros_like(seg_result)
            before_skel[np.where(seg_result==semantic_id)] = 1
            skel = skeletonize(before_skel.astype(np.int8), method='lee')
            gt_points = tuple2list(np.where(mask == semantic_id))
            graph_points = tuple2list(np.where(skel != 0))
            if (len(gt_points) > 0):
                gt_tree = scipy.spatial.cKDTree(gt_points)
                for c_i, thre in enumerate([buff]):
                    if (len(graph_points) > 0):
                        graph_tree = scipy.spatial.cKDTree(graph_points)
                        graph_dds, _ = graph_tree.query(gt_points, k=1)
                        gt_acc_dds, gt_acc_iis = gt_tree.query(graph_points, k=1)

                        all_DGs += len([x for x in graph_dds if x < thre])
                        all_gt_pts += len(graph_dds)
                        all_TPs += len([x for x in gt_acc_dds if x < thre])
                        all_seg_pts += len(gt_acc_dds)
                    else:
                        all_gt_pts += len(gt_points)
            else:  
                all_TPs += 0
                all_seg_pts += len(graph_points)
            
    if all_seg_pts > 0:
        graph_acc = all_TPs / all_seg_pts
    if all_gt_pts > 0:
        graph_recall = all_DGs / all_gt_pts
    r_f = 0
    if graph_acc * graph_recall:
        r_f = 2 * graph_recall * graph_acc / (graph_acc + graph_recall)
    return graph_acc, graph_recall, r_f, all_TPs, all_seg_pts, all_DGs, all_gt_pts

def eval_metric_endp_detector(endp_pred, endp_gt, r_thre=10):
    gt_points = tuple2list(np.where(endp_gt > 0.99))
    graph_points = tuple2list(np.where(endp_pred >0.99))

    all_TPs = 0  # The total number of the segmented true positives, corresponding to the all segmented pts
    all_DGs = 0  # The total number of the segmented ture positives, cooresponding to the all ground truth pts
    all_seg_pts = 0
    all_gt_pts = 0
    graph_acc = 0.
    graph_recall = 0.

    if(len(gt_points) > 0):
        gt_tree = scipy.spatial.cKDTree(gt_points)
    for c_i, thre in enumerate([r_thre]):
        if (len(graph_points) > 0) and (len(gt_points) > 0) :
            graph_tree = scipy.spatial.cKDTree(graph_points)
            graph_dds, _ = graph_tree.query(gt_points, k=1)
            gt_acc_dds, gt_acc_iis = gt_tree.query(graph_points, k=1)
            all_DGs += len([x for x in graph_dds if x < thre])
            all_gt_pts += len(graph_dds)
            all_TPs += len([x for x in gt_acc_dds if x < thre])
            all_seg_pts += len(gt_acc_dds)

    r_f = 0
    if all_seg_pts > 0.:
        graph_acc = all_TPs / all_seg_pts
    if all_gt_pts > 0.:
        graph_recall = all_DGs / all_gt_pts
    if (graph_acc + graph_recall) > 0.:
        r_f = 2 * graph_recall * graph_acc / (graph_acc + graph_recall)

    return graph_acc, graph_recall, r_f, all_TPs, all_seg_pts, all_DGs, all_gt_pts