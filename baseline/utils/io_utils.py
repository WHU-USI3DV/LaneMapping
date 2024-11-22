'''
@Author: Xiaoxin Mi
@E-mail: mixiaoxin@whu.edu.cn
'''
import os
import numpy as np
import json

from data.convert_data import NpEncoder
# import NpEncoder
    
def save_seqs_json(seq_list, seq_path):
    with open(seq_path, "w") as f:
        json.dump(seq_list, f, indent=4, cls=NpEncoder)
        # print("Finish save sequence in ", seqs_filename)
        f.close()

def save_seqs_txt(seq_list, seq_path):
    with open(seq_path, "w") as f:
        for id, seq in enumerate(seq_list):
            for idx_v in seq['seq']:
                str_out = ' '.join(str(item) for item in idx_v)
                str_out += (' ' + str(id) + '\n')
                f.write(str_out)

        f.close()

def save_seqs_list(lane_vertexes, lane_seq_path):
    # 1. get the concise vertex:
    num_lane = len(lane_vertexes)
    seqs = []
    seq_len = np.zeros(num_lane)
    max_seq_l = 0
    lines_pred = []
    for idx_lane in range(num_lane):
        pred_vert = lane_vertexes[idx_lane]
        seqs.append(pred_vert)
        seq_len[idx_lane] = pred_vert.shape[0]
        if pred_vert.shape[0] > max_seq_l:
            max_seq_l = pred_vert.shape[0]

        if pred_vert.shape[0] < 2:
            continue
        line = {}
        line['seq_len'] = pred_vert.shape[0]
        line['seq'] = pred_vert
        line['init_vertex'] = pred_vert[0, :]
        line['end_vertex'] = pred_vert[-1, :]
        lines_pred.append(line)

    # 2. save in json file:
    (filepath_stem, filepath_ext) = os.path.splitext(lane_seq_path)
    if filepath_ext == '.txt':
        save_seqs_txt(lines_pred, lane_seq_path)
    else:
        save_seqs_json(lines_pred, lane_seq_path)

def save_lane_seq_2d(lane_vertexes, lane_seq_path, with_pervertex_semantics=True):
    # 1. get the concise vertex:
    num_lane, max_ver_num, _ = lane_vertexes.shape
    seqs = []
    seq_len = np.zeros(num_lane)
    max_seq_l = 0
    lines_pred = []
    for idx_lane in range(num_lane):
        pred_vert = lane_vertexes[idx_lane, :, :]
        pred_vert = pred_vert[np.where(pred_vert[:, 1] > 0)]  # num_vert, 2
        seqs.append(pred_vert)
        seq_len[idx_lane] = pred_vert.shape[0]
        if pred_vert.shape[0] > max_seq_l:
            max_seq_l = pred_vert.shape[0]

        if pred_vert.shape[0] < 2:
            continue
        line = {}
        line['seq_len'] = pred_vert.shape[0]
        if with_pervertex_semantics:
            line['seq'] = pred_vert # with pervertex semantics
            line['init_vertex'] = pred_vert[0, :]
            line['end_vertex'] = pred_vert[-1, :]
        else:
            # without pervertex semantics
            line['seq'] = pred_vert[:, :-1]  # with pervertex semantics
            line['init_vertex'] = pred_vert[0, :-1]
            line['end_vertex'] = pred_vert[-1, :-1]
        lines_pred.append(line)

    # 2. save in json file:
    (filepath_stem, filepath_ext) = os.path.splitext(lane_seq_path)
    if filepath_ext == '.txt':
        save_seqs_txt(lines_pred, lane_seq_path)
    else:
        save_seqs_json(lines_pred, lane_seq_path)

'''
@brief: load sequeeze information from .json file
if the lane vertexes have 2 dimensions, the @praram:dim_coor=2;
when the lane vertexes have 3 dimensions, the @param:dim_coor=3. 
'''
def load_lane_seq(seqfile_path, dim_coor=2):
    with open(seqfile_path) as json_file:
        load_json = json.load(json_file)
        data_json = load_json

    seq_lens = []
    init_points = []
    end_points = []

    # print("data json: ", len(data_json))
    for area in data_json:
        # print("area: ", area)
        seq_lens.append(area['seq_len'])
        init_points.append(area['init_vertex'])
        end_points.append(area['end_vertex'])
    if len(seq_lens) < 2:
        seq = []
    else:
        seq = np.zeros((len(seq_lens), max(seq_lens), dim_coor))
        for idx, area in enumerate(data_json):
            if seq_lens[idx] == 0:
                continue
            seq[idx, :seq_lens[idx]] = [x[0:dim_coor] for x in area['seq']]
    return seq, seq_lens, init_points, end_points

def load_pc_2_img_transform_paras(param_path):
    # opening the file in read mode
    my_file = open(param_path, "r")
    # reading the file
    data = my_file.read()
    # replacing end splitting the text when newline ('\n') is seen.
    data_into_list = data.split("\n")
    # print(data_into_list)
    my_file.close()

    params = {}
    params['coor_las_path'] = data_into_list[1]
    params['las_read_offset'] = data_into_list[3]
    params['las_rotation_trans_quan'] = data_into_list[5]
    params['bev_img_offset'] = data_into_list[7]
    params['img_reso'] = data_into_list[9]
    params['local_min_ele'] = data_into_list[11]
    params['ele_reso'] = data_into_list[13]

    params['las_read_offset'] = [float(item) for item in params['las_read_offset'].split(' ')]
    params['las_rotation_trans_quan'] = [float(item) for item in params['las_rotation_trans_quan'].split(' ')]
    params['bev_img_offset'] = [float(item) for item in params['bev_img_offset'].split(' ')]
    params['img_reso'] = [float(item) for item in params['img_reso'].split(' ')]
    params['local_min_ele'] = float(params['local_min_ele'])
    params['ele_reso'] = float(params['ele_reso'])
    return params

if __name__=="__main__":
    param_path = "/home/mxx/Desktop/181013_0130.txt"
    tmp_params = load_pc_2_img_transform_paras(param_path)
    print("temp_params: ", tmp_params)


