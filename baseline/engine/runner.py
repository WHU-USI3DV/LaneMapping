'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* Modified by mixiaoxin@whu.edu.cn
'''
import shutil
from sys import api_version
import time
import torch
import torchvision.utils
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from baseline.models.registry import build_net
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from baseline.datasets import build_dataloader
from baseline.datasets import build_dataset
from baseline.utils.metric_utils import calc_measures, cal_coor_measures, eval_metric_line_segmentor, eval_metric_endp_detector
from baseline.utils.net_utils import save_model, load_network
from baseline.utils.dist_utils import dist_tqdm
from baseline.utils.io_utils import save_lane_seq_2d
EPS = 1e-16

## for DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from baseline.utils import synchronize
from baseline.utils.config import Config

# to load data with various length
from mmengine.dataset import pseudo_collate

# new version referring the Pytorch homepage tutorial
def ddp_setup(rank: int, world_size: int):
    """

    Args:
        rank: unique identifier of each process
        world_size: total number of process
    Returns:

    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def load_config_and_runner(path_config, gpus):
    cfg = Config.fromfile(path_config)
    cfg.log_dir = cfg.log_dir + '/vis'
    os.makedirs(cfg.log_dir, exist_ok=True)
    cfg.work_dirs = cfg.log_dir + '/' + cfg.dataset.train.type
    os.makedirs(cfg.work_dirs, exist_ok=True)
    cfg.gpus = len(gpus.split(','))
    runner = Runner(cfg)

    return cfg, runner
class Runner(object):
    def __init__(self, cfg, rank=0):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.gpu_id = rank
        self.cfg = cfg
        self.log_dir = cfg.log_dir
        self.epoch = 0
        self.train_dataloader = None

        ### Custom logs ###
        self.batch_bar = None #tqdm(total = 1, desc = 'batch', position = 1)
        self.val_bar = None #tqdm(total = 1, desc = 'val', position = 2)
        self.info_bar = tqdm(total = 0, position = 3, bar_format='{desc}')
        self.val_info_bar = tqdm(total = 0, position = 4, bar_format='{desc}')

        self.tensor_vis_writer = SummaryWriter(self.log_dir)
        ### Custom logs ###
        
        self.net = build_net(self.cfg)
        # self.net.to(torch.device('cuda'))
        if self.cfg.distributed:
            if self.gpu_id == 0:
                with open('.work_dir_tmp_file.txt', 'w') as f:
                    f.write(self.cfg.work_dirs)
            else:
                while not os.path.exists('.work_dir_tmp_file.txt'):
                    time.sleep(0.1)
                with open('.work_dir_tmp_file.txt', 'r') as f:
                    work_dir = f.read().strip()
            synchronize()
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net = self.net.to(self.gpu_id)
            self.net = DDP(self.net, device_ids=[self.gpu_id], find_unused_parameters=True)
        else:
            self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(self.cfg.gpus)).cuda()

        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        
        self.warmup_scheduler = None
        if self.cfg.optimizer.type == 'SGD':
            self.warmup_scheduler = warmup.LinearWarmup(
                self.optimizer, warmup_period=5000)
                
        self.metric = 0.
        self.val_loader = None

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from,
                finetune_from=self.cfg.finetune_from)
        

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if k == 'image_name':   # add by Xiaoxin
                continue
             # batch[k] = batch[k].cuda()
            ################################################
            # original only bev images as input
            # batch[k] = batch[k].cuda(non_blocking=True)
            #################################################
            
            #################################################
            # Xiaoxin modify this to fit mmdetection3d
            if isinstance(batch[k], list):
                if isinstance(batch[k][0], torch.Tensor):
                    batch[k] = [ item.unsqueeze(0) for item in batch[k]]
                    batch[k] = torch.cat(batch[k], dim=0).cuda()
                elif isinstance(batch[k][0], np.ndarray):
                    batch[k] = [ torch.from_numpy(item).unsqueeze(0) for item in batch[k]]
                    batch[k] = torch.cat(batch[k], dim=0).cuda()
                else:
                    batch[k] = [item.cuda() for item in batch[k]]
            else:
                batch[k] = batch[k].cuda(non_blocking=True)
            ####################################################

        return batch
    
    def write_to_log(self, log, log_file_name):
        f = open(log_file_name, 'a')
        f.write(log)
        f.close()
    
    def train_epoch(self, epoch):
        b_sz = self.train_dataloader.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_dataloader)}")

        self.net.train()
        end = time.time()
        max_iter = len(self.train_dataloader)
        self.batch_bar = tqdm(total = max_iter, desc='batch', position=1)

        for i, data in enumerate(self.train_dataloader):
            if i == max_iter - 1:
                continue

            date_time = time.time() - end
            self.optimizer.zero_grad()
            data = self.to_cuda(data)
            output = self.net(data)
            loss = output['loss']

            if torch.isfinite(loss):
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.warmup_scheduler:
                    self.warmup_scheduler.dampen()
                batch_time = time.time() - end
                end = time.time()

                ### Logging ###
                log_train = f'epoch={epoch}/{self.cfg.epochs}, loss={loss.detach().cpu()}'
                loss_stats = output['loss_stats']
                for k, v in loss_stats.items():
                    log_train += f', {k}={v.detach().cpu()}'
                    self.tensor_vis_writer.add_scalar(k, v, epoch * max_iter + i)
                # show the trained images in tensorboard
                # img_grid = torchvision.utils.make_grid(data['proj'][0].detach().cpu())
                # plt.imshow(img_grid.permute(1, 2, 0))
                # self.tensor_vis_writer.add_image('road surface images: ', img_grid)
                # self.tensor_vis_writer.add_graph(self.net, data)  # only tuple availabel

                self.info_bar.set_description_str(log_train)
                self.write_to_log(log_train + '\n', os.path.join(self.log_dir, 'train.txt'))

                ### Logging ###
            else:
                # pass
                print(f'problem index = {i}')
                self.write_to_log(f'problem index = {i}' + '\n', os.path.join(self.log_dir, 'prob.txt'))

            self.batch_bar.update(1)

    def train(self):
        self.train_dataloader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)
        for epoch in range(self.cfg.epochs):
            if self.cfg.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            if ((epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1) and self.gpu_id == 0:
                self.save_ckpt(epoch)
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate(epoch)



    def validate(self, epoch=None, is_small=False, valid_samples=40):
        if is_small:
            self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=True)
        else:
            if not self.val_loader:
                self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
        self.net.eval()

        if is_small:
            self.val_bar = tqdm(total = valid_samples, desc='val', position=2)
        else:
            self.val_bar = tqdm(total = len(self.val_loader), desc='val', position=2)

        # for segmentation / polyline detection
        list_coor_conf_f1 = []
        coor_TP = 0   # the count of the true positive in segmented pts
        coor_DG = 0   # the count of the detected ground truth
        coor_segs = 0  # the total number of the segmented pts
        coor_gts = 0   # the total number of the ground truth pts
        f1_ave = 0

        # for endpoints
        endp_TP = 0
        endp_DG = 0
        endp_dets = 0
        endp_gts = 0
        endp_f1_av = 0
        for i, data in enumerate(self.val_loader):
            if is_small:
                if i > valid_samples:
                    break
            data = self.to_cuda(data)
            with torch.no_grad():

                if self.cfg.net.type == 'Segmentor':
                    result_infer = self.net(data)
                    seg_map_pred = result_infer['seg'].cpu().detach().numpy()
                    seg_map_gt = data['mask'].cpu().detach().numpy()
                    batch_f1 = 0.

                    endp_map_pred = result_infer['endp'].cpu().detach().numpy()
                    endp_map_gt = data['endp_map'].cpu().detach().numpy()

                    for batch_idx in range(self.cfg.batch_size):
                        b_acc, b_recall, b_f1, TPs, seg_pts, DGs, gt_pts = eval_metric_line_segmentor(seg_map_pred[batch_idx, ...], seg_map_gt[batch_idx, ...], bi_seg=False, semantics=2)
                        batch_f1 = (batch_f1*batch_idx + b_f1) / (batch_idx+1)
                        coor_TP += TPs
                        coor_segs += seg_pts
                        coor_DG += DGs
                        coor_gts += gt_pts

                        _, _, _, TP2, dets, DG2, gts = eval_metric_endp_detector(endp_map_pred[batch_idx, ...], endp_map_gt[batch_idx, ...], r_thre=10)
                        endp_TP += TP2
                        endp_dets += dets
                        endp_DG += DG2
                        endp_gts += gts
                    f1_ave = (f1_ave * i + batch_f1) / (i + 1)
                elif self.cfg.heads.type == 'RowSharNotReducRef':
                    output = self.net(data)
                    lane_maps = output['lane_maps']
                    # print(lane_maps.keys())
                    batch_f1 = 0.
                    for batch_idx in range(self.cfg.batch_size):
                        coor_label = lane_maps['coor_label'][batch_idx]
                        cls_coors = lane_maps['cls_offset_smooth'][batch_idx]  # offset-added after smoothing
                        # Coordinates
                        b_acc, b_recall, b_f1, TPs, seg_pts, DGs, gt_pts = \
                            cal_coor_measures(coor_label, cls_coors, 'conf', offset_thre=16)
                        batch_f1 = (batch_f1 * batch_idx + b_f1) / (batch_idx + 1)
                        coor_TP += TPs
                        coor_segs += seg_pts
                        coor_DG += DGs
                        coor_gts += gt_pts
                else:
                    output = self.net(data)
                    lane_maps = output['lane_maps']
                    endp_map_gt = data['endp_map'].cpu().detach().numpy()

                    data_length = len(lane_maps['coor_label'])
                    batch_f1 = 0.
                    for batch_idx in range(data_length):
                    # for batch_idx in range(self.cfg.batch_size):
                        coor_label = lane_maps['coor_label'][batch_idx]
                        # endp_map_pred = output['endp'][batch_idx]

                        # conf_exist = lane_maps['exist_pred'][batch_idx]
                        # cls_coors = lane_maps['cls_coor_pred'][batch_idx]  # original
                        # cls_coors = lane_maps['cls_coor_pred_smooth'][batch_idx]  # original after smoothing
                        # cls_coors = lane_maps['cls_exp_smooth'][batch_idx]  # expectation after smoothing
                        cls_coors = lane_maps['cls_offset_smooth'][batch_idx][:, :, 0]  # offset-added after smoothing

                        # Coordinates
                        b_acc, b_recall, b_f1, TPs, seg_pts, DGs, gt_pts = cal_coor_measures(coor_label, cls_coors, 'conf', offset_thre=self.cfg.validate_buffer)
                        batch_f1 = (batch_f1 * batch_idx + b_f1) / (batch_idx + 1)
                        coor_TP += TPs
                        coor_segs += seg_pts
                        coor_DG += DGs
                        coor_gts += gt_pts

                    f1_ave = (f1_ave * i + batch_f1) / (i + 1)

            self.val_bar.update(1)
        coor_con_pre = 0.
        coor_con_rec = 0.
        endp_con_pre = 0.
        endp_con_rec = 0.
        if coor_segs > 0:
            coor_con_pre = coor_TP / (coor_segs + EPS)
        if coor_gts > 0.:
            coor_con_rec = coor_DG / (coor_gts + EPS)
        if endp_dets > 0.:
            endp_con_pre = endp_TP / (endp_dets + EPS)
        if endp_gts > 0.:
            endp_con_rec = endp_DG / (endp_gts + EPS)

        coor_con_f1s = 0.
        endp_conf_f1s = 0.
        if ((coor_con_rec + coor_con_pre) > 0):
            coor_con_f1s = 2. * coor_con_rec * coor_con_pre / (coor_con_rec + coor_con_pre)
        if ((endp_con_rec + endp_con_pre) > 0):
            endp_conf_f1s = 2.0 * endp_con_rec * endp_con_pre / (endp_con_rec + endp_con_pre)
        metric = 0.9 * coor_con_f1s + 0.1 * endp_conf_f1s
        if metric > self.metric:
            self.metric = metric
            self.save_ckpt(epoch, is_best=True)

        ### Logging ###
        log_val = f'epoch = {epoch}, {coor_con_pre}, {coor_con_rec}, {coor_con_f1s},{endp_con_pre}, {endp_con_rec}, {endp_conf_f1s}'
        self.write_to_log(log_val + '\n', os.path.join(self.log_dir, 'val.txt'))
        self.val_info_bar.set_description_str(log_val)
        ### Logging ###



    def save_ckpt(self, epoch, is_best=False):
        if self.cfg.distributed == True:
            save_model(self.net.module, self.optimizer, self.scheduler, epoch, self.cfg.log_dir, is_best=is_best)
        else:
            save_model(self.net, self.optimizer, self.scheduler, epoch, self.cfg.log_dir, is_best=is_best)

    ### Small dataset ###
    def train_epoch_small(self, epoch, train_loader, maximum_batch = 200):
        self.net.train()
        self.batch_bar = tqdm(total = maximum_batch, desc='batch', position=1)

        for i, data in enumerate(train_loader):
            if i > maximum_batch:
                break
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if self.warmup_scheduler:
                self.warmup_scheduler.dampen()

            ### Logging ###
            log_train = f'epoch={epoch}/{self.cfg.epochs}, loss={loss.detach().cpu()}'
            self.info_bar.set_description_str(log_train)
            self.write_to_log(log_train + '\n', os.path.join(self.log_dir, 'train.txt'))
            ### Logging ###

            self.batch_bar.update(1)

    def train_small(self, train_batch = 200, valid_samples = 80):
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)

        for epoch in range(self.cfg.epochs):
            self.train_epoch_small(epoch, train_loader, train_batch)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt(epoch)
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate(epoch, is_small=True, valid_samples=valid_samples)
    
    def load_ckpt(self, path_ckpt):
        trained_model = torch.load(path_ckpt)
        self.net.load_state_dict(trained_model['net'], strict=True)


    def process_one_sample(self, path_ckpt=None, mode_show=True):
        self.cfg.batch_size = 1
        self.val_loader = build_dataloader(self.cfg.dataset.single, self.cfg, is_train=False)
        if path_ckpt:
            trained_model = torch.load(path_ckpt)
            self.net.load_state_dict(trained_model['net'], strict=True)
        self.net.eval()
        for i, data in enumerate(self.val_loader):
            # data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                lane_maps = output['lane_maps']
                pred_maps = output['pred_maps']
                # print(lane_maps.keys())

                for batch_idx in range(self.cfg.batch_size):
                    conf_label = lane_maps['conf_label'][batch_idx]
                    conf_label_raw = lane_maps['conf_label_raw'][batch_idx]
                    cls_label = lane_maps['cls_label'][batch_idx]
                    # exist_label = lane_maps['exist_label'][batch_idx]
                    coor_label = lane_maps['coor_label'][batch_idx]

                    # conf_by_cls = lane_maps['conf_by_cls'][batch_idx]
                    # cls_idx = lane_maps['cls_idx'][batch_idx]
                    # conf_exist = lane_maps['exist_pred'][batch_idx]
                    # cls_coors = lane_maps['cls_coor_pred'][batch_idx]  # original
                    # cls_coors = lane_maps['cls_coor_pred_smooth'][batch_idx]  # original after smoothing
                    # cls_coors = lane_maps['cls_exp_smooth'][batch_idx]  # expectation after smoothing
                    cls_coors = lane_maps['cls_offset_smooth'][batch_idx]  # offset-added after smoothing

                    # get the smoothing lane coordinate:
                    pred_lane_coors_smooth = np.zeros((self.cfg.heads.num_prop, 144, 2))
                    pred_lane_coors_smooth[:, :, 0] = np.arange(3, 1152, 8)
                    pred_lane_coors_smooth[:, :, 1] = output['lane_maps']['cls_coor_pred_smooth'][batch_idx]

                    # get the expectation lane coordinate:
                    pred_lane_coors_exp = np.zeros((self.cfg.heads.num_prop, 144, 2))
                    pred_lane_coors_exp[:, :, 0] = np.arange(3, 1152, 8)
                    pred_lane_coors_exp[:, :, 1] = output['lane_maps']['cls_exp_smooth'][batch_idx]

                    # Coordinates
                    pre_coor_conf, rec_coor_conf, f1_coor_conf, TP_coor_conf, FP_coor_conf, FN_coor_conf = \
                        cal_coor_measures(coor_label, cls_coors, 'conf', offset_thre=8)


                    print(f'local_coor_conf_prec={pre_coor_conf}')
                    print(f'local_coor_conf_rec={rec_coor_conf}')
                    print(f'local_coor_conf_f1={f1_coor_conf}')
                    print('\n')

                    if mode_show:
                        pred_lane_on_image = pred_maps['pred_lanes_on_image'][batch_idx]
                        pred_org_lane_on_img = pred_maps['pred_org_lanes_on_image'][batch_idx]
                        pred_smooth_lane_on_img = pred_maps['pred_smooth_lanes_on_image'][batch_idx]
                        pred_smooth_lane_vertex = pred_maps['pred_smooth_lane_vertex'][batch_idx]
                        pred_bi_seg_on_img = pred_maps['pred_bi_seg_on_image'][batch_idx]
                        pred_exp_lane_on_img = pred_maps['pred_exp_lanes_on_image'][batch_idx]
                        pred_offset_lane_on_img = pred_maps['pred_offset_lanes_on_image'][batch_idx]


                        cv2.imshow('pred_org', pred_org_lane_on_img/255.)
                        cv2.imshow('pred_org_smooth', pred_smooth_lane_on_img/255.)
                        cv2.imshow('pred_exp_smooth', pred_exp_lane_on_img/255.)
                        cv2.imshow('pred_offset_smooth', pred_offset_lane_on_img/255.)
                        cv2.imshow('pred_seg', pred_bi_seg_on_img/255.)
                        cv2.waitKey(0)

        return 1

    def infer_lane(self, path_ckpt=None, mode_imshow=None, is_calc_f1=False, write_lane_vertex=False):
        self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
        
        if path_ckpt:
            trained_model = torch.load(path_ckpt)
            self.net.load_state_dict(trained_model['net'], strict=True)
        con_TP = 0
        con_FP = 0
        con_FN = 0
        cls_TP = 0
        cls_FP = 0
        cls_FN = 0
        con_TP_s = 0
        con_FP_s = 0
        con_FN_s = 0
        cls_TP_s = 0
        cls_FP_s = 0
        cls_FN_s = 0

        coor_con_TP = 0
        coor_con_FP = 0
        coor_con_FN = 0
        coor_cls_TP = 0
        coor_cls_FP = 0
        coor_cls_FN = 0

        self.net.eval()
        for i, data in enumerate(self.val_loader):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                # print(output)

                lane_maps = output['lane_maps']
                pred_maps = output['pred_maps']
                # print(lane_maps.keys())


                for batch_idx in range(self.cfg.batch_size):
                    conf_label = lane_maps['conf_label'][batch_idx]
                    conf_label_raw = lane_maps['conf_label_raw'][batch_idx]
                    cls_label = lane_maps['cls_label'][batch_idx]
                    # exist_label = lane_maps['exist_label'][batch_idx]
                    coor_label = lane_maps['coor_label'][batch_idx]

                    # conf_by_cls = lane_maps['conf_by_cls'][batch_idx]
                    # cls_idx = lane_maps['cls_idx'][batch_idx]
                    # conf_exist = lane_maps['exist_pred'][batch_idx]
                    # cls_coors = lane_maps['cls_coor_pred'][batch_idx]  # original
                    # cls_coors = lane_maps['cls_coor_pred_smooth'][batch_idx]  # original after smoothing
                    # cls_coors = lane_maps['cls_exp_smooth'][batch_idx]  # expectation after smoothing
                    cls_coors = lane_maps['cls_offset_smooth'][batch_idx]  # offset-added after smoothing

                    # get the smoothing lane coordinate:
                    pred_lane_coors_smooth = np.zeros((self.cfg.heads.num_prop, 144, 2))
                    pred_lane_coors_smooth[:, :, 0] = np.arange(3, 1152, 8)
                    pred_lane_coors_smooth[:, :, 1] = output['lane_maps']['cls_coor_pred_smooth'][batch_idx]

                    # get the expectation lane coordinate:
                    pred_lane_coors_exp = np.zeros((self.cfg.heads.num_prop, 144, 2))
                    pred_lane_coors_exp[:, :, 0] = np.arange(3, 1152, 8)
                    pred_lane_coors_exp[:, :, 1] = output['lane_maps']['cls_exp_smooth'][batch_idx]

                    # Coordinates
                    pre_coor_conf, rec_coor_conf, f1_coor_conf, TP_coor_conf, FP_coor_conf, FN_coor_conf = \
                        cal_coor_measures(coor_label, cls_coors, 'conf', offset_thre = 16)

                    coor_con_TP += TP_coor_conf
                    coor_con_FP += FP_coor_conf
                    coor_con_FN += FN_coor_conf

                    print(f'local_coor_conf_prec={pre_coor_conf}')
                    print(f'local_coor_conf_rec={rec_coor_conf}')
                    print(f'local_coor_conf_f1={f1_coor_conf}')
                    print('\n')


                    if mode_imshow == 'cls' or mode_imshow == 'all':
                        pred_lane_on_image = pred_maps['pred_lanes_on_image'][batch_idx]
                        pred_org_lane_on_img = pred_maps['pred_org_lanes_on_image'][batch_idx]
                        pred_smooth_lane_on_img = pred_maps['pred_smooth_lanes_on_image'][batch_idx]
                        pred_smooth_lane_vertex = pred_maps['pred_smooth_lane_vertex'][batch_idx]
                        pred_bi_seg_on_img = pred_maps['pred_bi_seg_on_image'][batch_idx]
                        pred_exp_lane_on_img = pred_maps['pred_exp_lanes_on_image'][batch_idx]
                        pred_offset_lane_on_img = pred_maps['pred_offset_lanes_on_image'][batch_idx]
                        # cv2.imshow('pred_lanes', pred_lane_on_image)
                        # cv2.waitKey(0)
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '.png'),
                                    pred_lane_on_image)
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_org.png'),
                                    pred_org_lane_on_img)
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_smooth.png'),
                                    pred_smooth_lane_on_img)
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_exp.png'),
                                    pred_exp_lane_on_img)
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_offset.png'),
                                    pred_offset_lane_on_img)
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_seg.png'),
                                    pred_bi_seg_on_img)
                        if write_lane_vertex:
                            # TODO: write out the vertex per batch in json file
                            pred_lane_path = os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '.json')
                            save_lane_seq_2d(pred_smooth_lane_vertex, pred_lane_path)


                    if mode_imshow == 'conf' or mode_imshow == 'all':
                        # cv2.imshow('conf_by_cls', conf_by_cls.astype(np.uint8)*255)
                        cv2.imshow('label', conf_label.astype(np.uint8)*255)

                    if mode_imshow == 'rgb' or mode_imshow == 'all':
                        rgb_cls_label = lane_maps['rgb_cls_label'][batch_idx]
                        rgb_cls_idx = lane_maps['rgb_cls_idx'][batch_idx]
                        rgb_conf_cls_idx = lane_maps['rgb_conf_cls_idx'][batch_idx]
                        cv2.imshow('rgb_cls_label', rgb_cls_label)
                        cv2.imshow('rgb_cls_idx', rgb_cls_idx)
                        cv2.imshow('rgb_conf_cls_idx', rgb_conf_cls_idx)
                        cv2.waitKey(0)

                        # cv2.imwrite(os.path.join(self.log_dir, 'rgb', data['meta']['image_name'][batch_idx][-16:-5] + 'rgb_conf_cls_idx' + '.jpg'), rgb_conf_cls_idx)
                        # cv2.imwrite(os.path.join(self.log_dir, 'rgb', data['meta']['image_name'][batch_idx][-16:-5] + 'rgb_cls_idx' + '.jpg'), rgb_cls_idx)

                    if not mode_imshow == None:
                        cv2.waitKey(0)

        coor_con_pre = coor_con_TP / (coor_con_TP + coor_con_FP + EPS)
        coor_con_rec = coor_con_TP / (coor_con_TP + coor_con_FN + EPS)
        coor_con_f1s = (2 * coor_con_TP / (2 * coor_con_TP + coor_con_FP + coor_con_FN + EPS))

        print(f'coor_conf_prec={coor_con_pre}')
        print(f'coor_conf_rec={coor_con_rec}')
        print(f'coor_conf_f1={coor_con_f1s}')

    # 
    def infer_lane_coordinate(self, path_ckpt=None, mode_view=False, gt_avail=True, write_lane_vertex=False):
        self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)

        if path_ckpt:
            trained_model = torch.load(path_ckpt)
            self.net.load_state_dict(trained_model['net'], strict=True)
        
        self.net.eval()
        self.val_bar = tqdm(total=len(self.val_loader), desc='val', position=2)
        
        # for segmentation / polyline detection
        list_coor_conf_f1 = []
        coor_TP = 0  # the count of the true positive in segmented pts
        coor_DG = 0  # the count of the detected ground truth
        coor_segs = 0  # the total number of the segmented pts
        coor_gts = 0  # the total number of the ground truth pts
        f1_ave = 0

        for i, data in enumerate(self.val_loader):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                ########################################################################################################
                ### quantitative evaluation ###
                if gt_avail:
                    lane_maps = output['lane_maps']
                    endp_map_gt = data['endp_map'].cpu().detach().numpy()
                    semantic_map_gt = data['mask'].cpu().detach().numpy()
                    batch_f1 = 0.
                    total_batch_size = self.cfg.batch_size
                    if i == (len(self.val_loader)-1):
                        total_batch_size = len(lane_maps['coor_label'])
                for batch_idx in range(total_batch_size):
                    if gt_avail:
                        coor_label = lane_maps['coor_label'][batch_idx]
                        cls_coors = lane_maps['cls_offset_smooth'][batch_idx][:, :]  # for Klane

                        # Coordinates
                        b_acc, b_recall, b_f1, TPs, seg_pts, DGs, gt_pts = \
                            cal_coor_measures(coor_label, cls_coors, 'conf', offset_thre=self.cfg.validate_buffer)
                        batch_f1 = (batch_f1 * batch_idx + b_f1) / (batch_idx + 1)
                        coor_TP += TPs
                        coor_segs += seg_pts
                        coor_DG += DGs
                        coor_gts += gt_pts
                    
                    f1_ave = (f1_ave * i + batch_f1) / (i + 1)

                ########################################################################################################
                    ### begin: qualitative evaluation ###
                   
                    if mode_view: 
                        pred_maps = output['pred_maps']
                        pred_offset_lane_on_img = pred_maps['pred_offset_lanes_on_image'][batch_idx]
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_klane.png'),
                                pred_offset_lane_on_img)
                        
                        if self.cfg.view_detail:
                            pass
                    if write_lane_vertex:
                        # TODO: write out the vertex per batch in json file
                        pred_maps = output['pred_maps']
                        pred_smooth_lane_vertex = pred_maps['pred_smooth_lane_vertex'][batch_idx]
                        pred_lane_path = os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '.json')
                        save_lane_seq_2d(pred_smooth_lane_vertex, pred_lane_path, with_pervertex_semantics=True)
                ### end: qualitative evaluation ###
            self.val_bar.update(1)


        coor_pre = 0.
        coor_rec = 0.
        coor_f1s = 0.

        if gt_avail:
            coor_pre = coor_TP / (coor_segs + EPS)
            coor_rec = coor_DG / (coor_gts + EPS)
            if (coor_pre + coor_rec) > 0.:
                coor_f1s = 2. * coor_pre * coor_rec / (coor_pre + coor_rec)
        
        print(f'coordinate_prec={coor_pre}')
        print(f'coordinate_rec={coor_rec}')
        print(f'coordinate_f1={coor_f1s}')
        

    def infer_lane_coordinate_endpoint_semantics(self, path_ckpt=None, mode_data=None, mode_view=False, gt_avail=True, \
                                                 write_lane_vertex=False, \
                                                 eval_coor=True, eval_endp=True, eval_semantic=True):
        if mode_data==None:
            self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
        else:
            self.val_loader = build_dataloader(mode_data, self.cfg, is_train=False)
        print('data_loader length: ', len(self.val_loader))
        if path_ckpt:
            trained_model = torch.load(path_ckpt)
            self.net.load_state_dict(trained_model['net'], strict=True)

        self.net.eval()
        self.val_bar = tqdm(total=len(self.val_loader), desc='val', position=2)
        # for segmentation / polyline detection
        list_coor_conf_f1 = []
        coor_TP = 0  # the count of the true positive in segmented pts
        coor_DG = 0  # the count of the detected ground truth
        coor_segs = 0  # the total number of the segmented pts
        coor_gts = 0  # the total number of the ground truth pts
        f1_ave = 0

        # for endpoints
        endp_TP = 0
        endp_DG = 0
        endp_dets = 0
        endp_gts = 0
        endp_f1_av = 0

        # for semantics
        semantic_TP = 0
        semantic_DG = 0
        semantic_dets = 0
        semantic_gts = 0
        semantic_f1_av = 0
        for i, data in enumerate(self.val_loader):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                ########################################################################################################
                ### quantitative evaluation ###
                lane_maps = output['lane_maps']
                total_batch_size = self.cfg.batch_size
                if i == (len(self.val_loader)-1):
                    total_batch_size = len(lane_maps['coor_label'])
                        
                if gt_avail:  
                    endp_map_gt = data['endp_map'].cpu().detach().numpy()
                    semantic_map_gt = data['mask'].cpu().detach().numpy()
                    batch_f1 = 0.
                   
                for batch_idx in range(total_batch_size):
                    if gt_avail & eval_coor:
                        coor_label = lane_maps['coor_label'][batch_idx]
                        # conf_exist = lane_maps['exist_pred'][batch_idx]
                        # cls_coors = lane_maps['cls_coor_pred'][batch_idx]  # original
                        # cls_coors = lane_maps['cls_coor_pred_smooth'][batch_idx]  # original after smoothing
                        # cls_coors = lane_maps['cls_exp_smooth'][batch_idx]  # expectation after smoothing
                        cls_coors = lane_maps['cls_offset_smooth'][batch_idx][:, :, 0]  # for Xiaoxin's method: offset-added after smoothing, 0 is coordinates, 1 is semantics
                        # cls_coors = lane_maps['cls_offset_smooth'][batch_idx][:, :]  # for Klane

                        # get the smoothing lane coordinate:
                        # pred_lane_coors_smooth = np.zeros((self.cfg.heads.num_prop, 144, 2))
                        # pred_lane_coors_smooth[:, :, 0] = np.arange(3, 1152, 8)
                        # pred_lane_coors_smooth[:, :, 1] = lane_maps['cls_coor_pred_smooth'][batch_idx]

                        # get the expectation lane coordinate:
                        # pred_lane_coors_exp = np.zeros((self.cfg.heads.num_prop, 144, 2))
                        # pred_lane_coors_exp[:, :, 0] = np.arange(3, 1152, 8)
                        # pred_lane_coors_exp[:, :, 1] = lane_maps['cls_exp_smooth'][batch_idx]

                        # Coordinates
                        b_acc, b_recall, b_f1, TPs, seg_pts, DGs, gt_pts = \
                            cal_coor_measures(coor_label, cls_coors, 'conf', offset_thre=self.cfg.validate_buffer)
                        batch_f1 = (batch_f1 * batch_idx + b_f1) / (batch_idx + 1)
                        coor_TP += TPs
                        coor_segs += seg_pts
                        coor_DG += DGs
                        coor_gts += gt_pts

                    if gt_avail and eval_endp:
                        endp_map_pred = output['endp'][batch_idx]
                        _, _, _, TP2, dets, DG2, gts = eval_metric_endp_detector(endp_map_pred, endp_map_gt[batch_idx, ...],
                                                                             r_thre=self.cfg.validate_buffer*2)
                        endp_TP += TP2
                        endp_dets += dets
                        endp_DG += DG2
                        endp_gts += gts

                    if gt_avail and eval_semantic:
                        sem_acc, sem_recall, sem_f1, sem_TPs, sem_pts, sem_DGs, sem_gt_pts = eval_metric_line_segmentor(
                            lane_maps['semantic_line'][batch_idx], semantic_map_gt[batch_idx, ...], \
                            bi_seg=False, semantics=2, buff=self.cfg.validate_buffer)
                        batch_f1 = (batch_f1 * batch_idx + b_f1) / (batch_idx + 1)
                        semantic_TP += sem_TPs
                        semantic_dets += sem_pts
                        semantic_DG += sem_DGs
                        semantic_gts += sem_gt_pts
                    
                    # f1_ave = (f1_ave * i + batch_f1) / (i + 1)

                ########################################################################################################
                    ### begin: qualitative evaluation ###
                    if mode_view: 
                        pred_maps = output['pred_maps']
                        source_img = pred_maps['source_img_gray'][batch_idx]                      
                        pred_bi_seg_on_img = pred_maps['pred_bi_seg_on_image'][batch_idx]
                        pred_offset_lane_on_img = pred_maps['pred_offset_lanes_on_image'][batch_idx]
                        
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_source.png'),
                                source_img)
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_offset.png'),
                                pred_offset_lane_on_img)
                        cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_seg.png'),
                                pred_bi_seg_on_img)
                        
                        if gt_avail:
                            gt_on_source_img = pred_maps['gt_on_img'][batch_idx]
                            cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_gt.png'),
                                gt_on_source_img)
                            
                        if self.cfg.view_detail:
                            pred_org_lane_on_img = pred_maps['pred_org_lanes_on_image'][batch_idx]
                            pred_smooth_lane_on_img = pred_maps['pred_smooth_lanes_on_image'][batch_idx]
                            pred_exp_lane_on_img = pred_maps['pred_exp_lanes_on_image'][batch_idx]
                            # cv2.imshow('pred_lanes', pred_lane_on_image)
                            # cv2.waitKey(0)
                            cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_org.png'),
                                    pred_org_lane_on_img)
                            cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_smooth.png'),
                                    pred_smooth_lane_on_img)
                            cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_exp.png'),
                                    pred_exp_lane_on_img)
                    if write_lane_vertex:
                        # TODO: write out the vertex per batch in json file
                        pred_maps = output['pred_maps']
                        pred_smooth_lane_vertex = pred_maps['pred_smooth_lane_vertex'][batch_idx]
                        pred_lane_path = os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '.json')
                        save_lane_seq_2d(pred_smooth_lane_vertex, pred_lane_path, with_pervertex_semantics=True)
                ### end: qualitative evaluation ###
            self.val_bar.update(1)


        coor_pre = 0.
        coor_rec = 0.
        coor_f1s = 0.
        endp_pre = 0.
        endp_rec = 0.
        endp_f1s = 0.
        semantic_pre = 0.
        semantic_rec = 0.
        semantic_f1s = 0.

        if gt_avail & eval_coor:
            coor_pre = coor_TP / (coor_segs + EPS)
            coor_rec = coor_DG / (coor_gts + EPS)
            if (coor_pre + coor_rec) > 0.:
                coor_f1s = 2. * coor_pre * coor_rec / (coor_pre + coor_rec)
        if gt_avail & eval_endp:
            endp_pre = endp_TP / (endp_dets + EPS)
            endp_rec = endp_DG / (endp_gts + EPS)
            if (endp_pre + endp_rec) > 0.:
                endp_f1s = 2. * endp_pre * endp_rec / (endp_pre + endp_rec)
        if gt_avail & eval_semantic:
            semantic_pre = semantic_TP / (semantic_dets + EPS)
            semantic_rec = semantic_DG / (semantic_gts + EPS)
            if (semantic_pre + semantic_rec) > 0.:
                semantic_f1s = 2. * semantic_pre * semantic_rec / (semantic_pre + semantic_rec)

        print(f'coordinate_prec={coor_pre}')
        print(f'coordinate_rec={coor_rec}')
        print(f'coordinate_f1={coor_f1s}')
        print(f'endpoint_prec={endp_pre}')
        print(f'endpoint_rec={endp_rec}')
        print(f'endpoint_f1={endp_f1s}')
        print(f'semantic_prec={semantic_pre}')
        print(f'semantic_rec={semantic_rec}')
        print(f'semantic_f1={semantic_f1s}')



    def infer_lane_segmentation(self, path_ckpt=None, mode_imshow=None, is_calc_f1=False, write_lane_vertex=False):
        self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)

        if path_ckpt:
            trained_model = torch.load(path_ckpt)
            self.net.load_state_dict(trained_model['net'], strict=True)
        # for segmentation
        coor_TP = 0
        coor_segs = 0
        coor_DG = 0
        coor_gts = 0
        coor_f1_av = 0.

        # for endpoints
        endp_TP = 0
        endp_DG = 0
        endp_dets = 0
        endp_gts = 0
        endp_f1_av = 0

        self.net.eval()
        for i, data in enumerate(self.val_loader):
            data = self.to_cuda(data)
            with torch.no_grad():
                if self.cfg.net.type == 'Segmentor':
                    result_infer = self.net(data)
                    seg_map_pred = result_infer['seg'].cpu().detach().numpy()
                    seg_map_gt = data['mask'].cpu().detach().numpy()
                    batch_f1 = 0.

                    endp_map_pred = result_infer['endp'].cpu().detach().numpy()
                    endp_map_gt = data['endp_map'].cpu().detach().numpy()
                    

                    for batch_idx in range(self.cfg.batch_size):
                        b_acc, b_recall, b_f1, TPs, seg_pts, DGs, gt_pts = eval_metric_line_segmentor(seg_map_pred[batch_idx, ...], seg_map_gt[batch_idx, ...], bi_seg=False, semantics=2, buff=10)
                        batch_f1 = (batch_f1*batch_idx + b_f1) / (batch_idx+1)
                        coor_TP += TPs
                        coor_segs += seg_pts
                        coor_DG += DGs
                        coor_gts += gt_pts

                        _, _, _, TP2, dets, DG2, gts = eval_metric_endp_detector(endp_map_pred[batch_idx, ...], endp_map_gt[batch_idx, ...], r_thre=10)
                        endp_TP += TP2
                        endp_dets += dets
                        endp_DG += DG2
                        endp_gts += gts

                        if mode_imshow == 'all':
                            seg_endp_map_pred = result_infer['pred_display'][batch_idx]
                            cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_preseg.png'),
                                    seg_endp_map_pred)

                
        coor_con_pre = coor_TP / (coor_segs + EPS)
        coor_con_rec = coor_DG / (coor_gts + EPS)
        endp_con_pre = endp_TP / (endp_dets + EPS)
        endp_con_rec = endp_DG / (endp_gts + EPS)

        coor_con_f1s = 0.
        endp_conf_f1s = 0.
        if ((coor_con_rec + coor_con_pre) > 0):
            coor_con_f1s = 2. * coor_con_rec * coor_con_pre / (coor_con_rec + coor_con_pre)
        if ((endp_con_rec + endp_con_pre) > 0):
            endp_conf_f1s = 2.0 * endp_con_rec * endp_con_pre / (endp_con_rec + endp_con_pre)


        print(f'coor_conf_prec={coor_con_pre}')
        print(f'coor_conf_rec={coor_con_rec}')
        print(f'coor_conf_f1={coor_con_f1s}')
        print(f'endp_conf_prec={endp_con_pre}')
        print(f'endp_conf_rec={endp_con_rec}')
        print(f'endp_conf_f1={endp_conf_f1s}')

    def infer_lane_geometry_segmentation_segmentor(self, 
                                                   path_ckpt=None, 
                                                   mode_view=False, 
                                                   write_lane_vertex=False):
        self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)

        if path_ckpt:
            trained_model = torch.load(path_ckpt)
            self.net.load_state_dict(trained_model['net'], strict=True)
        # for geometry
        coor_TP = 0
        coor_pt = 0
        coor_DG = 0
        coor_gt_pt = 0
        coor_f1_av = 0.


        # for semantics
        semantic_TP = 0
        semantic_DG = 0
        semantic_dets = 0
        semantic_gts = 0


        self.net.eval()
        self.val_bar = tqdm(total=len(self.val_loader), desc='val', position=2)
        for i, data in enumerate(self.val_loader):
            data = self.to_cuda(data)
            with torch.no_grad():
                if self.cfg.net.type == 'Segmentor':
                    result_infer = self.net(data)
                    seg_map_pred = result_infer['seg'].cpu().detach().numpy()
                    seg_map_gt = data['mask'].cpu().detach().numpy()
                    batch_f1 = 0.

                    endp_map_pred = result_infer['endp'].cpu().detach().numpy()
                    endp_map_gt = data['endp_map'].cpu().detach().numpy()
                    

                    for batch_idx in range(self.cfg.batch_size):
                        sem_acc, sem_recall, sem_f1, sem_TPs, sem_pts, sem_DGs, sem_gt_pts = eval_metric_line_segmentor(
                                seg_map_pred[batch_idx], seg_map_gt[batch_idx, ...], \
                                bi_seg=False, semantics=2, buff=self.cfg.validate_buffer)
                            
                        semantic_TP += sem_TPs
                        semantic_dets += sem_pts
                        semantic_DG += sem_DGs
                        semantic_gts += sem_gt_pts

                        sem_acc, sem_recall, sem_f1, c_TPs, c_pts, c_DGs, c_gt_pts = eval_metric_line_segmentor(
                                seg_map_pred[batch_idx], seg_map_gt[batch_idx, ...], \
                                bi_seg=True, semantics=1, buff=self.cfg.validate_buffer)
                        coor_TP += c_TPs
                        coor_pt += c_pts
                        coor_DG += c_DGs
                        coor_gt_pt += c_gt_pts

                        # _, _, _, TP2, dets, DG2, gts = eval_metric_endp_detector(endp_map_pred[batch_idx, ...], endp_map_gt[batch_idx, ...], r_thre=10)
                        # endp_TP += TP2
                        # endp_dets += dets
                        # endp_DG += DG2
                        # endp_gts += gts

                        if mode_view:
                            seg_endp_map_pred = result_infer['pred_display'][batch_idx]
                            seg_skeleton_map_pred = result_infer['pred_skeleton_display'][batch_idx]
                            cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_segmentor.png'),
                                    seg_endp_map_pred)
                            cv2.imwrite(os.path.join(self.cfg.work_dirs, data['image_name'][batch_idx] + '_seg_skeleton.png'),
                                    seg_skeleton_map_pred)                           
            self.val_bar.update(1)

                
        coor_con_pre = coor_TP / (coor_pt + EPS)
        coor_con_rec = coor_DG / (coor_gt_pt + EPS)
        sem_con_pre = semantic_TP / (semantic_dets + EPS)
        sem_con_rec = semantic_DG / (semantic_gts + EPS)

        coor_con_f1s = 0.
        sem_con_f1s = 0.
        if ((coor_con_rec + coor_con_pre) > 0):
            coor_con_f1s = 2. * coor_con_rec * coor_con_pre / (coor_con_rec + coor_con_pre)
        if ((sem_con_rec + sem_con_pre) > 0):
            sem_con_f1s = 2.0 * sem_con_rec * sem_con_pre / (sem_con_rec + sem_con_pre)


        print(f'coor_conf_prec={coor_con_pre}')
        print(f'coor_conf_rec={coor_con_rec}')
        print(f'coor_conf_f1={coor_con_f1s}')
        print(f'sem_conf_prec={sem_con_pre}')
        print(f'sem_conf_rec={sem_con_rec}')
        print(f'sem_conf_f1={sem_con_f1s}')
