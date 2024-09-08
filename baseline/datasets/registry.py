# Thanks to TuZheng (LaneDet) https://github.com/Turoad/lanedet

from baseline.utils import Registry, build_from_cfg

import torch
from functools import partial
import numpy as np
import random

from mmengine.dataset import pseudo_collate

DATASETS = Registry('datasets')
PROCESS = Registry('process')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return torch.nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_dataset(split_cfg, cfg):
    return build(split_cfg, DATASETS, default_args=dict(cfg=cfg))

def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(split_cfg, cfg, is_train=True):
    if is_train:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    dataset = build_dataset(split_cfg, cfg)
    if cfg.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        if is_train:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    init_fn = partial(
            worker_init_fn, seed=cfg.seed)


    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size = cfg.batch_size, sampler = sampler, #shuffle = shuffle,
            num_workers = cfg.workers, pin_memory = True, drop_last = drop_last,
            worker_init_fn=init_fn,
            # collate_fn=pseudo_collate  #if cfg.use_lidar == True   # to fit the las points: If you are loading las points, then uncomment this line.
            )


    return data_loader
