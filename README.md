<h2>
<a href="https://whu-usi3dv.github.io/LaneMapping/" target="_blank">A Benchmark Approach and Dataset for Large-scale Lane Mapping from MLS Point Clouds</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **A Benchmark Approach and Dataset for Large-scale Lane Mapping from MLS Point Clouds**<br/>
> [Xiaoxin Mi](https://mixiaoxin.github.io/), [Zhen Dong](https://dongzhenwhu.github.io/index.html), Zhipeng Cao, [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm),[Zhen Cao](https://github.com/a4152684), Chao Zheng, [Jantien Stoter](https://3d.bk.tudelft.nl/jstoter/), [Liangliang Nan](https://3d.bk.tudelft.nl/liangliang/)`<br/>`
> *International Journal of Applied Earth Observation and Geoinformation(JAG) 2024*`<br/>`
> [**Paper**](TODO:url) | [**Project-page**]() | [**Video**]()

## 🔭 Introduction

<p align="center" style="font-size:18px">
<strong>A Benchmark Approach and Dataset for Large-scale Lane Mapping from MLS Point Clouds</strong>
</p>
<img src="media/teaser.jpg" alt="Network" style="zoom:10%;">

<p align="justify">
<strong>Abstract:</strong> Accurate lane maps with semantics are crucial for various applications, such as high-definition maps (HD Maps), intelligent transportation systems (ITS), and digital twins. Manual annotation of lanes is labor-intensive and costly, prompting researchers to explore automatic lane extraction methods. 
This paper presents an end-to-end large-scale lane mapping method that considers both lane geometry and semantics.
This study represents lane markings as polylines with uniformly sampled points and associated semantics, allowing for adaptation to varying lane shapes. Additionally, we propose an end-to-end network to extract lane polylines from mobile laser scanning (MLS) data, enabling the inference of vectorized lane instances without complex post-processing. The network consists of three components: a feature encoder, a column proposal generator, and a lane information decoder. 
The feature encoder encodes textual and structural information of lane markings to enhance the method's robustness to data imperfections, such as varying lane intensity, uneven point density, and occlusion-induced incomplete data. The column proposal generator generates regions of interest for the subsequent decoder. Leveraging the embedded multi-scale features from the feature encoder, the lane decoder effectively predicts lane polylines and their associated semantics without requiring step-by-step conditional inference.
Comprehensive experiments conducted on three lane datasets have demonstrated the performance of the proposed method, even in the presence of incomplete data and complex lane topology.
</p>

## 🆕 News

- 2024-09-10: [LaneMapping] code and dataset are publicly accessible! 🎉
- 2024-09-02: our paper is accepted for publication in International Journal of Applied Earth Observation and Geoinformation(JAG)! 🎉

## 💻 Requirements

The code has been trained on:

- Ubuntu 20.04
- CUDA 11.8 (Other versions should be okay.)
- Python 3.8
- Pytorch 2.1.0
- A GeForce RTX 4090.

## 🔧 Installation

#### a. Create a conda virtual environment and activate it.

```
conda create -n lanemap python=3.8 -y  
conda activate lanemap
```

### b. Install PyTorch and torchvision following the official instructions.

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### c. Install mmcv-full. (optional: to validate LidarEncoder: SparseConv)

follow the instructions here: https://mmdetection3d.readthedocs.io/en/latest/get_started.html

```
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
```

```
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
# "-b dev-1.x" means checkout to the `dev-1.x` branch.
cd mmdetection3d
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in edtiable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### d. install other third party libs

```
pip install einops==0.8.0
pip install timm
pip install laspy  # to load LiDAR points
pip install pytorch_warmup
```

## Train and Test

```
export PATH=$PATH:/path/to/your/lanemap/dir  # set the work path
```

## 💾 Datasets

We used WHU-Lane for training and three datasets for evaluation.

#### WHU-Lane

WHU-Lane data structure is as follows:

```
WHU-Lane
├── TrainValAll
│    ├── cropped_tiff
│    │    ├── 000000_0001.png
|    |    ├── 000000_0002.png
│    │    └── ...
│    ├── labels
│    │    ├── sparse_seq
|    |    ├── sparse_instance
|    |    ├── sparse_orient
|    |    ├── sparse_endp
│    │    └── sparse_semantic
│    └── ...
├── TestArea1
│    ├── cropped_tiff
│    │    ├── 000000_0001.png
|    |    ├── 000000_0002.png
│    │    └── ...
│    ├── labels
│    │    ├── sparse_seq
|    |    ├── sparse_instance
|    |    ├── sparse_orient
|    |    ├── sparse_endp
│    │    └── sparse_semantic
│    └── ...
└── TestArea2
     ├── cropped_tiff
     │    ├── 000000_0001.png
     |    ├── 000000_0002.png
     │    └── ...
     ├── labels
     │    ├── sparse_seq
     |    ├── sparse_instance
     |    ├── sparse_orient
     |    ├── sparse_endp
     │    └── sparse_semantic
     └── ...
```

## 🚅 Pretrained model (TODO)

You can download the pretrained model from [GoogleDrive](https://drive.google.com/drive/folders/1EmTFrqGnnh9a5ZsQ8ydSZC3PK-NeGDlX?usp=sharing), and put it in folder `pretrain/`.

## ⏳ Train (TODO)

To train SparseDC, you should prepare the dataset, and replace the [&#34;data_dir&#34;](/configs/paths/default.yaml) to your data path. Then, you use the follow command:

```bash
$ python train.py experiment=final_version         # for NYUDepth
$ python train.py experiment=final_version_kitti   # for KITTIDC
```

## ✏️ Test (TODO)

To eval SparseDC on three benchmarks, you can use the following commands:

```bash
$ ./eval_nyu.sh final_version final_version pretrain/nyu.ckpt
$ ./eval_kitti.sh final_version_kitti final_version_kitti_test pretrain/kitti.ckpt
$ ./eval_sunrgbd.sh final_version final_version pretrain/nyu.ckpt
```

## 💡 Citation (TODO)

If you find this repo helpful, please give us a 😍 star 😍.
Please consider citing SparseDC if this program benefits your project

```Tex
@article{
}
```

## 🔗 Related Projects

We sincerely thank the excellent projects:

- [KLane](https://github.com/kaist-avelab/K-Lane.git) for code framework;
- [MapTR](https://github.com/hustvl/MapTR) for lidar encoder;
- [FreeReg](https://github.com/WHU-USI3DV/FreeReg) for readme template;
