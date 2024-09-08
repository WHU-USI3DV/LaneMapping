import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import torchvision
from scipy import cluster
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import skimage
from skimage.morphology import skeletonize

from baseline.models.registry import PCENCODER
from baseline.utils.vis_utils import get_bi_seg_on_image, get_endp_on_raw_image
from baseline.utils.train_sample_utils import get_endpoint_maps_per_batch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

@PCENCODER.register_module
class PostProjector(nn.Module):
    def __init__(self,
                resnet='resnet50',
                pretrained=False,
                replace_stride_with_dilation=[False, True, False],
                out_conv=True,
                in_channels=[64, 128, 256, -1],
                cfg=None):
        super(PostProjector, self).__init__()
        self.cfg = cfg
        self.resnet = ResNetWrapper(
            resnet=resnet,
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            out_conv=out_conv,
            in_channels=in_channels,
            cfg=cfg
        )

    def forward(self, sample):
        proj = sample['proj']
        out = self.resnet(proj)

        return out

# projector & feature extractor
@PCENCODER.register_module
class PostProjector2(nn.Module):
    def __init__(self,
                resnet='resnet50',
                pretrained=False,
                replace_stride_with_dilation=[False, True, False],
                out_conv=True,
                in_channels=[64, 128, 256, -1],
                cfg=None):
        super(PostProjector2, self).__init__()
        self.cfg = cfg
        self.fpn = FPNWrapper(
            resnet=resnet,
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            out_conv=out_conv,
            in_channels=in_channels,
            cfg=cfg
        )



    def forward(self, sample):
        proj = sample['proj']
        fea_downsample, fea_up, out_binary_seg, out_endp_seg = self.fpn(proj)
        return fea_downsample, fea_up, out_binary_seg, out_endp_seg

    def loss(self, pred, gt_data):
        EPS = 1E-6
        loss = dict()
        # for semantic segmentation
        seg_loss = F.cross_entropy(pred['seg'], gt_data['mask'].type(torch.int64), reduction='sum')

        # for endpoints estimation
        lb_lc_endp = gt_data['endp_map']
        endp_exist = torch.where(torch.sum(lb_lc_endp, dim=(1, 2)) > 1.)  # batch_id

        endp_weight = lb_lc_endp.clone().detach()
        endp_weight[torch.where(endp_weight > EPS)] *= 10
        endp_weight[torch.where(endp_weight < EPS)] = 0.1
        lb_lc_endp[torch.where(lb_lc_endp > EPS)] = 1.
        lb_lc_endp[torch.where(lb_lc_endp < EPS)] = 0.
        endp_loss_none = torchvision.ops.sigmoid_focal_loss(pred['endp'][:, 0, :, :][endp_exist], lb_lc_endp[endp_exist], reduction='none')
        endp_loss = torch.sum(endp_weight[endp_exist] * endp_loss_none)
        
        del endp_loss_none, endp_weight

        b_size, _, f_h, f_w = pred['seg'].shape
        seg_loss /= (b_size * f_h * f_w)
        endp_loss = 50. * endp_loss / (f_h * f_w)
        loss['loss'] = seg_loss + endp_loss
        loss['loss_stats'] = {'seg_loss':seg_loss, 'endp_loss':endp_loss}
        return loss


    '''
    @param:display, if the display is True, then overlay the infered results to the raw images
    '''
    def infer_validate(self, preds, seg_thre=None, endp_thre=None, display=None):
        bi_seg = preds['seg']
        endp_result = preds['endp']
        # segmentation:
        if seg_thre is None:
            semantic_seg = bi_seg.argmax(1).detach().cpu()
        else:
            semantic_seg = torch.zeros((bi_seg.shape[0], bi_seg.shape[2], bi_seg.shape[3]))
            if bi_seg.shape[1] == 2:
                semantic_seg[torch.where(bi_seg[:, 1, :, :] > seg_thre)] = 1
            if bi_seg.shape[1] == 3:
                semantic_seg[torch.where((bi_seg[:, 1, :, :] > bi_seg[:, 2, :, :]) & (bi_seg[:, 1, :, :] > seg_thre))] = 1
                semantic_seg[torch.where((bi_seg[:, 2, :, :] > bi_seg[:, 1, :, :]) & (bi_seg[:, 2, :, :] > seg_thre))] = 2
        
        # endpoints
        ###
        # for endpoints: BEGIN
        ###
        b_size, _, img_h, img_w = endp_result.shape
        arr_endp = torch.zeros((b_size, img_h, img_w))
        clip_w = 20
        if endp_thre is None:
            endp_thre = 0.08
        # print("org h - w: ", org_img_h, org_img_w)
        # print("endp_est shape: ", out['endp_est'].shape)
        for idx_b in range(b_size):
            # temp_endp_score = out[f'endpoint'][idx_b, :, :]
            # temp_endp_score = torch.squeeze(out[f'endpoint'][idx_b, 0, clip_w:(img_h-clip_w), clip_w:(img_w-clip_w)])
            temp_endp_score = torch.squeeze(endp_result[idx_b, 0, clip_w:(img_h - clip_w), clip_w:(img_w - clip_w)])
            temp_endp_score = torch.sigmoid(temp_endp_score)
            temp_endp_score_flat = temp_endp_score.flatten()
            # Whether we need mask here to choose the endpoints in the area where temp_exist is True
            # temp_endp_score_flat = temp_endp_score_flat * (arr_endp_mask.view(-1))
            sorted_endp_score_flat = torch.argsort(temp_endp_score_flat, descending=True)

            # BEGIN: if we get the end points through the clustering
            local_endp_topK = 6
            loop_flag = True
            while loop_flag:
                topk_index = sorted_endp_score_flat[:local_endp_topK]  # we need topK
                topk_score = temp_endp_score_flat[topk_index]
                topk_h, topk_w = topk_index // (img_w - 2 * clip_w), topk_index % (img_w - 2 * clip_w)
            
                # add clustering method and select 2 clustering centers
                topk_h, topk_w = self.cluster_select_topK_pts(topk_h.cpu().detach(), topk_w.cpu().detach(), cluster_r=20, select_K=8)
                if (len(topk_h) > 4) or local_endp_topK > 100:
                    loop_flag = False
                else:
                    local_endp_topK += 10
                    # print("topk_idx: ", topk_index)
            # END: if we get the end points through the clustering
            # BEGIN: NO CLUSTERING
            # select_K = min(int((temp_endp_score_flat >endp_thre).sum().item()), local_endp_topK)
            # topk_index = sorted_endp_score_flat[:select_K]  # we need topK
            # topk_index_score = temp_endp_score_flat[select_K]
            # topk_h, topk_w = topk_index // (img_w - 2*clip_w), topk_index % (img_w - 2*clip_w)
            # END: NO CLUSTERING

            topk_h += clip_w
            topk_w += clip_w
            arr_endp[idx_b, topk_h.long(), topk_w.long()] = 1
        ###
        # for endpoints: END
        ###
        result_infer = dict()
        result_infer['seg'] = semantic_seg
        result_infer['endp'] = arr_endp

        return result_infer


    def get_bi_seg_maps(self, semantic_seg, batch, seg_thre=None):
        # overlay the segmentation results on the source image
        img_segmented = []
        for batch_idx in range(semantic_seg.shape[0]):
            raw_image = batch['proj'][batch_idx].cpu().numpy()
            raw_image_gray = cv2.cvtColor(raw_image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            raw_image_gray = cv2.cvtColor(raw_image_gray, cv2.COLOR_GRAY2RGB)
            raw_image_gray *= 255
            for semantic_id in [1, 2]:
                bi_seg_coors = np.where(semantic_seg == semantic_id)
                bi_seg_coors = np.array(bi_seg_coors)

                # draw on image
                if bi_seg_coors.shape[1] > 0:
                    raw_image_gray = get_bi_seg_on_image(bi_seg_coors, raw_image_gray, semantic_id=semantic_id)
            img_segmented.append(raw_image_gray)
        return img_segmented

    '''
    overlay the predicted endp on the original images
    '''
    def get_pred_endp_maps(self, endp_pred, batch):
        # overlay the predicted endpoints on the source image
        img_with_endps = []
        for batch_idx in range(endp_pred.shape[0]):
            raw_image = batch['proj'][batch_idx].cpu().numpy()
            raw_image_gray = cv2.cvtColor(raw_image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            raw_image_gray = cv2.cvtColor(raw_image_gray, cv2.COLOR_GRAY2RGB)
            raw_image_gray *= 255

            endp_coors = np.where(endp_pred[batch_idx] == 1.)
            raw_image_gray = get_bi_seg_on_image(endp_coors, raw_image_gray)
            img_with_endps.append(raw_image_gray)
        return img_with_endps
        
    def get_pred_seg_endp_displays(self, preds, batch):
        # overlay the infered results on the source image
        semantic_seg = preds['seg'].cpu().numpy()
        endp_pred = preds['endp'].cpu().numpy()
        
        img_display = []
        img_skeleton_display = []
        for batch_idx in range(semantic_seg.shape[0]):
            raw_image = batch['proj'][batch_idx].cpu().numpy()
            raw_image_gray = cv2.cvtColor(raw_image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            raw_image_gray = cv2.cvtColor(raw_image_gray, cv2.COLOR_GRAY2RGB)
            raw_image_gray *= 255
            raw_image_gray_skeleton = raw_image_gray.copy()
            for semantic_id in [1, 2]:
                bi_seg_coors = np.where(semantic_seg[batch_idx] == semantic_id)
                bi_seg_coors = np.array(bi_seg_coors)

                # skeletonize:
                before_skel = np.zeros_like(semantic_seg[batch_idx])
                before_skel[np.where(semantic_seg[batch_idx]==semantic_id)] = 1
                skel = skeletonize(before_skel.astype(np.int8), method='lee')
                kernel = skimage.morphology.rectangle(ncols=3, nrows=1)
                skel = skimage.morphology.dilation(skel, kernel)
                skel_coors = np.where(skel > 0)


                # draw on image
                if bi_seg_coors.shape[1] > 0:
                    raw_image_gray = get_bi_seg_on_image(bi_seg_coors, raw_image_gray, semantic_id=semantic_id)
                    raw_image_gray_skeleton = get_bi_seg_on_image(skel_coors, raw_image_gray_skeleton, semantic_id=semantic_id)
            # for endps:
            # endp_coors = np.where(endp_pred[batch_idx] == 1.)
            # endp_coors_array = np.concatenate((endp_coors[0], endp_coors[1]), axis=1)
            # raw_image_gray = get_endp_on_raw_image(endp_coors_array, raw_image_gray)
            
            img_display.append(raw_image_gray)
            img_skeleton_display.append(raw_image_gray_skeleton)
        result = dict()
        result['pred_display'] = img_display
        result['pred_skeleton_display'] = img_skeleton_display
        return result

    def cluster_select_topK_pts(self, pts_h, pts_w, cluster_r = 4, select_K=2):
        X = np.concatenate((pts_h.reshape((len(pts_h), 1)), pts_w.reshape((len(pts_w), 1))), axis=1)
        # print("X.shape", X.shape)
        clustering = DBSCAN(eps=cluster_r, min_samples=1, metric="euclidean").fit(X)
        labels_idx = clustering.labels_
        cluster_labels, cluster_sizes = np.unique(labels_idx,return_counts=True)
        centroid_labels = np.zeros((len(cluster_labels), X.shape[1]))
        nearest_sample_2_centroid = np.zeros_like(centroid_labels)

        for id, label in enumerate(cluster_labels):
            label_id_values = X[np.where(labels_idx == label)[0], :]
            label_center = np.mean(label_id_values, axis=0)
            centroid_labels[id, :] = label_center

            knearest = NearestNeighbors().fit(label_id_values)
            n_dist, n_idx = knearest.kneighbors([label_center], n_neighbors=1)
            nearest_sample_2_centroid[id, :] = label_id_values[n_idx[0], :]
        cluster_size_sorted = np.argsort(cluster_sizes)
        # choose the nearest pt or cluster center?
        center_h = [nearest_sample_2_centroid[k][0] for k in cluster_size_sorted]
        center_w = [nearest_sample_2_centroid[k][1] for k in cluster_size_sorted]

        return torch.tensor(center_h), torch.tensor(center_w)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError(
        #         "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetWrapper(nn.Module):

    def __init__(self, 
                resnet = 'resnet18',
                pretrained=True,
                replace_stride_with_dilation=[False, False, False],
                out_conv=False,
                fea_stride=8,
                out_channel=128,
                in_channels=[64, 128, 256, 512],
                cfg=None):
        super(ResNetWrapper, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels 

        self.model = eval(resnet)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation, in_channels=self.in_channels)
        self.out = None
        if out_conv:
            out_channel = 512
            for chan in reversed(self.in_channels):
                if chan < 0: continue
                out_channel = chan
                break
            self.out = conv1x1(
                out_channel * self.model.expansion, cfg.featuremap_out_channel)

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x = self.out(x)
        return x

class FPNWrapper(nn.Module):
    def __init__(self,
                resnet = 'resnet18',
                block = BasicBlock,
                layers=[3, 4, 6, 3],
                pretrained=True,
                replace_stride_with_dilation=[False, False, False],
                out_conv=False,
                fea_stride=8,
                out_channel=128,
                in_channels=[64, 128, 256, 512],
                cfg=None,
                norm_layer=None):
        super(FPNWrapper, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels


        # ResNet Model:
        self.model_buttomup = eval(resnet)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation, in_channels=self.in_channels)

        # ResNet Details
        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        self.dilation = 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        # Layer-0
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, in_channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, in_channels[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        if in_channels[2] > 0:
            self.layer3 = self._make_layer(block, in_channels[2], layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        if in_channels[3] > 0:
            self.layer4 = self._make_layer(block, in_channels[3], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        self.expansion = block.expansion
        #!!!IMPORTANT!!! store output feature for Vision transformer
        self.out = None
        if out_conv:
            out_channel = 512
            for chan in reversed(self.in_channels):
                if chan < 0: continue
                out_channel = chan
                break
            self.out = conv1x1(
                out_channel * self.expansion, cfg.featuremap_out_channel)

        # upsample and connect
        # Top layer
        self.toplayer = nn.Conv2d(out_channel * self.expansion, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        if in_channels[3] > 0:
            self.latlayer1 = nn.Conv2d(self.in_channels[2] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d(self.in_channels[1] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
            self.latlayer3 = nn.Conv2d(self.in_channels[0] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
        elif in_channels[2] > 0:
            self.latlayer1 = nn.Conv2d(self.in_channels[1] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d(self.in_channels[0] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
        else:
            self.latlayer1 = nn.Conv2d(self.in_channels[0] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)

        self.semantic_branch = nn.Conv2d(self.inplanes * self.expansion, int(self.inplanes * self.expansion * 0.5), kernel_size=3, stride=1, padding=1)
        self.semantic_branch2 = nn.Conv2d(self.inplanes * self.expansion, int(self.inplanes * self.expansion * 0.5),
                                         kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)

        self.feature_layer = nn.Conv2d(int(self.inplanes * self.expansion * 0.5), 8, kernel_size=1, stride=1, padding=0)
        self.output_layer_binary_seg = nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0)
        self.output_layer_endp = nn.Conv2d(int(self.inplanes * self.expansion * 0.5), 1, kernel_size=1, stride=1, padding=0)
        self.gn11 = nn.GroupNorm(int(self.inplanes * self.expansion * 0.5), int(self.inplanes * self.expansion * 0.5))
        self.gn12 = nn.GroupNorm(self.inplanes * self.expansion, self.inplanes * self.expansion)
        self.gn21 = nn.GroupNorm(int(self.inplanes * self.expansion * 0.5), int(self.inplanes * self.expansion * 0.5))
        self.gn22 = nn.GroupNorm(self.inplanes * self.expansion, self.inplanes * self.expansion)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        _, _, x_h, x_w = x.shape
        # Bottom-up
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)

        if self.in_channels[2] > 0:
            c4 = self.layer3(c3)
            if self.out:
                fea_downsample = self.out(c4)
        if self.in_channels[3] > 0:
            c5 = self.layer4(c4)
            if self.out:
                fea_downsample = self.out(c5)

        # print(f'c1.shape={c1.shape}, c2.shape={c2.shape}, c3.shape={c3.shape}, c4.shape={c4.shape}')
        # c1.shape = torch.Size([4, 64, 288, 288]), c2.shape = torch.Size([4, 64, 288, 288]), c3.shape = torch.Size(
        #     [4, 128, 144, 144]), c4.shape = torch.Size([4, 256, 144, 144])

        # Top-down
        if self.in_channels[3] > 0:
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p3 = self._upsample_add(p4, self.latlayer2(c3))
            p2 = self._upsample_add(p3, self.latlayer3(c2))
        if self.in_channels[2] > 0:
            p4 = self.toplayer(c4)
            p3 = self._upsample_add(p4, self.latlayer1(c3))
            p2 = self._upsample_add(p3, self.latlayer2(c2))

        # Smooth
        if self.in_channels[2] > 0:
            p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # print(f'p2.shape={p2.shape},p3.shape={p3.shape},p4.shape={p4.shape}')
        # p2.shape = torch.Size([4, 256, 288, 288]), p3.shape = torch.Size([4, 256, 144, 144]), p4.shape = torch.Size(
        #     [4, 256, 144, 144])
        # Semantic
        _, _, h, w = p2.size()
        if self.in_channels[3] > 0:
            # 256->256
            s5 = self._upsample(F.relu(self.gn12(self.conv2(p5))), h, w)
            # 256->256
            s5 = self._upsample(F.relu(self.gn12(self.conv2(s5))), h, w)
            # 256->128
            s5 = self._upsample(F.relu(self.gn11(self.semantic_branch(s5))), h, w)
        if self.in_channels[2] > 0:
            # 256->256
            s4 = self._upsample(F.relu(self.gn12(self.conv2(p4))), h, w)
            # 256->128
            s4 = self._upsample(F.relu(self.gn11(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn11(self.semantic_branch(p3))), h, w)
        s2 = F.relu(self.gn11(self.semantic_branch(p2)))
        # print(f's2.shape={s2.shape},s3.shape={s3.shape},s4.shape={s4.shape}, s234.shape={(s2+s3+s4).shape}')
        # s2.shape = torch.Size([4, 128, 288, 288]), s3.shape = torch.Size([4, 128, 288, 288]), s4.shape = torch.Size(
        #     [4, 128, 288, 288]), s234.shape = torch.Size([4, 128, 288, 288]), fea_upsample.shape = torch.Size([4, 8, 288, 288])
        if self.in_channels[3] > 0:
            fea_upsample = self.feature_layer(s2 + s3 + s4 + s5)
        elif self.in_channels[2] > 0:
            fea_upsample = self.feature_layer(s2 + s3 + s4)
        else:
            fea_upsample = self.conv3(s2 + s3)
        out_binary_seg = self._upsample(self.output_layer_binary_seg(F.relu(fea_upsample)), x_h, x_w)

        if self.in_channels[3] > 0:
            s5 = self._upsample(F.relu(self.gn22(self.conv3(p5))), h, w)
            # 256->256
            s5 = self._upsample(F.relu(self.gn22(self.conv3(s5))), h, w)
            # 256->128
            s5 = self._upsample(F.relu(self.gn21(self.semantic_branch2(s5))), h, w)
        if self.in_channels[2] > 0:
            # 256->256
            s4 = self._upsample(F.relu(self.gn22(self.conv3(p4))), h, w)
            # 256->128
            s4 = self._upsample(F.relu(self.gn21(self.semantic_branch2(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn21(self.semantic_branch2(p3))), h, w)
        s2 = F.relu(self.gn21(self.semantic_branch2(p2)))
        if self.in_channels[3] > 0:
            out_endp = self._upsample(self.output_layer_endp(s2 + s3 + s4 + s5), x_h, x_w)
        elif self.in_channels[2] > 0:
            out_endp = self._upsample(self.output_layer_endp(s2 + s3 + s4), x_h, x_w)
        else:
            out_endp = self._upsample(self.output_layer_endp(s2 + s3), x_h, x_w)

        return fea_downsample, fea_upsample, out_binary_seg, out_endp


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, in_channels=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = in_channels
        self.layer1 = self._make_layer(block, in_channels[0], layers[0])
        self.layer2 = self._make_layer(block, in_channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        if in_channels[2] > 0:
            self.layer3 = self._make_layer(block, in_channels[2], layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        if in_channels[3] > 0:
            self.layer4 = self._make_layer(block, in_channels[3], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        self.expansion = block.expansion

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.in_channels[2] > 0:
            x = self.layer3(x)
        if self.in_channels[3] > 0:
            x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

'''
First stage: encode features from the input data
Second stage: decode as the nearest distance map & nearest lane id map
'''


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('pretrained model: ', model_urls[arch])
        # state_dict = torch.load(model_urls[arch])['net']
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
