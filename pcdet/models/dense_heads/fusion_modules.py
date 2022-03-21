import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from pcdet.models.dense_heads.utils import Sequential, NaiveSepHead
from pcdet.ops.dcn import DeformConv
from torch.nn.modules.utils import _pair, _single
import itertools


def fusion_mul(map, feat):
    """
    Args:
        map: tensor size = bs, c1, h, w
        feat: tensor size = bs, c2, h, w

    Returns:
        weighted_feat size = (bs, c1 x c2, h, w)
    """
    bs, c1, h, w = map.size()
    c2 = feat.size(1)
    rlt = map.view(bs, c1, 1, h, w) * feat.view(bs, 1, c2, h, w)
    rlt = rlt.view(bs, c1 * c2, h, w)
    return rlt


class NaiveFusion(nn.Module):
    def __init__(self, feat_channels, out_channels, num_classes):
        super().__init__()
        self.total_channels = feat_channels * num_classes * (2 + 4)
        self.post = nn.Sequential(
            nn.Conv2d(self.total_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, center_map, center_feat, corner_map, corner_feat, foreground_feat, foreground_map):
        fg_map = foreground_map.clone().detach().sigmoid()
        ctr_map = center_map.clone().detach().sigmoid()
        cnr_map = corner_map.clone().detach().sigmoid()

        weighted_fg_map = fusion_mul(fg_map, foreground_feat)
        weighted_ctr_map = fusion_mul(ctr_map, center_feat)
        weighted_cnr_map = fusion_mul(cnr_map, corner_feat)

        feat = torch.cat([weighted_fg_map, weighted_ctr_map, weighted_cnr_map], dim=1)
        rlt = self.post(feat)

        return rlt


class FeatureAdaption(nn.Module):
    def __init__(self, in_ch, off_in_ch, out_ch, kernel_size=3):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(off_in_ch, offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def init_weights(self):
        pass

    def forward(self, x, off):
        offset = self.conv_offset(off)
        rlt = self.relu(self.conv_adaption(x, offset))
        return rlt


class DCNFusion(nn.Module):
    def __init__(self, feat_channels, offset_ch, out_channels, num_classes):
        super().__init__()
        assert out_channels % 2 == 0
        base_ch = num_classes * feat_channels

        self.offset_prep = nn.Sequential(
            nn.Conv2d(offset_ch, base_ch, 1, bias=True),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        self.fg_fuse = FeatureAdaption(base_ch, base_ch, out_channels)
        self.cnr_fuse = FeatureAdaption(4 * base_ch, base_ch, out_channels)
        self.ctr_fuse = FeatureAdaption(base_ch, base_ch, out_channels)

    def forward(self, orig_feat, center_map, center_feat, corner_map, corner_feat, foreground_feat, foreground_map):
        fg_map = foreground_map.clone().detach().sigmoid()
        ctr_map = center_map.clone().detach().sigmoid()
        cnr_map = corner_map.clone().detach().sigmoid()

        weighted_fg_map = fusion_mul(fg_map, foreground_feat)
        weighted_ctr_map = fusion_mul(ctr_map, center_feat)
        weighted_cnr_map = fusion_mul(cnr_map, corner_feat)

        offset_map = self.offset_prep(orig_feat)

        out_fg = self.fg_fuse(weighted_fg_map, offset_map)
        out_ctr = self.ctr_fuse(weighted_ctr_map, offset_map)
        out_cnr = self.cnr_fuse(weighted_cnr_map, offset_map)

        rlt = torch.cat([out_fg, out_ctr, out_cnr], dim=1)
        return rlt


class DCNFusionV2(nn.Module):
    def __init__(self, feat_ch, offset_ch, out_ch, **kwargs):
        super().__init__()

        self.offset_prep = nn.Sequential(
            nn.Conv2d(offset_ch, feat_ch, 1, bias=True),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True)
        )

        self.fg_fuse = FeatureAdaption(feat_ch, feat_ch, out_ch)
        self.cnr_fuse = FeatureAdaption(feat_ch, feat_ch, out_ch)
        self.ctr_fuse = FeatureAdaption(feat_ch, feat_ch, out_ch)

        self.base_queries = nn.Sequential(
            nn.Conv2d(offset_ch, feat_ch // 4, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(feat_ch // 4),
            nn.ReLU(),
        )

        self.center_key = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 4, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(feat_ch // 4),
            nn.ReLU(),
        )
        self.center_value = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(),
        )
        self.corner_key = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 4, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(feat_ch // 4),
            nn.ReLU(),
        )
        self.corner_value = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(),
        )
        self.fg_key = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 4, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(feat_ch // 4),
            nn.ReLU(),
        )
        self.fg_value = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(),
        )

    def forward(self, orig_feat, center_feat, corner_feat, foreground_feat, **kwargs):
        offset_map = self.offset_prep(orig_feat)

        fg_feat = self.fg_fuse(foreground_feat, offset_map)
        ctr_feat = self.ctr_fuse(center_feat, offset_map)
        cnr_feat = self.cnr_fuse(corner_feat, offset_map)

        base_q = self.base_queries(orig_feat)

        center_k = self.center_key(ctr_feat)
        center_v = self.center_value(ctr_feat)
        center_a = (base_q * center_k).sum(1, keepdim=True)  # (B, 1, W, H)

        corner_k = self.corner_key(cnr_feat)
        corner_v = self.corner_value(cnr_feat)
        corner_a = (base_q * corner_k).sum(1, keepdim=True)  # (B, 1, W, H)

        fg_k = self.fg_key(fg_feat)
        fg_v = self.fg_value(fg_feat)
        fg_a = (base_q * fg_k).sum(1, keepdim=True)  # (B, 1, W, H)

        attention_weights = F.softmax(torch.cat([center_a, corner_a, fg_a], dim=1), dim=1).unsqueeze(2)
        attention_feats = torch.stack([center_v, corner_v, fg_v], dim=1)  # (B, 3, C, H, W)
        rlt = (attention_weights * attention_feats).sum(1)  # (B, C, H, W)

        return rlt


class DCNFusionV3(nn.Module):
    def __init__(self, feat_ch, offset_ch, out_ch, **kwargs):
        super().__init__()

        self.offset_prep = nn.Sequential(
            nn.Conv2d(offset_ch, feat_ch, 1, bias=True),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True)
        )

        self.fg_fuse = FeatureAdaption(feat_ch, feat_ch, out_ch)
        self.cnr_fuse = FeatureAdaption(feat_ch, feat_ch, out_ch)
        self.ctr_fuse = FeatureAdaption(feat_ch, feat_ch, out_ch)

    def forward(self, orig_feat, center_feat, corner_feat, foreground_feat, **kwargs):
        offset_map = self.offset_prep(orig_feat)

        out_fg = self.fg_fuse(foreground_feat, offset_map)
        out_ctr = self.ctr_fuse(center_feat, offset_map)
        out_cnr = self.cnr_fuse(corner_feat, offset_map)

        return out_fg, out_ctr, out_cnr


class DCNFusionHead(nn.Module):
    def __init__(self, in_channels, num_cls, heads, dcn_version='V1', head_conv=64, final_kernel=1, bn=False,
                 init_bias=-2.19, **kwargs):
        super(DCNFusionHead, self).__init__()

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.center_head.bias.data.fill_(init_bias)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_conv, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.corner_head.bias.data.fill_(init_bias)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.fg_head.bias.data.fill_(init_bias)

        # other regression target
        self.task_head = NaiveSepHead(in_channels + head_conv, heads,
                                      head_ch=head_conv, bn=bn, final_kernel=final_kernel)

        if dcn_version == 'V1':
            self.fusion_module = DCNFusion(head_conv, in_channels, head_conv, num_cls)
        elif dcn_version == 'V2':
            self.fusion_module = DCNFusionV2(head_conv, in_channels, head_conv)

        self.fusion_head = Sequential(
            nn.Conv2d(in_channels + head_conv, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.fusion_head[-1].bias.data.fill_(init_bias)

    def forward(self, x):
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)
        fusion_feat = self.fusion_module(orig_feat=x,
                                         center_map=center_map,
                                         center_feat=center_feat,
                                         corner_map=corner_map,
                                         corner_feat=corner_feat,
                                         foreground_feat=fg_feat,
                                         foreground_map=fg_map)

        fused_feat = torch.cat([x, fusion_feat], dim=1)
        fusion_map = self.fusion_head(fused_feat)
        ret = self.task_head(fused_feat)

        ret['fusion_map'] = fusion_map
        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        return ret


class NaiveFusionHead(nn.Module):
    def __init__(self, in_channels, num_cls, heads, head_conv=64, final_kernel=1, bn=False, init_bias=-2.19, **kwargs):
        super(NaiveFusionHead, self).__init__()

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.center_head.bias.data.fill_(init_bias)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_conv, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.corner_head.bias.data.fill_(init_bias)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.fg_head.bias.data.fill_(init_bias)

        # other regression target
        self.task_head = NaiveSepHead(in_channels + head_conv, heads, head_ch=head_conv, bn=bn,
                                      final_kernel=final_kernel)

        self.fusion_module = NaiveFusion(head_conv, head_conv, num_cls)
        self.fusion_head = Sequential(
            nn.Conv2d(in_channels + head_conv, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.fusion_head[-1].bias.data.fill_(init_bias)

    def forward(self, x):
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(center_map, center_feat, corner_map, corner_feat, fg_feat, fg_map)
        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret = self.task_head(final_feat)
        fusion_map = self.fusion_head(final_feat)

        ret['fusion_map'] = fusion_map
        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        return ret