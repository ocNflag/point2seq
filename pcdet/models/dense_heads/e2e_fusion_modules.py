import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils import common_utils, box_utils

from pcdet.utils.common_utils import limit_period_torch
from pcdet.models.dense_heads.utils import FeatureAdaption, Sequential
from pcdet.ops.dcn import ModulatedDeformConv


class FeatureAdaption(nn.Module):
    def __init__(self, in_ch, off_in_ch, out_ch, kernel_size=3):
        super(FeatureAdaption, self).__init__()
        offset_ch = kernel_size * kernel_size * 3
        self.conv_offset = nn.Conv2d(off_in_ch, offset_ch, 1, bias=True)
        self.conv_adaption = ModulatedDeformConv(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
        self.init_offset()

    def init_offset(self):
        self.conv_offset.bias.data.zero_()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, off):
        out = self.conv_offset(off)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        rlt = self.relu(self.conv_adaption(x, offset, mask))
        return rlt


class DCNV2Fusion(nn.Module):
    def __init__(self, feat_ch, num_cls, offset_ch, out_ch, naive=False, **kwargs):
        super().__init__()
        self.naive = naive

        if self.naive:
            self.fg_fuse = nn.Conv2d(feat_ch + num_cls, out_ch, kernel_size=1)
            self.cnr_fuse = nn.Conv2d(feat_ch + (4 * num_cls), out_ch, kernel_size=1)
            self.ctr_fuse = nn.Conv2d(feat_ch + num_cls, out_ch, kernel_size=1)
        else:
            self.fg_fuse = FeatureAdaption(feat_ch + num_cls, offset_ch, out_ch)
            self.cnr_fuse = FeatureAdaption(feat_ch + (4 * num_cls), offset_ch, out_ch)
            self.ctr_fuse = FeatureAdaption(feat_ch + num_cls, offset_ch, out_ch)

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
            self.corner_key = nn.Sequential(
                nn.Conv2d(feat_ch, feat_ch // 4, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(feat_ch // 4),
                nn.ReLU(),
            )
            self.fg_key = nn.Sequential(
                nn.Conv2d(feat_ch, feat_ch // 4, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(feat_ch // 4),
                nn.ReLU(),
            )

    def forward(self, orig_feat, center_feat, corner_feat, foreground_feat, **kwargs):
        assert (orig_feat != orig_feat).sum().item() == 0
        if self.naive:
            ctr_feat = self.ctr_fuse(center_feat)
            cnr_feat = self.cnr_fuse(corner_feat)
            fg_feat = self.fg_fuse(foreground_feat)
            return ctr_feat + cnr_feat + fg_feat

        ctr_feat = self.ctr_fuse(center_feat, orig_feat)
        cnr_feat = self.cnr_fuse(corner_feat, orig_feat)
        fg_feat = self.fg_fuse(foreground_feat, orig_feat)

        base_q = self.base_queries(orig_feat)

        ctr_k = self.center_key(ctr_feat)
        ctr_a = (base_q * ctr_k).sum(1, keepdim=True)  # (B, 1, W, H)

        cnr_k = self.corner_key(cnr_feat)
        cnr_a = (base_q * cnr_k).sum(1, keepdim=True)  # (B, 1, W, H)

        fg_k = self.fg_key(fg_feat)
        fg_a = (base_q * fg_k).sum(1, keepdim=True)  # (B, 1, W, H)

        attention_weights = F.softmax(torch.cat([ctr_a, cnr_a, fg_a], dim=1), dim=1).unsqueeze(2)
        attention_feats = torch.stack([ctr_feat, cnr_feat, fg_feat], dim=1)  # (B, 3, C, H, W)
        rlt = (attention_weights * attention_feats).sum(1)  # (B, C, H, W)

        return rlt


class OneNetFusionHead(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']
        code_size = heads['code_size']
        if heads['encode_angle_by_sincos']:
            code_size += 1

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch)

        self.cls_head = Sequential(
            nn.Conv2d(in_channels + head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2),
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels + head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, code_size, kernel_size=ks, stride=1, padding=ks // 2),
        )

        self._reset_parameters()

        self.cls_head[-1].bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat
        ret['pred_logits'] = self.cls_head(final_feat)
        ret['pred_boxes'] = self.bbox_head(final_feat)

        return ret


class OneNetSeqFusionHead(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.wlh_box_scale_factor = heads.get('cls_box_scale_factor', 1.0)
        self.is_nusc = heads.get('is_nusc', False)
        self.naive = heads.get('naive', False)
        if self.is_nusc:
            theta_ch = 4
        else:
            theta_ch = 2

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, theta_ch, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        context['raw_xy'] = raw_xy
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape
        raw_xy = context['raw_xy']

        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = raw_xy.permute(0, 2, 3, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        context['flatten_xy'] = tmp_xy

        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']
        flatten_xy = context['flatten_xy']

        n, _, h, w = wlh.shape
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, flatten_xy, box_scale_ratio=self.wlh_box_scale_factor)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        xyz_feat = self.xyz_conv(final_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        theta_feat = torch.cat([final_feat, xyz_feat], dim=1)
        theta_feat = self.theta_conv(theta_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        wlh_feat = torch.cat([final_feat, theta_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)

        wlh = torch.clamp(wlh, min=-2, max=3)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        pred_feats = torch.cat([final_feat, wlh_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadDense(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)
        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (6 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (6 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        context['raw_xy'] = raw_xy
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape
        raw_xy = context['raw_xy']

        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = raw_xy.permute(0, 2, 3, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        context['flatten_xy'] = tmp_xy

        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']
        flatten_xy = context['flatten_xy']

        n, _, h, w = wlh.shape
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, flatten_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        xyz_feat = self.xyz_conv(final_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        theta_feat = torch.cat([final_feat, xyz_feat], dim=1)
        theta_feat = self.theta_conv(theta_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        wlh_feat = torch.cat([final_feat, xyz_feat, theta_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-2, max=3)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        pred_feats = torch.cat([final_feat, xyz_feat, wlh_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadTCS(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.naive = heads.get('naive', False)

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        n, _, h, w = xy.shape
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        context['flatten_xy'] = raw_xy.permute(0, 2, 3, 1).contiguous().view(n * h * w, 2)
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape

        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']
        flatten_xy = context['flatten_xy']

        n, _, h, w = wlh.shape
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, flatten_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        theta_feat = self.theta_conv(final_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        xyz_feat = torch.cat([final_feat, theta_feat], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        wlh_feat = torch.cat([final_feat, xyz_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-2, max=3)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        pred_feats = torch.cat([final_feat, wlh_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadCST(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.naive = heads.get('naive', False)

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        n, _, h, w = xy.shape
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        context['raw_xy'] = raw_xy
        tmp_xy = raw_xy.permute(0, 2, 3, 1).contiguous().view(n * h * w, 2)
        context['flatten_xy'] = tmp_xy
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape

        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w

        context['theta'] = theta
        flatten_xy = context['flatten_xy']

        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, flatten_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        flatten_xy = context['flatten_xy']

        n, _, h, w = wlh.shape
        theta = wlh.new_zeros(n * h * w)

        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, flatten_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        xyz_feat = self.xyz_conv(final_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        wlh_feat = torch.cat([final_feat, xyz_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-2, max=3)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        theta_feat = torch.cat([final_feat, wlh_feat], dim=1)
        theta_feat = self.theta_conv(theta_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        pred_feats = torch.cat([final_feat, theta_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadTSC(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.naive = heads.get('naive', False)

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        n, _, h, w = xy.shape
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape
        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']

        n, _, h, w = wlh.shape
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        theta_feat = self.theta_conv(final_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        wlh_feat = torch.cat([final_feat, theta_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-2, max=3)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        xyz_feat = torch.cat([final_feat, wlh_feat], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        pred_feats = torch.cat([final_feat, xyz_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadCLSCTS(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.wlh_box_scale_factor = heads.get('cls_box_scale_factor', 1.0)
        self.is_nusc = heads.get('is_nusc', False)
        self.naive = heads.get('naive', False)
        if self.is_nusc:
            theta_ch = 4
        else:
            theta_ch = 2

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, theta_ch, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        context['raw_xy'] = raw_xy
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape
        raw_xy = context['raw_xy']

        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = raw_xy.permute(0, 2, 3, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        context['flatten_xy'] = tmp_xy

        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']
        flatten_xy = context['flatten_xy']

        n, _, h, w = wlh.shape
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, flatten_xy, box_scale_ratio=self.wlh_box_scale_factor)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        pred_feats = self.cls_conv(final_feat)
        pred_logits = self.cls_head(pred_feats)

        xyz_feat = torch.cat([final_feat, pred_feats], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        theta_feat = torch.cat([final_feat, xyz_feat], dim=1)
        theta_feat = self.theta_conv(theta_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        wlh_feat = torch.cat([final_feat, theta_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)

        wlh = torch.clamp(wlh, min=-2, max=3)
        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadCLSTSC(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.naive = heads.get('naive', False)

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        n, _, h, w = xy.shape
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape
        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']

        n, _, h, w = wlh.shape
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        pred_feats = self.cls_conv(final_feat)
        pred_logits = self.cls_head(pred_feats)

        theta_feat = torch.cat([final_feat, pred_feats], dim=1)
        theta_feat = self.theta_conv(theta_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        wlh_feat = torch.cat([final_feat, theta_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-2, max=3)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        xyz_feat = torch.cat([final_feat, wlh_feat], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadCLSSTC(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.naive = heads.get('naive', False)

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        n, _, h, w = xy.shape
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape
        wlh = context['wlh']

        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        sample_point = box_utils.box_to_surface_bev(wlh, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        n, _, h, w = wlh.shape
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)

        context['wlh'] = wlh
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, torch.zeros_like(wlh[:, 0]), tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        pred_feats = self.cls_conv(final_feat)
        pred_logits = self.cls_head(pred_feats)

        wlh_feat = torch.cat([final_feat, pred_feats], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-2, max=3)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        theta_feat = torch.cat([final_feat, wlh_feat], dim=1)
        theta_feat = self.theta_conv(theta_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        xyz_feat = torch.cat([final_feat, theta_feat], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadCLSTCS(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.naive = heads.get('naive', False)

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        n, _, h, w = xy.shape
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        context['flatten_xy'] = raw_xy.permute(0, 2, 3, 1).contiguous().view(n * h * w, 2)
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape

        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']
        flatten_xy = context['flatten_xy']

        n, _, h, w = wlh.shape
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, flatten_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        pred_feats = self.cls_conv(final_feat)
        pred_logits = self.cls_head(pred_feats)

        theta_feat = torch.cat([final_feat, pred_feats], dim=1)
        theta_feat = self.theta_conv(theta_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        xyz_feat = torch.cat([final_feat, theta_feat], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        wlh_feat = torch.cat([final_feat, xyz_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-2, max=3)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret


class OneNetSeqFusionHeadFTSC(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__(**kwargs)

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])
        self.naive = heads.get('naive', False)

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)

        self.fusion_module = DCNV2Fusion(head_ch, num_cls, in_channels, head_ch, naive=self.naive)

        self.full_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.full_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(in_channels + (1 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels + (5 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels + (2 * head_ch), head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)
        self.fg_head.bias.data.fill_(init_bias)
        self.corner_head.bias.data.fill_(init_bias)
        self.center_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def to_grid_coord(self, global_x, global_y):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        x_v, y_v, _ = self.voxel_size
        xall = xmax - xmin - x_v
        yall = ymax - ymin - y_v
        grid_x = (global_x - (xmin + (x_v / 2))) / xall * 2 - 1
        grid_y = (global_y - (ymin + (y_v / 2))) / yall * 2 - 1
        return grid_x.contiguous(), grid_y.contiguous()

    def get_center_sampled_feat(self, xy, xy_feat, context):
        n, _, h, w = xy.shape
        raw_xy = xy[:, :2] + self.xy_offset  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, _, h, w = theta.shape
        sint, cost = theta[:, 1, :, :].contiguous(), theta[:, 0, :, :].contiguous()
        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        sample_point = box_utils.template_to_surface_bev(self.template_box, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']

        n, _, h, w = wlh.shape
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        # wlh = torch.clamp(wlh, min=0, max=9.4)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret = {}
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)

        fusion_feat = self.fusion_module(
            orig_feat=x,
            center_feat=torch.cat([center_feat, center_map], dim=1),
            corner_feat=torch.cat([corner_feat, corner_map], dim=1),
            foreground_feat=torch.cat([fg_feat, fg_map], dim=1)
        )

        final_feat = torch.cat([x, fusion_feat], dim=1)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        ret['final_feat'] = final_feat

        context = {}

        theta_feat = self.theta_conv(final_feat)
        theta = self.theta_head(theta_feat)
        theta = torch.clamp(theta, min=-2, max=2)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        wlh_feat = torch.cat([final_feat, theta_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-2, max=3)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        xyz_feat = torch.cat([final_feat, wlh_feat], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz = torch.clamp(xyz, min=-4, max=4)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        pred_feats = torch.cat([final_feat, xyz_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret['pred_logits'] = pred_logits
        ret['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)

        return ret