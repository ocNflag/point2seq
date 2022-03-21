import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils import box_utils

from pcdet.utils.common_utils import limit_period_torch


class GroundTruthProcessor(object):
    def __init__(self, gt_processor_cfg):
        self.tasks = gt_processor_cfg.tasks
        self.class_to_idx = gt_processor_cfg.mapping
        self.period = 2 * np.pi

    def limit_period_wrapper(self, input, offset=0, dim=6):
        prev, r, rem = input[..., :dim], input[..., dim:dim + 1], input[..., dim + 1:]
        r = limit_period_torch(r, offset=offset, period=self.period)
        return torch.cat([prev, r, rem], dim=-1)

    def process(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)

        Returns:
            gt_dicts: a dict key is task id
            each item is a dict {
                gt_class: list(Tensor), len = batchsize, Tensor with size (box_num, 10)
                gt_boxes: list(Tensor), len = batchsize, Tensor with size (box_num, 10)
            }
        """

        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, -1]  # begin from 1
        gt_boxes = gt_boxes[:, :, :-1]

        gt_dicts = {}

        for task_id, task in enumerate(self.tasks):
            gt_dicts[task_id] = {}
            gt_dicts[task_id]['gt_classes'] = []
            gt_dicts[task_id]['gt_boxes'] = []

        for k in range(batch_size):
            # remove padding
            iter_box = gt_boxes[k]
            count = len(iter_box) - 1
            while count > 0 and iter_box[count].sum() == 0:
                count -= 1

            iter_box = iter_box[:count + 1]
            iter_gt_classes = gt_classes[k][:count + 1].int()

            for task_id, task in enumerate(self.tasks):
                boxes_of_tasks = []
                classes_of_tasks = []
                class_offset = 0

                for class_name in task.class_names:
                    class_idx = self.class_to_idx[class_name]
                    class_mask = (iter_gt_classes == class_idx)
                    _boxes = iter_box[class_mask]
                    _boxes = self.limit_period_wrapper(_boxes)
                    _class = _boxes.new_full((_boxes.shape[0],), class_offset).long()
                    boxes_of_tasks.append(_boxes)
                    classes_of_tasks.append(_class)
                    class_offset += 1

                task_boxes = torch.cat(boxes_of_tasks, dim=0)
                task_classes = torch.cat(classes_of_tasks, dim=0)
                gt_dicts[task_id]['gt_boxes'].append(task_boxes)
                gt_dicts[task_id]['gt_classes'].append(task_classes)
                gt_dicts[task_id]['gt_cls_num'] = len(task.class_names)

        return gt_dicts



class OneNetSeqTokenHeadTSC(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__()

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']
        self.bin_num = heads['bin_num']
        self.bin_size = heads['bin_size']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.ReLU(inplace=True),
        )

        self.theta_conv = nn.Sequential(
            nn.Conv2d(head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2 * self.bin_num['theta'], kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(5 * head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3 * self.bin_num['wlh'], kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(5 * head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, (2 * self.bin_num['xy']) + self.bin_num['z'], kernel_size=ks, stride=1, padding=ks // 2)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(2 * head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_box_params(self, pred, anc, attr_key, dim=-1):
        rlt = (pred.argmax(dim=dim) - int(self.bin_num[attr_key] / 2) + 0.5) * self.bin_size[attr_key] + anc
        return rlt

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
        xp, yp, _ = torch.split(xy, [self.bin_num['xy'], self.bin_num['xy'], self.bin_num['z']], dim=1)
        xa, ya = self.xy_offset[:, 0], self.xy_offset[:, 1]
        raw_x = self._get_box_params(xp, xa, 'xy', dim=1)
        raw_y = self._get_box_params(yp, ya, 'xy', dim=1)

        raw_xy = torch.stack([raw_x, raw_y], dim=1)  # n, 2, h, w
        grid_x, grid_y = self.to_grid_coord(raw_xy[:, 0], raw_xy[:, 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1)
        sample_feats = F.grid_sample(xy_feat, sample_grids)
        return sample_feats

    def get_theta_sampled_feat(self, theta, theta_feat, context):
        n, c, h, w = theta.shape
        cost, sint = torch.split(theta, [self.bin_num['theta'], self.bin_num['theta']], dim=1)
        sint = self._get_box_params(sint, 0 , 'theta', dim=1)
        cost = self._get_box_params(cost, 0, 'theta', dim=1)

        theta = torch.atan2(sint, cost).view(-1)  # N, h, w - > n*h*w
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)

        context['theta'] = theta
        sample_point = box_utils.template_to_surface_bev(self.template_box[1:], theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(theta_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def get_wlh_sampled_feat(self, wlh, wlh_feat, context):
        theta = context['theta']

        wt, lt, ht = torch.split(wlh, [self.bin_num['wlh'], self.bin_num['wlh'], self.bin_num['wlh']], dim=1)
        wt = self._get_box_params(wt, self.template_box[1], 'wlh', dim=1)
        lt = self._get_box_params(lt, self.template_box[2], 'wlh', dim=1)
        ht = self._get_box_params(ht, self.template_box[3], 'wlh', dim=1)
        wlh = torch.stack([wt, lt, ht], dim=1)

        n, _, h, w = wlh.shape
        tmp_xy = self.xy_offset.permute(0, 2, 3, 1).repeat(n, 1, 1, 1).contiguous().view(n * h * w, 2)
        wlh = torch.exp(wlh).permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        sample_point = box_utils.box_to_surface_bev(wlh, theta, tmp_xy)
        sample_point = sample_point.view(n, h, w, 4, 2)

        grid_x, grid_y = self.to_grid_coord(sample_point[..., 0], sample_point[..., 1])
        sample_grids = torch.stack([grid_x, grid_y], dim=-1).view(n, h, w * 4, 2)  # n, 2, 4, h, w
        sample_feats = F.grid_sample(wlh_feat, sample_grids).view(n, -1, h, w, 4).permute(0, 1, 4, 2, 3).contiguous()
        sample_feats = sample_feats.view((n, -1, h, w))
        return sample_feats

    def forward(self, x):
        ret_dict = {}

        final_feat = self.conv1(x)
        context = {}

        theta_feat = self.theta_conv(final_feat)
        theta = self.theta_head(theta_feat)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        wlh_feat = torch.cat([final_feat, theta_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        xyz_feat = torch.cat([final_feat, wlh_feat], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        pred_feats = torch.cat([final_feat, xyz_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret_dict['pred_logits'] = pred_logits
        # instead of cat we divided the attr to single
        x, y, z = torch.split(xyz, [self.bin_num['xy'], self.bin_num['xy'], self.bin_num['z']], dim=1)
        w, l, h = torch.split(wlh, [self.bin_num['wlh'], self.bin_num['wlh'], self.bin_num['wlh']], dim=1)
        cos, sin = torch.split(theta, [self.bin_num['theta'], self.bin_num['theta']], dim=1)
        ret_dict['pred_box_bins'] = [x, y, z, w, l, h, cos, sin]
        ret_dict['anchor_zwlh'] = self.template_box
        ret_dict['bin_size'] = self.bin_size
        ret_dict['bin_num'] = self.bin_num
        return ret_dict