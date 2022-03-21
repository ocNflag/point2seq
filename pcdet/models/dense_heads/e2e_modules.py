import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils import box_utils

from pcdet.utils.common_utils import limit_period_torch


class Performer(nn.Module):
    def __init__(self, in_ch, emb_ch, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.emb = emb_ch * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Linear(in_ch, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(in_ch)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.ReLU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SA(nn.Module):
    def __init__(self, in_ch=64, attn_ch=64, emb_ch=16, ks=3):
        super().__init__()
        self.ks2 = ks * ks
        self.emb_ch = emb_ch
        self.proj1 = nn.Conv2d(in_ch, attn_ch, kernel_size=1, stride=1, bias=True)
        self.soft_split = nn.Unfold(kernel_size=ks, stride=1, padding=ks // 2)
        self.attention = Performer(in_ch=attn_ch, emb_ch=emb_ch, kernel_ratio=0.5)
        self.proj2 = nn.Conv2d(emb_ch * ks * ks, in_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        input = x
        x = self.proj1(x)
        B, C, H, W = x.size()
        x = self.soft_split(x).transpose(1, 2).contiguous()
        x = x.view(B * H * W, C, self.ks2).transpose(1, 2).contiguous()
        x = self.attention(x)
        x = x.reshape(B, H, W, self.ks2 * self.emb_ch).permute(0, 3, 1, 2)
        x = self.proj2(x)
        return x.contiguous() + input


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


class OneNetSingleHead(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super(OneNetSingleHead, self).__init__(**kwargs)

        ks = heads['kernel_size']
        self.use_sa = heads.get('use_sa', False)

        if self.use_sa:
            out_ch_conv1 = heads['head_channels']
        else:
            out_ch_conv1 = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_ch_conv1, kernel_size=ks, stride=1, padding=ks // 2),
            nn.ReLU(inplace=True),
        )

        if self.use_sa:
            self.sa = SA(in_ch=out_ch_conv1, attn_ch=out_ch_conv1)

        self.cls_head = nn.Conv2d(out_ch_conv1, heads['num_classes'], kernel_size=ks, stride=1, padding=ks // 2)

        code_size = heads['code_size']
        if heads['encode_angle_by_sincos']:
            code_size += 1

        self.bbox_head = nn.Sequential(
            nn.Conv2d(out_ch_conv1, heads['head_channels'], kernel_size=ks, stride=1, padding=ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(heads['head_channels'], code_size, kernel_size=ks, stride=1, padding=ks // 2),
        )

        self._reset_parameters()
        self.cls_head.bias.data.fill_(heads['init_bias'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        ret_dict = {}
        feat = self.conv1(x)
        if self.use_sa:
            feat = self.sa(feat)
        ret_dict['pred_logits'] = self.cls_head(feat)
        ret_dict['pred_boxes'] = self.bbox_head(feat)
        return ret_dict


class OneNetSeqHead(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__()

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

        self.register_buffer('template_box', torch.tensor(heads['template_box']))
        self.pc_range = heads['pc_range']
        self.voxel_size = heads['voxel_size']
        self.register_buffer('xy_offset', heads['offset_grid'])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.ReLU(inplace=True),
        )

        self.cls_conv = nn.Sequential(
            nn.Conv2d(5 * head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(head_ch, num_cls, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(2 * head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(5 * head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self._reset_parameters()

        self.cls_head.bias.data.fill_(init_bias)

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
        wlh[wlh != wlh] = 0
        sample_point = box_utils.box_to_surface_bev(wlh, theta, flatten_xy)
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

        xyz_feat = self.xyz_conv(final_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        theta_feat = torch.cat([final_feat, xyz_feat], dim=1)
        theta_feat = self.theta_conv(theta_feat)
        theta = self.theta_head(theta_feat)
        theta_feat = self.get_theta_sampled_feat(theta, theta_feat, context)

        wlh_feat = torch.cat([final_feat, theta_feat], dim=1)
        wlh_feat = self.wlh_conv(wlh_feat)
        wlh = self.wlh_head(wlh_feat)
        wlh = torch.clamp(wlh, min=-5, max=5)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        pred_feats = torch.cat([final_feat, wlh_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret_dict['pred_logits'] = pred_logits
        ret_dict['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)
        return ret_dict


class OneNetSeqHeadTSC(nn.Module):
    def __init__(self, in_channels, heads, **kwargs):
        super().__init__()

        ks = heads['kernel_size']
        num_cls = heads['num_classes']
        head_ch = heads['head_channels']
        init_bias = heads['init_bias']

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
        self.theta_head = nn.Conv2d(head_ch, 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.wlh_conv = nn.Sequential(
            nn.Conv2d(5 * head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.wlh_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(5 * head_ch, head_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Conv2d(head_ch, 3, kernel_size=ks, stride=1, padding=ks // 2)

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
        wlh = torch.clamp(wlh, min=-7, max=5)
        wlh_feat = self.get_wlh_sampled_feat(wlh, wlh_feat, context)

        xyz_feat = torch.cat([final_feat, wlh_feat], dim=1)
        xyz_feat = self.xyz_conv(xyz_feat)
        xyz = self.xyz_head(xyz_feat)
        xyz_feat = self.get_center_sampled_feat(xyz, xyz_feat, context)

        pred_feats = torch.cat([final_feat, xyz_feat], dim=1)
        pred_feats = self.cls_conv(pred_feats)
        pred_logits = self.cls_head(pred_feats)

        ret_dict['pred_logits'] = pred_logits
        ret_dict['pred_boxes'] = torch.cat([xyz, wlh, theta], dim=1)
        return ret_dict