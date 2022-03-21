import copy

import torch
import torch.nn as nn
import numpy as np
from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch

from ...utils import box_coder_utils, box_utils
from pcdet.utils import matcher, common_utils
from pcdet.utils.set_crit import SetCritROI
from .roi_head_template import RoIHeadTemplate
from pcdet.ops.roipoint_pool3d_stack.roipoint_pool3d_stack_utils import RoIPointPool3dStack, RoIPointQueryStack, \
    grouping_operation


class PointInBoxTransformer(nn.Module):
    def __init__(self, feat_ch, emb_ch, out_ch, num_heads=2):
        super().__init__()
        self.ref_proj = nn.Linear(3, emb_ch)
        self.pts_coord_proj = nn.Linear(3, emb_ch)
        self.pts_feat_proj = nn.Linear(feat_ch, emb_ch)
        self.attn = nn.MultiheadAttention(emb_ch, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_ch)
        self.ffn = nn.Sequential(
            nn.Linear(emb_ch, emb_ch),
            nn.ReLU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.norm2 = nn.LayerNorm(emb_ch)
        self.out = nn.Sequential(
            nn.Linear(emb_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.ReLU()
        )

    def forward(self, ref_points, pts_coord, pts_feat, pts_mask):
        roi_center = ref_points[-1:, ...]
        ref_q = self.ref_proj(ref_points - roi_center)
        pts_kv = self.pts_coord_proj(pts_coord - roi_center)
        pts_v = self.pts_feat_proj(pts_feat)

        all_empty = (pts_mask.sum(dim=-1) == 0)
        pts_mask[:, 0] = False

        ref_feat, _ = self.attn(
            query=ref_q,
            key=pts_kv,
            value=pts_kv + pts_v,
            key_padding_mask=pts_mask
        )

        ref_feat[:, all_empty, :] = 0

        ref_feat = self.norm1(ref_feat + ref_q)
        act_feat = self.ffn(ref_feat)
        ref_feat = self.norm2(act_feat + ref_feat)
        rlt = self.out(ref_feat)

        return rlt


class SelfAttnTransformer(nn.Module):
    def __init__(self, in_ch, emb_ch, out_ch, num_heads=2):
        super().__init__()
        self.qkv_proj = nn.Linear(in_ch, 3 * emb_ch)
        self.ref_proj = nn.Linear(3, 3 * emb_ch)
        self.attn = nn.MultiheadAttention(emb_ch, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_ch)
        self.ffn = nn.Sequential(
            nn.Linear(emb_ch, emb_ch),
            nn.ReLU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.norm2 = nn.LayerNorm(emb_ch)
        self.out = nn.Sequential(
            nn.Linear(emb_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.ReLU()
        )

    def forward(self, box_feats, ref_points):
        # ref_points (bs, m, 5, 3)
        roi_center = ref_points[-1:, ...]
        ref = self.ref_proj(ref_points - roi_center)
        qkv = self.qkv_proj(box_feats)
        q, k, v = torch.chunk(qkv + ref, chunks=3, dim=-1)
        q = q[-1:, ...]
        attn_feat, _ = self.attn(
            query=q,
            key=k,
            value=v
        )
        ref_feat = self.norm1(attn_feat + q)
        act_feat = self.ffn(ref_feat)
        ref_feat = self.norm2(act_feat + ref_feat)
        rlt = self.out(ref_feat)

        return rlt


class BoxLevelTransformer(nn.Module):
    def __init__(self, box_ch, in_ch, emb_ch, out_ch, num_heads=2):
        super().__init__()
        self.box_proj = nn.Linear(box_ch, in_ch)
        self.qkv_proj = nn.Linear(in_ch, 3 * emb_ch)
        self.attn = nn.MultiheadAttention(emb_ch, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_ch)
        self.ffn = nn.Sequential(
            nn.Linear(emb_ch, emb_ch),
            nn.ReLU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.norm2 = nn.LayerNorm(emb_ch)
        self.out = nn.Sequential(
            nn.Linear(emb_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.ReLU()
        )

    def forward(self, box_feats, box_params):
        box_feats = box_feats + self.box_proj(box_params)
        q, k, v = torch.chunk(self.qkv_proj(box_feats), chunks=3, dim=-1)
        attn_feat, _ = self.attn(
            query=q,
            key=k,
            value=v
        )
        ref_feat = self.norm1(box_feats + attn_feat)
        act_feat = self.ffn(ref_feat)
        ref_feat = self.norm2(act_feat + ref_feat)
        rlt = self.out(ref_feat)

        return rlt


class EarlyFusionLayer(nn.Module):
    def __init__(self, input_feats, input_chs, mid_ch, out_ch, **kwargs):
        super().__init__()
        self.mid_layer = nn.ModuleDict()
        self.weight_layer = nn.ModuleDict()
        self.input_feat_num = len(input_feats)
        for k in input_feats:
            self.mid_layer[k] = nn.Sequential(
                nn.Linear(input_chs[k], mid_ch),
                nn.LayerNorm(mid_ch),
                nn.ReLU(inplace=True),
                nn.Linear(mid_ch, mid_ch),
                nn.LayerNorm(mid_ch),
                nn.ReLU(inplace=True)
            )

        self.weight_layer = nn.Sequential(
            nn.Linear(self.input_feat_num * mid_ch, mid_ch),
            nn.LayerNorm(mid_ch),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, self.input_feat_num),
            nn.Softmax(dim=-1)
        )

        self.out_layer = nn.Sequential(
            nn.Linear(mid_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_dict):
        mid_feats_list = []
        for k in feat_dict:
            mid_feats_list.append(self.mid_layer[k](feat_dict[k]))
        mid_feats = torch.cat(mid_feats_list, dim=-1)
        s, n, _ = mid_feats.shape
        weight = self.weight_layer(mid_feats) #S N 3
        mid_feats = mid_feats.view(s, n, -1, self.input_feat_num)
        rlt = torch.einsum("snl, sncl -> snc", weight, mid_feats)
        rlt = self.out_layer(rlt)
        return rlt


class E2EROIFusionHead(nn.Module):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.post_cfg = model_cfg.TEST_CONFIG
        self.use_focal_loss = True
        self.num_class = num_class
        self.in_chs = input_channels
        self.period = 2 * np.pi
        self.forward_ret_dict = {}

        self.downsample_times_map = {
            'x_conv3': 4,
            'x_conv4': 8,
            'bev': 8
        }

        self.in_ch_map = {
            'points': 2,
            'x_conv3': 64,
            'x_conv4': 128,
            'bev': 512,
        }

        self.emb_ch_map = {
            'points': 32,
            'x_conv3': 64,
            'x_conv4': 128,
            'bev': 128,
        }

        self.roipoint_pool3d_layer = RoIPointPool3dStack(pool_extra_width=model_cfg.BOX_EXTRA_WIDTH)

        self.voxel_src = getattr(self.model_cfg, 'VOXEL_SRC', ['x_conv3', 'x_conv4'])
        self.input_feats = copy.deepcopy(self.voxel_src)
        self.use_point = getattr(self.model_cfg, 'USE_POINT', True)
        self.use_bev = getattr(self.model_cfg, 'USE_BEV', True)
        self.use_fusion = getattr(self.model_cfg, 'USE_FUSION', False)
        self.use_efl = getattr(self.model_cfg, 'USE_EFL', False)

        if self.use_fusion:
            self.in_ch_map['bev'] = 160

        total = 0
        if self.use_point:
            self.input_feats.append('points')
            total += self.emb_ch_map['points']

        if self.use_bev:
            self.input_feats.append('bev')
            total += self.emb_ch_map['bev']

        for iter in self.voxel_src:
            total += self.emb_ch_map[iter]

        self.emb_ch_map['total'] = total

        if self.use_efl:
            self.eflayer = EarlyFusionLayer(self.input_feats, self.emb_ch_map, 64, 128)
            self.emb_ch_map['total'] = 128

        self.pc_range = point_cloud_range
        self.voxel_size = voxel_size

        self.voxel_query = RoIPointQueryStack(
            pool_extra_width=model_cfg.BOX_EXTRA_WIDTH,
            num_sampled_points=64
        )
        self.grouping_operation = grouping_operation

        box_coder_config = self.model_cfg.CODER_CONFIG.get('BOX_CODER_CONFIG', {})
        box_coder = getattr(box_coder_utils, self.model_cfg.CODER_CONFIG.BOX_CODER)(**box_coder_config)

        set_crit_settings = model_cfg.SET_CRIT_CONFIG
        matcher_settings = model_cfg.MATCHER_CONFIG
        self.matcher_weight_dict = matcher_settings['weight_dict']
        self.box_coder = box_coder

        matcher_settings['period'] = self.period
        self.matcher_weight_dict = matcher_settings['weight_dict']
        self.matcher = matcher.ROIMatcher(**matcher_settings)

        set_crit_settings['box_coder'] = box_coder
        set_crit_settings['matcher'] = self.matcher
        self.set_crit = SetCritROI(**set_crit_settings)

        self.point_feat_chs = 3 + 2  # xyz + point_scores + point_depth

        self.cls_layers = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, self.num_class, kernel_size=1, bias=True)
        )

        self.reg_layers = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, self.box_coder.code_size, kernel_size=1, bias=True)
        )

        self.pib_layers = nn.ModuleDict()
        self.adapt_layer = nn.ModuleDict()

        if self.use_point:
            self.pib_layers['points'] = PointInBoxTransformer(
                feat_ch=self.in_ch_map['points'],
                emb_ch=self.emb_ch_map['points'],
                out_ch=self.emb_ch_map['points'],
                num_heads=2
            )

        for src_name in self.voxel_src:
            self.pib_layers[src_name] = PointInBoxTransformer(
                feat_ch=self.in_ch_map[src_name],
                emb_ch=self.emb_ch_map[src_name],
                out_ch=self.emb_ch_map[src_name],
                num_heads=2
            )
            self.adapt_layer[src_name] = nn.Sequential(
                nn.Linear(self.in_ch_map[src_name], self.in_ch_map[src_name]),
                nn.BatchNorm1d(self.in_ch_map[src_name]),
                nn.ReLU()
            )

        if self.use_bev:
            self.adapt_layer['bev'] = nn.Sequential(
                nn.Conv2d(self.in_ch_map['bev'], self.emb_ch_map['bev'], kernel_size=1),
                nn.BatchNorm2d(self.emb_ch_map['bev']),
                nn.ReLU()
            )

        self.rib_layer = SelfAttnTransformer(
            in_ch=self.emb_ch_map['total'],
            emb_ch=256,
            out_ch=256,
            num_heads=2
        )
        self.box_layer = BoxLevelTransformer(box_ch=9, in_ch=256, emb_ch=256, out_ch=256, num_heads=2)
        self.box_proj = nn.Linear(num_class + 7, 256)
        self.init_weights(weight_init='xavier')

        self.cls_layers[-1].bias.data.fill_(-2.19)

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def get_roi_ref_points(self, batch_dict):
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)
        bs, m, _ = rois.shape

        roi_center = rois[:, :, 0:3].view(bs, m, 1, 3).contiguous()
        roi_ry = batch_dict['rois'][..., 6]

        roi_corners = box_utils.boxes_to_corners_3d(rois.view(bs * m, -1)[:, :7].contiguous()).view(bs, m, 8, 3)
        roi_queries = torch.cat([roi_corners, roi_center], dim=2).contiguous()
        roi_queries = self.to_local_coords(roi_queries, roi_center, roi_ry)
        return roi_queries

    def to_local_coords(self, points, ref_point, angle):
        # bs, m, p, 3
        # bs, m, 1, 3
        # bs, m
        tmp = points - ref_point
        b, m, p, _ = tmp.shape
        rlt = common_utils.rotate_points_along_z(tmp.view(b * m, p, -1), -angle.view(-1)).view(b, m, p, -1)
        return rlt

    def get_voxel_infos(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
        Returns:

        """
        rlt = {}
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)

        _, m, _ = rois.shape
        roi_center = rois[:, :, 0:3].view(batch_size, m, 1, 3).contiguous()
        roi_ry = batch_dict['rois'][..., 6]

        for src_name in self.voxel_src:
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()
            features = self.adapt_layer[src_name](features)
            batch_id = cur_coords[:, 0].int()
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.pc_range
            )
            voxel_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                voxel_batch_cnt[bs_idx] = (batch_id == bs_idx).sum()

            point_idx = self.voxel_query(xyz, batch_id, voxel_batch_cnt, rois)  # B, M, nsp
            id_batch_cnt = xyz.new_ones(batch_size).int() * m

            voxel_feats = self.grouping_operation(
                features,
                voxel_batch_cnt,
                point_idx.view(batch_size * m, -1),
                id_batch_cnt
            )
            nsample = voxel_feats.shape[-1]
            voxel_feats = voxel_feats.permute(0, 2, 1).contiguous().view(batch_size, m, nsample, -1)

            voxel_coords = self.grouping_operation(
                xyz,
                voxel_batch_cnt,
                point_idx.view(batch_size * m, -1),
                id_batch_cnt
            )

            voxel_coords = voxel_coords.permute(0, 2, 1).contiguous().view(batch_size, m, nsample, -1)
            voxel_coords = self.to_local_coords(voxel_coords, roi_center, roi_ry)

            masks = (point_idx == -1)
            tmp = {
                'coords': voxel_coords,
                'feats': voxel_feats,
                'masks': masks
            }
            rlt[src_name] = tmp

        return rlt

    def get_point_infos(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        point_batch_id = batch_dict['points'][:, 0].int()
        point_coords = batch_dict['points'][:, 1:4]
        point_features = batch_dict['points'][:, 4:]
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)

        bs, m, _ = rois.shape
        roi_center = rois[:, :, 0:3].view(bs, m, 1, 3).contiguous()
        roi_ry = batch_dict['rois'][..., 6]

        point_batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            point_batch_cnt[bs_idx] = (point_batch_id == bs_idx).sum()

        with torch.no_grad():
            pooled_features, pooled_empty_flag, point_empty_flag = self.roipoint_pool3d_layer(
                point_coords, point_features, point_batch_id, point_batch_cnt, rois
            )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)

            # canonical transformation
            sampled_points_coords = pooled_features[:, :, :, 0:3].contiguous()
            sampled_points_feats = pooled_features[:, :, :, 3:].contiguous()
            sampled_points_coords = self.to_local_coords(sampled_points_coords, roi_center, roi_ry)

        return sampled_points_coords, sampled_points_feats, point_empty_flag

    def get_bev_infos(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)
        bs, m, _ = rois.shape
        if self.use_fusion:
            spatial_features = self.adapt_layer['bev'](batch_dict['final_feat'])
        else:
            spatial_features = self.adapt_layer['bev'](batch_dict['spatial_features_2d'])

        roi_center = rois[:, :, 0:3].view(bs, m, 1, 3).contiguous()
        roi_corners = box_utils.boxes_to_corners_3d(rois.view(bs * m, -1)[:, :7].contiguous()).view(bs, m, 8, 3)
        roi_queries = torch.cat([roi_corners, roi_center], dim=2)[..., :2].contiguous().view(bs, m * 9, 2)

        x_idxs = (roi_queries[:, :, 0] - self.pc_range[0]) / self.voxel_size[0]
        y_idxs = (roi_queries[:, :, 1] - self.pc_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / self.downsample_times_map['bev']
        y_idxs = y_idxs / self.downsample_times_map['bev']

        bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = spatial_features[k].permute(1, 2, 0)  # (H, W, C)
            tmp = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            bev_features_list.append(tmp.unsqueeze(dim=0))

        bev_features = torch.cat(bev_features_list, dim=0)  # (B, N, C0)
        return bev_features.view(bs * m, 9, -1).permute(1, 0, 2).contiguous()


    @torch.no_grad()
    def pre_process(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']

        batch_dict['rois'] = batch_box_preds
        batch_dict['roi_scores'] = batch_cls_preds
        batch_dict['roi_labels'] = batch_cls_preds.argmax(dim=-1)
        if self.training:
            batch_dict['gt_dicts'] = batch_dict['gt_dicts'][0]
        batch_dict['has_class_labels'] = True
        batch_dict.pop('batch_index', None)

        return batch_dict

    def _get_pred_boxes(self, batch_dict, rcnn_cls, rcnn_reg):
        pred_dict = {}
        rois = batch_dict['rois']
        bs = rois.shape[0]

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)

        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        pred_boxes = self.box_coder.decode_torch(rcnn_reg, local_rois).view(-1, rois.shape[-1])

        pred_boxes = common_utils.rotate_points_along_z(
            pred_boxes.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        pred_boxes[:, 0:3] += roi_xyz
        pred_boxes = pred_boxes.view(bs, -1, rois.shape[-1])

        pred_dict['pred_boxes'] = pred_boxes
        pred_dict['pred_logits'] = rcnn_cls
        pred_dict['rcnn_regs'] = rcnn_reg
        pred_dict['roi_scores'] = batch_dict['roi_scores']
        pred_dict['rois'] = rois

        return pred_dict

    def forward(self, batch_dict):
        # get proposals
        self.pre_process(batch_dict)
        ref_points = self.get_roi_ref_points(batch_dict)
        pts_coord, pts_feat, pts_mask = self.get_point_infos(batch_dict)
        b, nr, np = pts_coord.shape[:3]
        # change dim
        ref_points = ref_points.view(b * nr, 9, 3).permute(1, 0, 2).contiguous()

        ref_feat_dict = {}

        if self.use_point:
            ref_feat_dict['points'] = self.pib_layers['points'](
                ref_points,
                pts_coord.view(b * nr, np, 3).permute(1, 0, 2).contiguous(),
                pts_feat.view(b * nr, np, self.in_ch_map['points']).permute(1, 0, 2).contiguous(),
                (pts_mask == -1).view(b * nr, np)
            )

        if self.use_bev:
            ref_feat_dict['bev'] = self.get_bev_infos(batch_dict)

        voxel_dict = self.get_voxel_infos(batch_dict)
        for src_name in self.voxel_src:
            tmp = voxel_dict[src_name]
            np = tmp['coords'].shape[2]
            ref_feat_dict[src_name] = self.pib_layers[src_name](
                ref_points,
                tmp['coords'].view(b * nr, np, 3).permute(1, 0, 2).contiguous(),
                tmp['feats'].view(b * nr, np, self.in_ch_map[src_name]).permute(1, 0, 2).contiguous(),
                tmp['masks'].view(b * nr, np)
            )

        if not self.use_efl:
            ref_feat_list = [ref_feat_dict[i] for i in ref_feat_dict]
            ref_feats = torch.cat(ref_feat_list, dim=-1)
        else:
            ref_feats = self.eflayer(ref_feat_dict)

        box_feats = self.rib_layer(ref_feats, ref_points).view(b, nr, -1).permute(1, 0, 2).contiguous()
        box_params = torch.cat(
            [
                batch_dict['rois'],
                batch_dict['roi_scores'],
                batch_dict['roi_labels'].unsqueeze(-1).float()
            ],
            dim=-1
        ).permute(1, 0, 2).contiguous()
        box_feats = self.box_layer(box_feats, box_params)  # + self.box_proj(box_params)
        box_feats = box_feats.permute(1, 2, 0).contiguous()

        rcnn_cls = self.cls_layers(box_feats).permute(0, 2, 1).contiguous()  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(box_feats).permute(0, 2, 1).contiguous()  # (B, C)

        pred_dict = self._get_pred_boxes(batch_dict, rcnn_cls, rcnn_reg)

        self.forward_ret_dict['pred_dicts'] = pred_dict
        if self.training:
            self.forward_ret_dict['gt_dicts'] = batch_dict['gt_dicts']
        else:
            self.generate_predicted_boxes(batch_dict)

        return batch_dict

    def get_loss(self, curr_epoch, **kwargs):
        tb_dict = {}

        pred_dicts = self.forward_ret_dict['pred_dicts']

        gt_dicts = self.forward_ret_dict['gt_dicts']
        loss_dicts = self.set_crit(pred_dicts, gt_dicts, curr_epoch)
        loss = loss_dicts['loss']

        return loss, tb_dict

    @torch.no_grad()
    def generate_predicted_boxes(self, data_dict):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        bs = pred_dicts['pred_boxes'].shape[0]

        k = self.post_cfg.k
        thresh = self.post_cfg.thresh

        if self.use_focal_loss:
            pred_scores = pred_dicts['pred_logits'].sigmoid()
        else:
            pred_scores = pred_dicts['pred_logits'].softmax(2)

        rlt = []
        for idx in range(bs):
            pred_score = pred_scores[idx]
            pred_box = pred_dicts['pred_boxes'][idx]

            cls_num = pred_score.size(1)
            tmp_scores, tmp_cat_inds = torch.topk(pred_score, k=k, dim=0)

            final_score, tmp_ids = torch.topk(tmp_scores.reshape(-1), k=k)
            final_label = (tmp_ids % cls_num)

            topk_box_cat = pred_box[tmp_cat_inds.reshape(-1), :]
            final_box = topk_box_cat[tmp_ids, :]

            # used for 2 stage network
            mask = final_score >= thresh
            final_score = final_score[mask]
            final_box = final_box[mask]
            final_label = final_label[mask]

            end = min(final_score.size(0), 300)

            record_dict = {
                "pred_boxes": final_box[:end],
                "pred_scores": final_score[:end],
                "pred_labels": final_label[:end]
            }
            rlt.append(record_dict)

        # import pdb; pdb.set_trace()
        data_dict['pred_dicts'] = rlt
        data_dict['has_class_labels'] = True  # Force to be true
        return data_dict
