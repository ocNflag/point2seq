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
from .roi_head_template import RoIHeadTemplate


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
        # ref_points (bs, m, 5, 3)
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


class ROIFusionHead(RoIHeadTemplate):
    def __init__(self, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.num_class = num_class
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
        self.use_point = getattr(self.model_cfg, 'USE_POINT', True)
        self.use_bev = getattr(self.model_cfg, 'USE_BEV', True)

        total = 0
        if self.use_point:
            total += self.emb_ch_map['points']

        if self.use_bev:
            total += self.emb_ch_map['bev']

        for iter in self.voxel_src:
            total += self.emb_ch_map[iter]

        self.emb_ch_map['total'] = total

        self.pc_range = point_cloud_range
        self.voxel_size = voxel_size

        self.voxel_query = RoIPointQueryStack(
            pool_extra_width=model_cfg.BOX_EXTRA_WIDTH,
            num_sampled_points=128
        )
        self.grouping_operation = grouping_operation

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


    def forward(self, batch_dict):
        # get proposals
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_scores'] = targets_dict['roi_scores']
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

        ref_feat_list = [ref_feat_dict[i] for i in ref_feat_dict]
        ref_feats = torch.cat(ref_feat_list, dim=-1)

        box_feats = self.rib_layer(ref_feats, ref_points).view(b, nr, -1).permute(1, 0, 2).contiguous()
        box_params = torch.cat(
            [
                batch_dict['rois'],
                batch_dict['roi_scores'].unsqueeze(-1),
                batch_dict['roi_labels'].unsqueeze(-1)
            ],
            dim=-1
        ).permute(1, 0, 2).contiguous()
        box_feats = self.box_layer(box_feats, box_params)  # + self.box_proj(box_params)
        box_feats = box_feats.permute(1, 2, 0).contiguous()

        rcnn_cls = self.cls_layers(box_feats).permute(0, 2, 1).contiguous()  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(box_feats).permute(0, 2, 1).contiguous()  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
