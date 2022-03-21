import torch
import torch.nn as nn
import numpy as np

from ...utils import box_coder_utils, box_utils
from pcdet.utils import matcher, common_utils
from pcdet.utils.set_crit import SetCritROI
from .roi_head_template import RoIHeadTemplate
from pcdet.utils.common_utils import GELU
from ...ops.iou3d_nms import iou3d_nms_cuda
from pcdet.ops.roipoint_pool3d_stack.roipoint_pool3d_stack_utils import RoIPointPool3dStack, RoIPointQueryStack, \
    grouping_operation


class PointInBoxTransformer(nn.Module):
    def __init__(self, feat_ch, emb_ch, out_ch, num_heads=2, ref_ch=None):
        super().__init__()
        self.ref_ch = ref_ch
        if ref_ch:
            self.ref_feat_proj = nn.Linear(ref_ch, emb_ch)
        self.ref_proj = nn.Linear(3, emb_ch)
        self.pts_coord_proj = nn.Linear(3, emb_ch)
        self.pts_feat_proj = nn.Linear(feat_ch, emb_ch)
        self.attn = nn.MultiheadAttention(emb_ch, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_ch)
        self.ffn = nn.Sequential(
            nn.Linear(emb_ch, emb_ch),
            GELU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.norm2 = nn.LayerNorm(emb_ch)
        self.out = nn.Sequential(
            nn.Linear(emb_ch, out_ch),
            nn.LayerNorm(out_ch),
            GELU(),
        )

    def forward(self, ref_points, pts_coord, pts_feat, pts_mask, in_ref_feat=None):
        # ref_points (bs, m, 5, 3)
        roi_center = ref_points[-1:, ...]
        ref_q = self.ref_proj(ref_points - roi_center)
        if self.ref_ch:
            ref_q = ref_q + self.ref_feat_proj(in_ref_feat)
        pts_kv = self.pts_coord_proj(pts_coord - roi_center)
        pts_v = self.pts_feat_proj(pts_feat)

        all_empty = (pts_mask.sum(dim=-1) == 0)
        pts_mask = pts_mask.clone()
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


class PointInBoxBlock(nn.Module):
    def __init__(self, feat_ch=2, emb_ch=64, out_ch=64, num_heads=2, layer=2):
        super().__init__()
        self.in_tr = PointInBoxTransformer(feat_ch, emb_ch, out_ch, num_heads=num_heads)
        self.tr = nn.ModuleList()
        for i in range(layer - 1):
            self.tr.append(
                PointInBoxTransformer(feat_ch, emb_ch, out_ch, num_heads=num_heads, ref_ch=out_ch)
            )

    def forward(self, ref_points, pts_coord, pts_feat, pts_mask):
        rlt = self.in_tr(ref_points, pts_coord, pts_feat, pts_mask)
        for iter in self.tr:
            rlt = iter(ref_points, pts_coord, pts_feat, pts_mask, in_ref_feat=rlt)

        return rlt


class SelfAttnBlock(nn.Module):
    def __init__(self, in_ch, emb_ch, out_ch, num_points, final_out_ch=256, num_heads=2, layer=3):
        super().__init__()
        self.in_tr = SelfAttnTransformer(in_ch, emb_ch, out_ch, num_heads)
        self.tr = nn.ModuleList()
        for i in range(layer - 1):
            self.tr.append(
                SelfAttnTransformer(out_ch, emb_ch, out_ch, num_heads)
            )
        self.final_out_layer = nn.Sequential(
            nn.Linear(num_points * out_ch, final_out_ch),
            nn.BatchNorm1d(final_out_ch),
            GELU(),
        )

    def forward(self, box_feats, ref_points):
        rlt = self.in_tr(box_feats, ref_points)
        for iter in self.tr:
            rlt = iter(rlt, ref_points)

        np, n, c = rlt.shape
        rlt = rlt.permute(1, 0, 2).contiguous().view(n, c * np)
        rlt = self.final_out_layer(rlt)
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
            GELU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.norm2 = nn.LayerNorm(emb_ch)
        self.out = nn.Sequential(
            nn.Linear(emb_ch, out_ch),
            nn.LayerNorm(out_ch),
            GELU(),
        )

    def forward(self, box_feats, ref_points):
        # ref_points (bs, m, 5, 3)
        roi_center = ref_points[-1:, ...]
        ref = self.ref_proj(ref_points - roi_center)
        qkv = self.qkv_proj(box_feats)
        q, k, v = torch.chunk(qkv + ref, chunks=3, dim=-1)
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


class BoxTransformer(nn.Module):
    def __init__(self, in_ch, emb_ch, out_ch, num_heads=2):
        super().__init__()
        self.qkv_proj = nn.Linear(in_ch, 3 * emb_ch)
        self.attn = nn.MultiheadAttention(emb_ch, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_ch)
        self.ffn = nn.Sequential(
            nn.Linear(emb_ch, emb_ch),
            GELU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.norm2 = nn.LayerNorm(emb_ch)
        self.out = nn.Sequential(
            nn.Linear(emb_ch, out_ch),
            nn.LayerNorm(out_ch),
            GELU(),
        )

    def forward(self, box_feats):
        # ref_points (bs, m, 5, 3)
        qkv = self.qkv_proj(box_feats)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
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
            GELU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.norm2 = nn.LayerNorm(emb_ch)
        self.out = nn.Sequential(
            nn.Linear(emb_ch, out_ch),
            nn.LayerNorm(out_ch),
            GELU(),
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


class BoxLevelTransformerBlock(nn.Module):
    def __init__(self, box_ch, in_ch, emb_ch, out_ch, block_num=1, num_heads=2):
        super().__init__()
        assert block_num >= 1
        self.trans1 = BoxLevelTransformer(box_ch=box_ch, in_ch=in_ch, emb_ch=emb_ch, out_ch=emb_ch, num_heads=num_heads)
        self.trans2 = nn.ModuleList()
        final_out_ch = emb_ch
        for i in range(block_num - 1):
            if i == block_num - 2:
                final_out_ch = out_ch
            self.trans2.append(
                BoxTransformer(in_ch=emb_ch, emb_ch=emb_ch, out_ch=final_out_ch, num_heads=num_heads)
            )

    def forward(self, box_feats, box_params):
        box_feats = self.trans1(box_feats, box_params)
        for i in self.trans2:
            box_feats = i(box_feats)

        return box_feats


class E2EROIHeadV2(nn.Module):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.post_cfg = model_cfg.TEST_CONFIG
        self.use_roi_score = self.model_cfg.get('USE_ROI_SCORE', True)
        self.pib_layer_num = self.model_cfg.get('PIB_LAYER_NUM', 1)
        self.rib_layer_num = self.model_cfg.get('RIB_LAYER_NUM', 1)
        self.use_focal_loss = True
        self.num_class = num_class
        self.in_chs = input_channels
        self.period = 2 * np.pi
        self.forward_ret_dict = {}
        self.sample_points = 27

        self.roipoint_pool3d_layer = RoIPointPool3dStack(pool_extra_width=model_cfg.BOX_EXTRA_WIDTH)

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
            GELU(),
            nn.Conv1d(64, self.num_class, kernel_size=1, bias=True)
        )

        self.reg_layers = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            GELU(),
            nn.Conv1d(64, self.box_coder.code_size, kernel_size=1, bias=True)
        )

        self.pib_layer = PointInBoxBlock(feat_ch=2, emb_ch=64, out_ch=64, num_heads=2, layer=self.pib_layer_num)
        self.rib_layer = SelfAttnBlock(in_ch=64, emb_ch=64, out_ch=64, num_heads=2, num_points=self.sample_points, layer=self.rib_layer_num)

        box_ch = 7
        if self.use_roi_score:
            box_ch += num_class

        self.box_layer = BoxLevelTransformer(box_ch=box_ch, in_ch=256, emb_ch=256, out_ch=256, num_heads=2)
        self.box_proj = nn.Linear(num_class + 7, 64)
        self.init_weights(weight_init='xavier')

        self.cls_layers[-1].bias.data.fill_(-2.19)

    def _nms_gpu_3d(self, boxes, scores, thresh, pre_maxsize=None, post_max_size = None):
        """
        :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        :param scores: (N)
        :param thresh:
        :return:
        """
        assert boxes.shape[1] == 7
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]

        boxes = boxes[order].contiguous()
        keep = torch.LongTensor(boxes.size(0))
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
        selected = order[keep[:num_out].cuda()].contiguous()

        if post_max_size is not None:
            selected = selected[:post_max_size]

        return selected

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

        roi_corners = box_utils.boxes_to_sps_3d(rois.view(bs * m, -1)[:, :7].contiguous()).view(bs, m, self.sample_points - 1, 3)
        roi_queries = torch.cat([roi_corners, roi_center], dim=2).contiguous()
        roi_queries = self.to_local_coords(roi_queries, roi_center, roi_ry)
        return roi_queries

    def to_local_coords(self, points, ref_point, angle):
        tmp = points - ref_point
        b, m, p, _ = tmp.shape
        rlt = common_utils.rotate_points_along_z(tmp.view(b * m, p, -1), -angle.view(-1)).view(b, m, p, -1)
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
        pred_dict['rois'] = rois
        pred_dict['roi_scores'] = batch_dict['roi_scores']
        return pred_dict

    def forward(self, batch_dict):
        # get proposals
        self.pre_process(batch_dict)
        ref_points = self.get_roi_ref_points(batch_dict)
        pts_coord, pts_feat, pts_mask = self.get_point_infos(batch_dict)
        b, nr, np = pts_coord.shape[:3]
        # change dim
        ref_points = ref_points.view(b * nr, self.sample_points, 3).permute(1, 0, 2).contiguous()
        ref_feats = self.pib_layer(
            ref_points,
            pts_coord.view(b * nr, np, 3).permute(1, 0, 2).contiguous(),
            pts_feat.view(b * nr, np, 2).permute(1, 0, 2).contiguous(),
            (pts_mask == -1).view(b * nr, np)
        )
        box_feats = self.rib_layer(ref_feats, ref_points).view(b, nr, -1).permute(1, 0, 2).contiguous()
        box_params = torch.cat([batch_dict['rois'], batch_dict['roi_scores']], dim=-1).permute(1, 0, 2).contiguous()

        box_feats = self.box_layer(box_feats, box_params) # + self.box_proj(box_params)
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
        use_nms = self.post_cfg.use_nms
        thresh = self.post_cfg.thresh
        mix_score = self.post_cfg.get('mix_score', False)

        if self.use_focal_loss:
            if mix_score:
                pred_scores = torch.sqrt(pred_dicts['pred_logits'].sigmoid() * pred_dicts['roi_scores'])
            else:
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

            if use_nms:
                if final_score.shape[0] != 0:
                    selected = self._nms_gpu_3d(final_box[:, :7],
                                                final_score,
                                                thresh=self.post_cfg.nms_iou_threshold,
                                                pre_maxsize=self.post_cfg.nms_pre_max_size,
                                                post_max_size=self.post_cfg.nms_post_max_size)
                else:
                    selected = []

                final_box = final_box[selected]
                final_score = final_score[selected]
                final_label = final_label[selected]

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
