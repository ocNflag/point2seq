import torch
import torch.nn as nn
import numpy as np
from pcdet.utils import box_coder_utils, matcher

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from ...utils.set_crit import SetCritROI


class E2EPVRCNNHead(nn.Module):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.post_cfg = model_cfg.TEST_CONFIG
        self.use_focal_loss = True
        self.num_class = num_class
        self.in_chs = input_channels
        self.period = 2 * np.pi
        self.forward_ret_dict = {}

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

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

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

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def roi_grid_pool(self, batch_dict):
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
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

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
        """
        :param input_data: input dict
        :return:
        """

        self.pre_process(batch_dict)

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        bs = batch_dict['batch_size']
        nbox = int(batch_size_rcnn / bs)

        rcnn_cls = rcnn_cls.view(bs, nbox, -1)
        rcnn_reg = rcnn_reg.view(bs, nbox, -1)

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