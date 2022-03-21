import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import math

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

from pcdet.models.dense_heads.utils import _sigmoid

from ...utils import box_coder_utils, common_utils
from pcdet.utils import matcher
from pcdet.utils.set_crit import SetCriterion
from pcdet.models.dense_heads.e2e_refine_modules import OneNetRefineHead
from pcdet.models.dense_heads.target_assigner.merged_assigner import MergedAssigner
from pcdet.utils import loss_utils
from ...ops.iou3d_nms import iou3d_nms_cuda

SingleHeadDict = {
    'OneNetRefineHead': OneNetRefineHead
}


class E2ERefinementHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, voxel_size,
                 point_cloud_range, predict_boxes_when_training, **kwargs):
        super().__init__()
        self.xoffset = None
        self.yoffset = None
        self.no_log = False
        self.forward_ret_dict = {}
        self.model_cfg = model_cfg
        self.voxel_size = [model_cfg.TARGET_ASSIGNER_CONFIG['out_size_factor'] * iter for iter in voxel_size]

        self.period = 2 * np.pi
        # if self.use_dir_classifier:
        #     self.period = self.period / self.num_dir_bins

        self.single_head = self.model_cfg.get('SingleHead', 'OneNetRefineHead')
        self.final_layer_loss = self.model_cfg.get('FinalLayerLoss', 'False')
        self.post_cfg = model_cfg.TEST_CONFIG
        self.in_channels = input_channels
        self.predict_boxes_when_training = predict_boxes_when_training

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.out_size_factor = model_cfg.OUT_SIZE_FACTOR
        self._generate_offset_grid()

        self.num_classes = [t["num_class"] for t in model_cfg.TASKS]
        self.class_names = [t["class_names"] for t in model_cfg.TASKS]
        self.template_boxes = [t["template_box"] for t in model_cfg.TASKS]
        self.total_classes = sum(self.num_classes)

        box_coder_config = self.model_cfg.CODER_CONFIG.get('BOX_CODER_CONFIG', {})
        box_coder_config['period'] = self.period
        box_coder = getattr(box_coder_utils, self.model_cfg.CODER_CONFIG.BOX_CODER)(**box_coder_config)

        set_crit_settings = model_cfg.SET_CRIT_CONFIG
        matcher_settings = model_cfg.MATCHER_CONFIG
        self.matcher_weight_dict = matcher_settings['weight_dict']
        self.use_focal_loss = model_cfg.USE_FOCAL_LOSS
        self.box_coder = box_coder

        matcher_settings['box_coder'] = box_coder
        matcher_settings['period'] = self.period
        self.matcher_weight_dict = matcher_settings['weight_dict']
        self.matcher = getattr(matcher, self.model_cfg.MATCHER)(**matcher_settings)

        set_crit_settings['box_coder'] = box_coder
        set_crit_settings['matcher'] = self.matcher
        self.set_crit = SetCriterion(**set_crit_settings)

        self.aux_loss_weights = self.model_cfg.AUX_LOSS_WEIGHTS
        self.loss_center = loss_utils.CenterNetFocalLoss()
        self.loss_corner = loss_utils.CenterNetFocalLoss()
        self.loss_foreground = loss_utils.ForegroundFocalLoss()

        self.target_assigner = MergedAssigner(model_cfg.TARGET_ASSIGNER_CONFIG, num_classes=sum(self.num_classes),
                                              no_log=self.no_log, grid_size=grid_size, pc_range=point_cloud_range,
                                              voxel_size=voxel_size)

        # self.box_n_dim = 9 if self.dataset == 'nuscenes' else 7
        # self.bev_only = True if model_cfg.MODE == "bev" else False
        shared_ch = model_cfg.PARAMETERS.shared_ch
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, shared_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(shared_ch),
            nn.ReLU(inplace=True)
        )

        self.common_heads = model_cfg.PARAMETERS.common_heads
        self.output_box_attrs = [k for k in self.common_heads]
        self.tasks = nn.ModuleList()

        for num_cls, template_box in zip(self.num_classes, self.template_boxes):
            heads = copy.deepcopy(self.common_heads)
            heads.update(
                dict(
                    num_classes=num_cls,
                    template_box=template_box,
                    pc_range=self.point_cloud_range,
                    offset_grid=self.offset_grid,
                    voxel_size=self.voxel_size
                )
            )
            self.tasks.append(
                SingleHeadDict[self.single_head](shared_ch, heads)
            )

    def _nms_gpu_3d(self, boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
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

    def _generate_offset_grid(self):
        x, y = self.grid_size[:2] // self.out_size_factor
        xmin, ymin, zmin, xmax, ymax, zmax = self.point_cloud_range

        xoffset = (xmax - xmin) / x
        yoffset = (ymax - ymin) / y

        yv, xv = torch.meshgrid([torch.arange(0, y), torch.arange(0, x)])
        yvp = (yv.float() + 0.5) * yoffset + ymin
        xvp = (xv.float() + 0.5) * xoffset + xmin

        yvc = yv.float() * yoffset + ymin
        xvc = xv.float() * xoffset + xmin

        # size (1, 2, h, w)
        self.register_buffer('offset_grid', torch.stack([xvp, yvp], dim=0)[None])
        self.register_buffer('xy_offset', torch.stack([xvc, yvc], dim=0)[None])

    def forward(self, data_dict):
        multi_head_features = []
        spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = self.shared_conv(spatial_features_2d)
        for task in self.tasks:
            multi_head_features.append(task(spatial_features_2d))

        self.forward_ret_dict['multi_head_features'] = multi_head_features
        final_feat = torch.cat([iter['final_feat'] for iter in multi_head_features] + [spatial_features_2d, ], dim=1)
        data_dict['final_feat'] = final_feat

        if self.training:
            self.forward_ret_dict['gt_dicts'] = self.target_assigner.assign_targets_cuda(data_dict['gt_boxes'])

        if not self.training and not self.predict_boxes_when_training:
            data_dict = self.generate_predicted_boxes(data_dict)
        # else:
        #     data_dict = self.generate_predicted_boxes_for_roi_head(data_dict)

        return data_dict

    def get_proper_xy(self, pred_boxes):
        tmp, res = pred_boxes[:, :2, :, :], pred_boxes[:, 2:, :, :]
        tmp = tmp + self.offset_grid
        return torch.cat([tmp, res], dim=1)

    def _reshape_corner_map(self, corner_map):
        bs, c, h, w = corner_map.size()
        return corner_map.view(bs, c // 4, 4, h, w)

    def get_loss(self, curr_epoch, **kwargs):
        tb_dict = {}
        pred_dicts = self.forward_ret_dict['multi_head_features']
        losses = []
        self.forward_ret_dict['pred_box_encoding'] = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            if self.final_layer_loss:
                pred_dict['pred_boxes'] = [pred_dict['pred_boxes'][-1], ]
                pred_dict['pred_logits'] = [pred_dict['pred_logits'][-1], ]
                pred_dict['pred_layer'] = 1

            tmp_loss = 0.0
            for i in range(pred_dict['pred_layer']):
                task_pred_boxes = self.get_proper_xy(pred_dict['pred_boxes'][i])
                bs, code, h, w = task_pred_boxes.size()
                task_pred_boxes = task_pred_boxes.permute(0, 2, 3, 1).view(bs, h * w, code)
                task_pred_logits = pred_dict['pred_logits'][i]
                _, cls, _, _ = task_pred_logits.size()
                task_pred_logits = task_pred_logits.permute(0, 2, 3, 1).view(bs, h * w, cls)
                task_pred_dicts = {
                    'pred_logits': task_pred_logits,
                    'pred_boxes': task_pred_boxes
                }

                task_gt_dicts = self.forward_ret_dict['gt_dicts'][task_id]
                layer_loss_dicts = self.set_crit(task_pred_dicts, task_gt_dicts, curr_epoch)

                tmp_loss = tmp_loss + layer_loss_dicts['loss']

            tmp_loss = tmp_loss / pred_dict['pred_layer']


            aux_loss_dict = {}

            pred_dict['center_map'] = _sigmoid(pred_dict['center_map'])
            pred_dict['corner_map'] = _sigmoid(self._reshape_corner_map(pred_dict['corner_map']))
            pred_dict['foreground_map'] = pred_dict['foreground_map']

            aux_loss_dict['loss_center'] = self.loss_center(
                pred_dict['center_map'],
                self.forward_ret_dict['gt_dicts']['center_map'][task_id]
            )
            aux_loss_dict['loss_corner'] = self.loss_corner(
                pred_dict['corner_map'],
                self.forward_ret_dict['gt_dicts']['corner_map'][task_id]
            )
            aux_loss_dict['loss_foreground'] = self.loss_foreground(
                pred_dict['foreground_map'],
                self.forward_ret_dict['gt_dicts']['foreground_map'][task_id]
            )

            for k in self.aux_loss_weights:
                tmp_loss = tmp_loss + self.aux_loss_weights[k] * aux_loss_dict[k]

            task_loss_dicts = {}
            task_loss_dicts.update(aux_loss_dict)
            task_loss_dicts['loss'] = tmp_loss

            losses.append(task_loss_dicts['loss'])

        return sum(losses), tb_dict

    @torch.no_grad()
    def generate_predicted_boxes_for_roi_head(self, data_dict):
        pred_dicts = self.forward_ret_dict['multi_head_features']

        task_box_preds = {}
        task_score_preds = {}

        k_list = self.post_cfg.k_list

        for task_id, pred_dict in enumerate(pred_dicts):
            tmp = {}
            tmp.update(pred_dict)
            _pred_boxes = self.get_proper_xy(tmp['pred_boxes'][-1])
            if self.use_focal_loss:
                _pred_score = tmp['pred_logits'][-1].sigmoid()
            else:
                _pred_score = tmp['pred_logits'][-1].softmax(2)

            _pred_score = _pred_score.flatten(2).permute(0, 2, 1)
            _pred_boxes = self.box_coder.decode_torch(_pred_boxes.flatten(2).permute(0, 2, 1))

            task_box_preds[task_id] = _pred_boxes
            task_score_preds[task_id] = _pred_score

        batch_cls_preds = []
        batch_box_preds = []

        bs = len(task_box_preds[0])
        for idx in range(bs):
            cls_offset = 1
            pred_boxes, pred_scores, pred_labels = [], [], []
            for task_id, class_name in enumerate(self.class_names):
                raw_scores = task_score_preds[task_id][idx]
                raw_boxes = task_box_preds[task_id][idx]

                cls_num = raw_scores.size(1)
                tmp_scores, tmp_cat_inds = torch.topk(raw_scores, k=k_list[task_id], dim=0)

                final_score_task, tmp_inds = torch.topk(tmp_scores.reshape(-1), k=k_list[task_id])
                final_label = (tmp_inds % cls_num) + cls_offset

                topk_boxes_cat = raw_boxes[tmp_cat_inds.reshape(-1), :]
                final_box = topk_boxes_cat[tmp_inds, :]
                raw_scores = raw_scores[tmp_cat_inds.reshape(-1), :]

                final_score = final_score_task.new_zeros((final_box.shape[0], self.total_classes))
                final_score[:, cls_offset - 1: cls_offset - 1 + cls_num] = raw_scores

                pred_boxes.append(final_box)
                pred_scores.append(final_score)
                pred_labels.append(final_label)

                cls_offset += len(class_name)

            batch_box_preds.append(torch.cat(pred_boxes))
            batch_cls_preds.append(torch.cat(pred_scores))

        data_dict['batch_cls_preds'] = torch.stack(batch_cls_preds, dim=0)
        data_dict['batch_box_preds'] = torch.stack(batch_box_preds, dim=0)
        if self.training:
            data_dict['gt_dicts'] = self.forward_ret_dict['gt_dicts']

        return data_dict

    @torch.no_grad()
    def generate_predicted_boxes(self, data_dict):
        cur_epoch = data_dict['cur_epoch']
        pred_dicts = self.forward_ret_dict['multi_head_features']

        task_box_preds = {}
        task_score_preds = {}

        k_list = self.post_cfg.k_list
        thresh_list = self.post_cfg.thresh_list
        num_queries = self.post_cfg.num_queries
        # use_nms = self.post_cfg.use_nms
        # vis_dir = getattr(self.post_cfg, 'bev_vis_dir', None)

        for task_id, pred_dict in enumerate(pred_dicts):
            tmp = {}
            tmp.update(pred_dict)
            _pred_boxes = self.get_proper_xy(tmp['pred_boxes'][-1])

            if self.use_focal_loss:
                _pred_score = tmp['pred_logits'][-1].sigmoid()
            else:
                _pred_score = tmp['pred_logits'][-1].softmax(2)

            _pred_score = _pred_score.flatten(2).permute(0, 2, 1)
            _pred_boxes = self.box_coder.decode_torch(_pred_boxes.flatten(2).permute(0, 2, 1))

            task_box_preds[task_id] = _pred_boxes
            task_score_preds[task_id] = _pred_score

        pred_dicts = []
        bs = len(task_box_preds[0])

        for idx in range(bs):
            cls_offset = 1
            final_boxes, final_scores, final_labels = [], [], []
            for task_id, class_name in enumerate(self.class_names):
                task_scores = task_score_preds[task_id][idx]
                task_boxes = task_box_preds[task_id][idx]

                cls_num = task_scores.size(1)
                topk_scores_cat, topk_inds_cat = torch.topk(task_scores, k=k_list[task_id], dim=0)
                topk_scores, topk_inds = torch.topk(topk_scores_cat.reshape(-1), k=k_list[task_id])
                topk_labels = (topk_inds % cls_num) + cls_offset
                topk_boxes_cat = task_boxes[topk_inds_cat.reshape(-1), :]
                topk_boxes = topk_boxes_cat[topk_inds, :]

                mask = topk_scores >= thresh_list[task_id]

                task_boxes = topk_boxes[mask]
                task_scores = topk_scores[mask]
                task_labels = topk_labels[mask]

                final_boxes.append(task_boxes)
                final_scores.append(task_scores)
                final_labels.append(task_labels)

                cls_offset += len(class_name)

            final_boxes = torch.cat(final_boxes)
            final_scores = torch.cat(final_scores)
            final_labels = torch.cat(final_labels)
            end = min(final_scores.size(0), num_queries)

            record_dict = {
                "pred_boxes": final_boxes[:end],
                "pred_scores": final_scores[:end],
                "pred_labels": final_labels[:end]
            }
            pred_dicts.append(record_dict)

        # import pdb; pdb.set_trace()
        data_dict['pred_dicts'] = pred_dicts
        data_dict['has_class_labels'] = True  # Force to be true
        return data_dict
