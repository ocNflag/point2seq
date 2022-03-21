import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

from ...utils import box_coder_utils, common_utils
from pcdet.utils import matcher
from pcdet.utils.set_crit import SetCritToken
from pcdet.models.dense_heads.e2e_seq_token_modules import OneNetSeqTokenHeadTSC
from pcdet.models.dense_heads.e2e_modules import GroundTruthProcessor

from ...ops.iou3d_nms import iou3d_nms_cuda

SingleHeadDict = {
    'OneNetSeqTokenHeadTSC': OneNetSeqTokenHeadTSC,
}


class E2ESeqTokenHead(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size,
                 point_cloud_range, predict_boxes_when_training, **kwargs):
        super().__init__()
        self.xoffset = None
        self.yoffset = None
        self.forward_ret_dict = {}
        self.model_cfg = model_cfg
        self.period = 2 * np.pi
        self.single_head = self.model_cfg.get('SingleHead', 'OneNetSeqHead')
        self.voxel_size = [model_cfg.OUT_SIZE_FACTOR * iter for iter in voxel_size]

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
        self.bin_size = [t["bin_size"] for t in model_cfg.TASKS]
        self.bin_num = [t["bin_num"] for t in model_cfg.TASKS]
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
        self.set_crit = SetCritToken(**set_crit_settings)

        gt_processor_settings = model_cfg.GT_PROCESSOR_CONFIG
        self.target_assigner = GroundTruthProcessor(
            gt_processor_cfg=gt_processor_settings
        )

        # self.box_n_dim = 9 if self.dataset == 'nuscenes' else 7
        # self.bev_only = True if model_cfg.MODE == "bev" else False

        self.common_heads = model_cfg.PARAMETERS.common_heads
        self.output_box_attrs = [k for k in self.common_heads]
        self.tasks = nn.ModuleList()

        for num_cls, template_box, bin_size, bin_num in zip(self.num_classes, self.template_boxes, self.bin_size, self.bin_num):
            heads = copy.deepcopy(self.common_heads)
            heads.update(
                dict(
                    num_classes=num_cls,
                    template_box=template_box,
                    bin_size=bin_size,
                    bin_num=bin_num,
                    pc_range=self.point_cloud_range,
                    offset_grid=self.offset_grid,
                    voxel_size=self.voxel_size
                )
            )
            self.tasks.append(
                OneNetSeqTokenHeadTSC(
                    self.in_channels,
                    heads,
                )
            )

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

    def _generate_offset_grid(self):
        x, y = self.grid_size[:2] // self.out_size_factor
        xmin, ymin, zmin, xmax, ymax, zmax = self.point_cloud_range

        xoffset = (xmax - xmin) / x
        yoffset = (ymax - ymin) / y

        yv, xv = torch.meshgrid([torch.arange(0, y), torch.arange(0, x)])
        yv = (yv.float() + 0.5) * yoffset + ymin
        xv = (xv.float() + 0.5) * xoffset + xmin

        # size (1, 2, h, w)
        self.register_buffer('offset_grid', torch.stack([xv, yv], dim=0)[None])
        self.register_buffer('xy_offset', torch.Tensor([xoffset, yoffset]).view(1, 2, 1, 1))

    def forward(self, data_dict):
        multi_head_features = []
        spatial_features_2d = data_dict['spatial_features_2d']
        for task in self.tasks:
            multi_head_features.append(task(spatial_features_2d))

        self.forward_ret_dict['multi_head_features'] = multi_head_features

        if self.training:
            self.forward_ret_dict['gt_dicts'] = self.target_assigner.process(data_dict['gt_boxes'])

        if not self.training and not self.predict_boxes_when_training:
            data_dict = self.generate_predicted_boxes(data_dict)
        # else:
        #     data_dict = self.generate_predicted_boxes_for_roi_head(data_dict)

        return data_dict

    def get_loss_box(self, pred_dict):
        rlt = self.box_coder.decode_matcher(
            pred_dict['pred_box_bins'], self.offset_grid, pred_dict['anchor_zwlh'],
            pred_dict['bin_size'], pred_dict['bin_num']
        )
        return rlt



    def get_pred_box(self, pred_dict):
        rlt = self.box_coder.decode_torch(
            pred_dict['pred_box_bins'], self.offset_grid, pred_dict['anchor_zwlh'],
            pred_dict['bin_size'], pred_dict['bin_num']
        )
        return rlt

    def get_loss(self, curr_epoch, **kwargs):
        tb_dict = {}
        pred_dicts = self.forward_ret_dict['multi_head_features']
        center_loss = []
        self.forward_ret_dict['pred_box_encoding'] = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            task_pred_boxes = self.get_loss_box(pred_dict)
            bs, h, w, code = task_pred_boxes.size()
            task_pred_boxes = task_pred_boxes.view(bs, h * w, code)
            task_pred_logits = pred_dict['pred_logits']
            _, cls, _, _ = task_pred_logits.size()
            task_pred_logits = task_pred_logits.permute(0, 2, 3, 1).view(bs, h * w, cls)

            pred_box_bins = [iter.permute(0, 2, 3, 1).contiguous().view(bs, h * w, -1) for iter in pred_dict['pred_box_bins']]
            anchor_xy = self.offset_grid.permute(0, 2, 3, 1).repeat(bs, 1, 1, 1).contiguous().view(bs, h * w, -1)

            task_pred_dicts = {
                'pred_box_bins': pred_box_bins,
                'pred_logits': task_pred_logits,
                'pred_boxes': task_pred_boxes,
                'anchor_xy': anchor_xy,
                'anchor_zwlh': pred_dict['anchor_zwlh'],
                'bin_size': pred_dict['bin_size'],
                'bin_num': pred_dict['bin_num'],
            }

            task_gt_dicts = self.forward_ret_dict['gt_dicts'][task_id]
            task_loss_dicts = self.set_crit(task_pred_dicts, task_gt_dicts, curr_epoch)
            loss = task_loss_dicts['loss']

            tb_key = 'task_' + str(task_id) + '/'
            tb_dict.update({
                tb_key + 'loss_x': task_loss_dicts['loc_loss_elem'][0].item(),
                tb_key + 'loss_y': task_loss_dicts['loc_loss_elem'][1].item(),
                tb_key + 'loss_z': task_loss_dicts['loc_loss_elem'][2].item(),
                tb_key + 'loss_w': task_loss_dicts['loc_loss_elem'][3].item(),
                tb_key + 'loss_l': task_loss_dicts['loc_loss_elem'][4].item(),
                tb_key + 'loss_h': task_loss_dicts['loc_loss_elem'][5].item(),
                tb_key + 'loss_sin': task_loss_dicts['loc_loss_elem'][6].item(),
                tb_key + 'loss_cos': task_loss_dicts['loc_loss_elem'][7].item(),
                tb_key + 'loss_ce': task_loss_dicts['loss_ce'],
                tb_key + 'loss_bbox': task_loss_dicts['loss_bbox'],
            })


            center_loss.append(loss)

        return sum(center_loss), tb_dict


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
            task_pred_boxes = self.get_pred_box(pred_dict)
            bs, h, w, code = task_pred_boxes.size()
            _pred_boxes = task_pred_boxes.view(bs, h * w, code)

            if self.use_focal_loss:
                _pred_score = tmp['pred_logits'].sigmoid()
            else:
                _pred_score = tmp['pred_logits'].softmax(2)

            _pred_score = _pred_score.flatten(2).permute(0, 2, 1)
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
                tmp_scores, tmp_cat_inds = torch.topk(task_scores, k=k_list[task_id], dim=0)

                task_scores, tmp_inds = torch.topk(tmp_scores.reshape(-1), k=k_list[task_id])
                task_labels = (tmp_inds % cls_num) + cls_offset

                topk_boxes_cat = task_boxes[tmp_cat_inds.reshape(-1), :]
                task_boxes = topk_boxes_cat[tmp_inds, :]

                # used for 2 stage network
                mask = task_scores >= thresh_list[task_id]
                task_scores = task_scores[mask]
                task_boxes = task_boxes[mask]
                task_labels = task_labels[mask]


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
