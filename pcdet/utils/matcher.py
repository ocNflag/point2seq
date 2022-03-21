import torch
import numpy as np
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

class CornerMatcher(nn.Module):
    def __init__(
            self,
            box_coder,
            losses,
            weight_dict,
            use_focal_loss,
            code_weights,
            period=None,
            **kwargs
    ):
        super().__init__()
        self.losses = losses
        self.box_coder = box_coder  # notice here we also need encode ground truth boxes
        self.weight_dict = weight_dict
        self.period = period

        if 'loss_dir' in self.losses:
            self.losses.remove('loss_dir')

        self.register_buffer('code_weights', torch.Tensor(code_weights))
        self.use_focal_loss = use_focal_loss
        self.loss_dict = {
            'loss_ce': self.loss_ce,
            'loss_bbox': self.loss_bbox,
        }

    def _preprocess(self, pred_dicts, gt_dicts):
        '''
        Args:
            pred_dicts: {
                pred_logits: list(Tensor), Tensor with size (box_num, cls_num)
                pred_boxes: list(Tensor), Tensor with size (box_num, 10)
            }
            gt_dicts: {
                gt_class: list(Tensor), Tensor with size (box_num)
                gt_boxes: list(Tensor), Tensor with size (box_num, 10)
            }

        Returns:

        '''

        examples = defaultdict(lambda: None)
        examples['pred_boxes'] = self.box_coder.pred_to_corner(pred_dicts['pred_boxes'])
        examples['gt_boxes'] = self.box_coder.gt_to_corner(gt_dicts['gt_boxes'])

        pred_logits = pred_dicts['pred_logits']
        if self.use_focal_loss:
            pred_logits = pred_logits.sigmoid()
        else:
            pred_logits = pred_logits.softmax(dim=-1)
        examples['pred_logits'] = pred_logits
        examples['gt_classes'] = gt_dicts['gt_classes']
        examples['batchsize'] = pred_dicts['pred_boxes'].size(0)

        return examples

    def loss_ce(self, pred_logits, gt_classes, **kwargs):
        loss = pred_logits[:, gt_classes]
        return loss


    def loss_bbox(self, pred_boxes, gt_boxes, **kwargs):
        _, corner_num, *_ = pred_boxes.size()
        loss = 0.0
        for i in range(corner_num):
            tmp = torch.exp(-torch.cdist(pred_boxes[:, i, :], gt_boxes[:, i, :], p=2))
            loss += tmp ** 0.5
        return loss

    def get_loss(self, example):
        rlt = {}
        for k in self.losses:
            if k not in self.weight_dict:
                continue
            rlt[k] = self.loss_dict[k](**example)
        return rlt

    def _get_per_scene_example(self, examples, idx):
        rlt = {}

        rlt['pred_logits'] = examples['pred_logits'][idx]
        rlt['gt_classes'] = examples['gt_classes'][idx]

        rlt['pred_boxes'] = examples['pred_boxes'][idx]
        rlt['gt_boxes'] = examples['gt_boxes'][idx]


        return rlt

    @torch.no_grad()
    def forward(self, pred_dicts, gt_dicts):
        rlt = {}
        examples = self._preprocess(pred_dicts, gt_dicts)
        indices = []

        for i in range(examples['batchsize']):

            if examples["gt_classes"][i].size(0) == 0:
                indices.append(([], []))
                size = examples['pred_logits'][i].size(0)
                continue

            example = self._get_per_scene_example(examples, i)
            loss_val_dict = self.get_loss(example)

            loss = -1.0
            for k in self.losses:
                if k not in self.weight_dict:
                    continue
                tmp = loss_val_dict[k] ** self.weight_dict[k]
                loss = loss * tmp

            ind = linear_sum_assignment(loss.cpu())
            indices.append(ind)


        rlt['inds'] = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                       indices]
        return rlt


class TimeMatcher(nn.Module):
    def __init__(
            self,
            box_coder,
            losses,
            weight_dict,
            use_focal_loss,
            code_weights,
            period=None,
            iou_th=-1,
            **kwargs
    ):
        super().__init__()
        self.losses = losses
        self.box_coder = box_coder  # notice here we also need encode ground truth boxes
        self.weight_dict = weight_dict
        self.period = period
        self.iou_th = iou_th
        self.register_buffer('code_weights', torch.Tensor(code_weights))
        self.use_focal_loss = use_focal_loss
        self.loss_dict = {
            'loss_ce': self.loss_ce,
            'loss_bbox': self.loss_bbox,
            'loss_iou': self.loss_iou,
        }

    def _preprocess(self, pred_dicts, gt_dicts):
        '''
        Args:
            pred_dicts: {
                pred_logits: list(Tensor), Tensor with size (box_num, cls_num)
                pred_boxes: list(Tensor), Tensor with size (box_num, 10)
            }
            gt_dicts: {
                gt_class: list(Tensor), Tensor with size (box_num)
                gt_boxes: list(Tensor), Tensor with size (box_num, 10)
            }

        Returns:

        '''

        examples = defaultdict(lambda: None)
        examples['pred_boxes'] = pred_dicts['pred_boxes']
        examples['gt_boxes'] = self.box_coder.encode(gt_dicts['gt_boxes'])

        if self.iou_th > 0.0:
            examples['pred_boxes_iou'] = self.box_coder.decode_torch(pred_dicts['pred_boxes'])
            examples['gt_boxes_iou'] = gt_dicts['gt_boxes']

        pred_logits = pred_dicts['pred_logits']
        if self.use_focal_loss:
            pred_logits = pred_logits.sigmoid()
        else:
            pred_logits = pred_logits.softmax(dim=-1)

        examples['pred_logits'] = pred_logits
        examples['gt_classes'] = gt_dicts['gt_classes']
        examples['batchsize'] = pred_dicts['pred_boxes'].size(0)
        examples['code_weights'] = pred_dicts.get('code_weights', None)

        return examples

    def loss_ce(self, pred_logits, gt_classes, **kwargs):
        loss = pred_logits[:, gt_classes]
        loss[loss == float("Inf")] = 0
        loss[loss != loss] = 0
        return loss

    def loss_bbox(self, pred_boxes, gt_boxes, code_weights=None, **kwargs):
        if code_weights is None:
            code_weights = self.code_weights
        weighted_preds = torch.einsum('bc,c->bc', pred_boxes, code_weights)
        weighted_gts = torch.einsum('bc,c->bc', gt_boxes, code_weights)
        loss = torch.exp(-torch.cdist(weighted_preds, weighted_gts, p=1))
        loss[loss == float("Inf")] = 0
        loss[loss != loss] = 0
        return loss

    def iou_thresh(self, pred_boxes_iou, gt_boxes_iou, **kwargs):
        tmp = boxes_iou3d_gpu(pred_boxes_iou, gt_boxes_iou)
        tmp[tmp != tmp] = 0
        tmp[tmp == float("Inf")] = 0
        thresh_mask = (tmp > self.iou_th).float()
        return thresh_mask


    def loss_iou(self, pred_boxes_iou, gt_boxes_iou, **kwargs):
        tmp = boxes_iou3d_gpu(pred_boxes_iou, gt_boxes_iou)
        tmp[tmp == float("Inf")] = 0
        tmp[tmp != tmp] = 0
        loss = torch.exp(tmp) / np.e
        return loss

    def get_loss(self, example):
        rlt = {}

        if self.iou_th > 0.0:
            rlt['iou_thresh_mask'] = self.iou_thresh(**example)

        for k in self.losses:
            if k not in self.weight_dict:
                continue
            rlt[k] = self.loss_dict[k](**example)
        return rlt

    def _get_per_scene_example(self, examples, idx):
        rlt = {}

        rlt['pred_logits'] = examples['pred_logits'][idx]
        rlt['gt_classes'] = examples['gt_classes'][idx]

        rlt['pred_boxes'] = examples['pred_boxes'][idx]
        rlt['gt_boxes'] = examples['gt_boxes'][idx]
        rlt['code_weights'] = examples.get('code_weights', None)

        if self.iou_th > 0.0:
            rlt['pred_boxes_iou'] = examples['pred_boxes_iou'][idx]
            rlt['gt_boxes_iou'] = examples['gt_boxes_iou'][idx]

        return rlt

    @torch.no_grad()
    def forward(self, pred_dicts, gt_dicts):
        rlt = {}
        examples = self._preprocess(pred_dicts, gt_dicts)
        indices = []

        for i in range(examples['batchsize']):

            if examples["gt_classes"][i].size(0) == 0:
                indices.append(([], []))
                continue

            example = self._get_per_scene_example(examples, i)
            loss_val_dict = self.get_loss(example)

            loss = -1.0
            if self.iou_th > 0.0:
                loss = loss * loss_val_dict['iou_thresh_mask']
            for k in self.losses:
                if k not in self.weight_dict:
                    continue
                tmp = loss_val_dict[k] ** self.weight_dict[k]
                loss = loss * tmp

            ind = linear_sum_assignment(loss.cpu())
            indices.append(ind)

        rlt['inds'] = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return rlt


class TokenMatcher(nn.Module):
    def __init__(
            self,
            box_coder,
            losses,
            weight_dict,
            use_focal_loss,
            code_weights,
            period=None,
            iou_th=-1,
            **kwargs
    ):
        super().__init__()
        self.losses = losses
        self.box_coder = box_coder  # notice here we also need encode ground truth boxes
        self.weight_dict = weight_dict
        self.period = period
        self.iou_th = iou_th
        self.register_buffer('code_weights', torch.Tensor(code_weights))
        self.use_focal_loss = use_focal_loss
        self.loss_dict = {
            'loss_ce': self.loss_ce,
            'loss_bbox': self.loss_bbox,
            'loss_iou': self.loss_iou,
        }

    def _preprocess(self, pred_dicts, gt_dicts):
        '''
        Args:
            pred_dicts: {
                pred_logits: list(Tensor), Tensor with size (box_num, cls_num)
                pred_boxes: list(Tensor), Tensor with size (box_num, 10)
            }
            gt_dicts: {
                gt_class: list(Tensor), Tensor with size (box_num)
                gt_boxes: list(Tensor), Tensor with size (box_num, 10)
            }

        Returns:

        '''

        examples = defaultdict(lambda: None)
        examples['pred_boxes'] = pred_dicts['pred_boxes']
        examples['gt_boxes'] = self.box_coder.encode(gt_dicts['gt_boxes'])

        examples['pred_boxes_iou'] = self.box_coder.decode_torch(
            pred_dicts['pred_boxes'],
            pred_dicts['anchor_xy'],
            pred_dicts['anchor_zwlh'],
        )
        examples['gt_boxes_iou'] = gt_dicts['gt_boxes']

        pred_logits = pred_dicts['pred_logits']
        if self.use_focal_loss:
            pred_logits = pred_logits.sigmoid()
        else:
            pred_logits = pred_logits.softmax(dim=-1)

        examples['pred_logits'] = pred_logits
        examples['gt_classes'] = gt_dicts['gt_classes']
        examples['batchsize'] = pred_dicts['pred_boxes'].size(0)

        return examples

    def loss_ce(self, pred_logits, gt_classes, **kwargs):
        loss = pred_logits[:, gt_classes]
        loss[loss == float("Inf")] = 0
        loss[loss != loss] = 0
        return loss

    def loss_bbox(self, pred_boxes, gt_boxes, **kwargs):
        weighted_preds = torch.einsum('bc,c->bc', pred_boxes, self.code_weights)
        weighted_gts = torch.einsum('bc,c->bc', gt_boxes, self.code_weights)
        loss = torch.exp(-torch.cdist(weighted_preds, weighted_gts, p=1))
        loss[loss == float("Inf")] = 0
        loss[loss != loss] = 0
        return loss

    def iou_thresh(self, pred_boxes_iou, gt_boxes_iou, **kwargs):
        tmp = boxes_iou3d_gpu(pred_boxes_iou, gt_boxes_iou)
        tmp[tmp != tmp] = 0
        tmp[tmp == float("Inf")] = 0
        thresh_mask = (tmp > self.iou_th).float()
        return thresh_mask


    def loss_iou(self, pred_boxes_iou, gt_boxes_iou, **kwargs):
        tmp = boxes_iou3d_gpu(pred_boxes_iou, gt_boxes_iou)
        tmp[tmp == float("Inf")] = 0
        tmp[tmp != tmp] = 0
        loss = torch.exp(tmp) / np.e
        return loss

    def get_loss(self, example):
        rlt = {}

        if self.iou_th > 0.0:
            rlt['iou_thresh_mask'] = self.iou_thresh(**example)

        for k in self.losses:
            if k not in self.weight_dict:
                continue
            rlt[k] = self.loss_dict[k](**example)
        return rlt

    def _get_per_scene_example(self, examples, idx):
        rlt = {}

        rlt['pred_logits'] = examples['pred_logits'][idx]
        rlt['gt_classes'] = examples['gt_classes'][idx]

        rlt['pred_boxes'] = examples['pred_boxes'][idx]
        rlt['gt_boxes'] = examples['gt_boxes'][idx]

        rlt['pred_boxes_iou'] = examples['pred_boxes_iou'][idx]
        rlt['gt_boxes_iou'] = examples['gt_boxes_iou'][idx]

        return rlt

    @torch.no_grad()
    def forward(self, pred_dicts, gt_dicts):
        rlt = {}
        examples = self._preprocess(pred_dicts, gt_dicts)
        indices = []

        for i in range(examples['batchsize']):

            if examples["gt_classes"][i].size(0) == 0:
                indices.append(([], []))
                continue

            example = self._get_per_scene_example(examples, i)
            loss_val_dict = self.get_loss(example)

            loss = -1.0
            if self.iou_th > 0.0:
                loss = loss * loss_val_dict['iou_thresh_mask']
            for k in self.losses:
                if k not in self.weight_dict:
                    continue
                tmp = loss_val_dict[k] ** self.weight_dict[k]
                loss = loss * tmp

            ind = linear_sum_assignment(loss.cpu())
            indices.append(ind)

        rlt['inds'] = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return rlt


class ROIMatcher(nn.Module):
    def __init__(
            self,
            losses,
            weight_dict,
            use_focal_loss,
            code_weights,
            period=None,
            iou_th=-1,
            iou_th_reg=0.55,
            assign_only=False,
            use_reg_idx=False,
            **kwargs
    ):
        super().__init__()
        self.losses = losses
        self.assign_only = assign_only
        self.use_reg_idx = use_reg_idx
        self.weight_dict = weight_dict
        self.period = period
        self.iou_th = iou_th
        self.iou_th_reg = iou_th_reg
        self.register_buffer('code_weights', torch.Tensor(code_weights))
        self.use_focal_loss = use_focal_loss
        self.loss_dict = {
            'loss_ce': self.loss_ce,
            'loss_bbox': self.loss_bbox,
            'loss_iou': self.loss_iou,
        }

    def _preprocess(self, pred_dicts, gt_dicts):
        '''
        Args:
            pred_dicts: {
                pred_logits: list(Tensor), Tensor with size (box_num, cls_num)
                pred_boxes: list(Tensor), Tensor with size (box_num, 10)
            }
            gt_dicts: {
                gt_class: list(Tensor), Tensor with size (box_num)
                gt_boxes: list(Tensor), Tensor with size (box_num, 10)
            }

        Returns:

        '''

        examples = defaultdict(lambda: None)
        examples['rois'] = pred_dicts['rois']
        examples['roi_scores'] = pred_dicts['roi_scores']
        examples['pred_boxes'] = pred_dicts['pred_boxes']
        examples['gt_boxes'] = gt_dicts['gt_boxes']

        pred_logits = pred_dicts['pred_logits']
        if self.use_focal_loss:
            pred_logits = pred_logits.sigmoid()
        else:
            pred_logits = pred_logits.softmax(dim=-1)

        examples['pred_logits'] = pred_logits
        examples['gt_classes'] = gt_dicts['gt_classes']
        examples['batchsize'] = pred_dicts['pred_boxes'].size(0)

        return examples

    def loss_ce(self, pred_logits, gt_classes, **kwargs):
        loss = pred_logits[:, gt_classes]
        return loss

    def loss_bbox(self, pred_boxes, gt_boxes, **kwargs):
        weighted_preds = torch.einsum('bc,c->bc', pred_boxes, self.code_weights)
        weighted_gts = torch.einsum('bc,c->bc', gt_boxes, self.code_weights)
        loss = torch.exp(-torch.cdist(weighted_preds, weighted_gts, p=1))
        return loss

    def iou_thresh(self, pred_boxes, gt_boxes, **kwargs):
        tmp = boxes_iou3d_gpu(pred_boxes, gt_boxes)
        tmp[tmp != tmp] = 0
        tmp[tmp == float("Inf")] = 0
        thresh_mask = (tmp > self.iou_th).float()
        return thresh_mask, tmp

    def roi_thresh(self, rois, gt_boxes, **kwargs):
        tmp = boxes_iou3d_gpu(rois, gt_boxes)
        tmp[tmp != tmp] = 0
        tmp[tmp == float("Inf")] = 0
        thresh_mask = (tmp > self.iou_th).float()
        return thresh_mask, tmp

    def loss_iou(self, pred_boxes, gt_boxes, **kwargs):
        tmp = boxes_iou3d_gpu(pred_boxes, gt_boxes)
        tmp[tmp != tmp] = 0
        tmp[tmp == float("Inf")] = 0
        loss = torch.exp(tmp) / np.e
        return loss

    def get_loss(self, example):
        rlt = {}

        if self.iou_th > 0.0:
            rlt['iou_thresh_mask'], rlt['iou'] = self.iou_thresh(**example)
            _, rlt['roi_iou'] = self.roi_thresh(**example)

        for k in self.losses:
            if k not in self.weight_dict:
                continue
            rlt[k] = self.loss_dict[k](**example)
        return rlt

    def _get_per_scene_example(self, examples, idx):
        rlt = {}

        rlt['rois'] = examples['rois'][idx]
        rlt['roi_scores'] = examples['roi_scores'][idx]
        rlt['pred_logits'] = examples['pred_logits'][idx]
        rlt['gt_classes'] = examples['gt_classes'][idx]

        rlt['pred_boxes'] = examples['pred_boxes'][idx]
        rlt['gt_boxes'] = examples['gt_boxes'][idx]

        # rlt['pred_boxes_iou'] = examples['pred_boxes_iou'][idx]
        # rlt['gt_boxes_iou'] = examples['gt_boxes_iou'][idx]

        return rlt

    @torch.no_grad()
    def forward(self, pred_dicts, gt_dicts):
        rlt = {}
        examples = self._preprocess(pred_dicts, gt_dicts)
        indices = []
        reg_indices = []

        for i in range(examples['batchsize']):

            if examples["gt_classes"][i].size(0) == 0:
                indices.append(([], []))
                reg_indices.append(([], []))
                continue

            example = self._get_per_scene_example(examples, i)
            loss_val_dict = self.get_loss(example)

            loss = -1.0
            # if self.iou_th > 0.0:
            #     loss = loss * loss_val_dict['iou_thresh_mask']
            for k in self.losses:
                if k not in self.weight_dict:
                    continue
                tmp = loss_val_dict[k] ** self.weight_dict[k]
                loss = loss * tmp

            if self.assign_only:
                ind0 = torch.tensor(range(loss_val_dict['iou'].shape[0]))
                _, ind1 = loss_val_dict['iou'].cpu().max(dim=1)
                ind = (ind0.numpy(), ind1.numpy())
            else:
                ind = linear_sum_assignment(loss.cpu())

            iou_mask = loss_val_dict['iou'][torch.Tensor(ind[0]).long(), torch.Tensor(ind[1]).long()] > self.iou_th
            ind = (ind[0][iou_mask.cpu()], ind[1][iou_mask.cpu()])
            indices.append(ind)

            r_ind0 = np.setdiff1d(np.array(range(tmp.shape[0])), ind[0])
            tmp_iou_max = torch.argmax(loss_val_dict['iou'], dim=1)
            r_ind1 = tmp_iou_max[r_ind0].cpu().clone().numpy()
            iou_mask_reg = loss_val_dict['iou'][torch.Tensor(r_ind0).long(), torch.Tensor(r_ind1).long()] > self.iou_th_reg
            r_ind = (
                np.concatenate((ind[0], r_ind0[iou_mask_reg.cpu()]), axis=0),
                np.concatenate((ind[1], r_ind1[iou_mask_reg.cpu()]), axis=0)
            )
            reg_indices.append(r_ind)


        rlt['inds'] = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]

        if self.use_reg_idx:
            rlt['reg_inds'] = [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in reg_indices
            ]
        else:
            rlt['reg_inds'] = rlt['inds']

        return rlt