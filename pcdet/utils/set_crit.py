from collections import defaultdict

import torch
import numpy as np
from pcdet.utils import common_utils
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from pcdet.utils.loss_utils import E2ESigmoidFocalClassificationLoss, SmoothL1Loss, IOU3DLoss


def is_distributed():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()


def label_to_one_hot(label, pred):
    label = label.unsqueeze(-1)
    bs, querys, cls = pred.size()
    one_hot = torch.full((bs, querys, cls + 1), 0, dtype=torch.float32, device=pred.device)
    one_hot.scatter_(dim=-1, index=label, value=1.0)
    return one_hot[..., 1:]


class SetCritROI(nn.Module):
    def __init__(self, matcher, weight_dict, losses, sigma, box_coder, code_weights,
                 gamma=2.0, alpha=0.25, use_focal_loss=False, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.use_focal_loss = use_focal_loss
        self.register_buffer('code_weights', torch.Tensor(code_weights))

        self.loss_map = {
            'loss_ce': self.loss_labels,
            'loss_bbox': self.loss_boxes,
            'loss_iou': self.loss_iou,
        }

        self.box_coder = box_coder
        if self.use_focal_loss:
            self.cls_loss = E2ESigmoidFocalClassificationLoss(gamma=gamma, alpha=alpha, reduction='sum')
        else:
            self.cls_loss = nn.BCELoss(reduction='sum')
        self.reg_loss = SmoothL1Loss(sigma=sigma)
        self.iou_loss = IOU3DLoss()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_gt_of_rois(self, gt_boxes, rois):
        # gt boxes (n, code_size)
        roi_center = rois[:, 0:3]
        roi_ry = rois[:, 6] % (2 * np.pi)
        gt_boxes[:, 0:3] = gt_boxes[:, 0:3] - roi_center

        n, c = gt_boxes.shape
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_boxes.view(-1, 1, gt_boxes.shape[-1]), angle=-roi_ry.view(-1)
        ).view(n, c)

        if not self.box_coder.encode_angle_by_sincos:
            gt_of_rois[:, 6] = gt_of_rois[:,  6] - roi_ry
            heading_label = gt_of_rois[:, 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
            heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

            gt_of_rois[:, 6] = heading_label

        rois_anchor = rois.clone().detach()
        rois_anchor[:, 0:3] = 0

        if not self.box_coder.encode_angle_by_sincos:
            rois_anchor[:, 6] = 0

        gt_of_rois = self.box_coder.encode_torch(gt_of_rois, rois_anchor)
        return gt_of_rois

    def _preprocess(self, pred_dicts, gt_dicts, cur_epoch, **kwargs):
        conds = defaultdict(lambda: None)
        conds['cur_epoch'] = cur_epoch

        matcher_dict = self.matcher(pred_dicts, gt_dicts)
        indices = matcher_dict['inds']
        idx = self._get_src_permutation_idx(indices)

        reg_indices = matcher_dict['reg_inds']
        reg_idx = self._get_src_permutation_idx(reg_indices)

        rois = pred_dicts['rois'][reg_idx]
        rcnn_regs = pred_dicts['rcnn_regs'][reg_idx]

        gt_boxes = torch.cat([iter[i] for iter, (_, i) in zip(gt_dicts['gt_boxes'], reg_indices)], dim=0)
        gt_of_rois = self._get_gt_of_rois(gt_boxes, rois)

        conds['gt_of_rois'] = gt_of_rois
        conds['rcnn_regs'] = rcnn_regs

        # preprocess for cls loss
        pred_logits = pred_dicts['pred_logits']
        gt_classes_pos = torch.cat([t[j] for t, (_, j) in zip(gt_dicts['gt_classes'], indices)]) + 1
        gt_classes = torch.full(pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_pos

        if not self.use_focal_loss:
            pred_logits = torch.sigmoid(pred_logits)
        conds['pred_logits'] = pred_logits
        conds['gt_classes'] = gt_classes


        num_boxes = idx[0].shape[0]
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        conds['num_boxes'] = num_boxes

        num_reg_boxes = rcnn_regs.shape[0]
        num_reg_boxes = torch.as_tensor([num_reg_boxes], dtype=torch.float, device=pred_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_reg_boxes)

        num_reg_boxes = torch.clamp(num_reg_boxes / get_world_size(), min=1).item()
        conds['num_reg_boxes'] = num_reg_boxes

        return conds

    def loss_iou(self, gt_boxes, pred_boxes, num_reg_boxes, **kwargs):
        loss_iou = self.iou_loss(gt_boxes, pred_boxes) / num_reg_boxes
        losses = {
            'loss_iou': loss_iou,
        }
        return losses

    def loss_boxes(self, gt_of_rois, rcnn_regs, num_reg_boxes, **kwargs):
        # tmp_delta = torch.einsum('bc,c->bc', tmp_delta, self.code_weights)

        loss_bbox_loc = self.reg_loss(gt_of_rois - rcnn_regs)
        loss_bbox = loss_bbox_loc.sum() / num_reg_boxes
        loss_bbox_loc = loss_bbox_loc.detach().clone() / num_reg_boxes

        losses = {
            'loss_bbox': loss_bbox,
            'loc_loss_elem': loss_bbox_loc
        }
        return losses

    def loss_labels(self, pred_logits, gt_classes, num_boxes, **kwargs):
        """
        Classification loss (NLL)
        """
        target_one_hot = label_to_one_hot(gt_classes, pred_logits)
        loss_ce = self.cls_loss(pred_logits, target_one_hot) / num_boxes

        losses = {'loss_ce': loss_ce}
        return losses

    def get_loss(self, loss, conds):
        assert loss in self.loss_map, f'do you really want to compute {loss} loss?'
        return self.loss_map[loss](**conds)

    def forward(self, pred_dicts, gt_dicts, curr_epoch):
        conds = self._preprocess(pred_dicts, gt_dicts, curr_epoch)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, conds))

        total = sum([losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict])
        losses['loss'] = total
        return losses


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses, sigma, box_coder, code_weights,
                 gamma=2.0, alpha=0.25, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.register_buffer('code_weights', torch.Tensor(code_weights))

        self.loss_map = {
            'loss_ce': self.loss_labels,
            'loss_bbox': self.loss_boxes,
            'loss_iou': self.loss_iou,
        }

        self.box_coder = box_coder
        self.focal_loss = E2ESigmoidFocalClassificationLoss(gamma=gamma, alpha=alpha, reduction='sum')
        self.reg_loss = SmoothL1Loss(sigma=sigma)
        self.iou_loss = IOU3DLoss()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _preprocess(self, pred_dicts, gt_dicts, cur_epoch, **kwargs):
        conds = defaultdict(lambda: None)
        conds['cur_epoch'] = cur_epoch

        matcher_dict = self.matcher(pred_dicts, gt_dicts)
        indices = matcher_dict['inds']
        idx = self._get_src_permutation_idx(indices)

        pred_boxes = pred_dicts['pred_boxes'][idx]
        gt_boxes = torch.cat([iter[i] for iter, (_, i) in zip(gt_dicts['gt_boxes'], indices)], dim=0)

        if self.box_coder is not None:
            tmp_delta = self.box_coder.get_delta(gt_boxes=gt_boxes, preds=pred_boxes)
        else:
            tmp_delta = gt_boxes - pred_boxes

        tmp_delta = torch.einsum('bc,c->bc', tmp_delta, self.code_weights)
        conds['delta'] = tmp_delta
        conds['gt_boxes'] = gt_boxes[:, :7]
        conds['pred_boxes'] = self.box_coder.decode_torch(preds=pred_boxes)[:, :7]

        # preprocess for cls loss
        pred_logits = pred_dicts['pred_logits']
        gt_classes_pos = torch.cat([t[j] for t, (_, j) in zip(gt_dicts['gt_classes'], indices)]) + 1
        gt_classes = torch.full(pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_pos

        conds['pred_logits'] = pred_logits
        conds['gt_classes'] = gt_classes

        # preprocess for card loss
        target_len = torch.as_tensor([len(iter) for iter in gt_dicts['gt_classes']], device=pred_logits.device)
        conds['target_len'] = target_len

        num_boxes = sum([len(iter) for iter in gt_dicts['gt_classes']])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        conds['num_boxes'] = num_boxes

        return conds

    def loss_iou(self, gt_boxes, pred_boxes, num_boxes, **kwargs):
        loss_iou = self.iou_loss(gt_boxes, pred_boxes) / num_boxes
        losses = {
            'loss_iou': loss_iou,
        }
        return losses

    def loss_boxes(self, delta, num_boxes, **kwargs):
        loss_bbox_loc = self.reg_loss(delta)
        loss_bbox = loss_bbox_loc.sum() / num_boxes
        loss_bbox_loc = loss_bbox_loc.detach().clone() / num_boxes

        losses = {
            'loss_bbox': loss_bbox,
            'loc_loss_elem': loss_bbox_loc
        }
        return losses

    def loss_labels(self, pred_logits, gt_classes, num_boxes, **kwargs):
        """
        Classification loss (NLL)
        """
        target_one_hot = label_to_one_hot(gt_classes, pred_logits)
        loss_ce = self.focal_loss(pred_logits, target_one_hot) / num_boxes

        losses = {'loss_ce': loss_ce}
        return losses

    def get_loss(self, loss, conds):
        assert loss in self.loss_map, f'do you really want to compute {loss} loss?'
        return self.loss_map[loss](**conds)

    def forward(self, pred_dicts, gt_dicts, curr_epoch):
        conds = self._preprocess(pred_dicts, gt_dicts, curr_epoch)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, conds))

        total = sum([losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict])
        losses['loss'] = total
        return losses



class SetCritToken(nn.Module):
    def __init__(self, matcher, weight_dict, losses, sigma, box_coder,
                 gamma=2.0, alpha=0.25, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        # self.register_buffer('code_weights', torch.Tensor(code_weights))

        self.loss_map = {
            'loss_ce': self.loss_labels,
            'loss_bbox': self.loss_boxes,
        }

        self.box_coder = box_coder
        self.focal_loss = E2ESigmoidFocalClassificationLoss(gamma=gamma, alpha=alpha, reduction='sum')
        self.reg_loss = nn.CrossEntropyLoss()
        self.iou_loss = IOU3DLoss()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _preprocess(self, pred_dicts, gt_dicts, cur_epoch, **kwargs):
        conds = defaultdict(lambda: None)
        conds['cur_epoch'] = cur_epoch

        matcher_dict = self.matcher(pred_dicts, gt_dicts)
        indices = matcher_dict['inds']
        idx = self._get_src_permutation_idx(indices)
        anchor_xy = pred_dicts['anchor_xy'][idx]


        pred_boxes = []
        for iter in pred_dicts['pred_box_bins']:
            tmp = iter[idx]
            pred_boxes.append(tmp)


        gt_boxes = torch.cat([iter[i] for iter, (_, i) in zip(gt_dicts['gt_boxes'], indices)], dim=0)

        gt_boxes = self.box_coder.get_tokenized_labels(
            gt_boxes, anchor_xy, pred_dicts['anchor_zwlh'], pred_dicts['bin_size'], pred_dicts['bin_num']
        )

        conds['gt_boxes'] = gt_boxes
        conds['pred_boxes'] = pred_boxes

        # preprocess for cls loss
        pred_logits = pred_dicts['pred_logits']
        gt_classes_pos = torch.cat([t[j] for t, (_, j) in zip(gt_dicts['gt_classes'], indices)]) + 1
        gt_classes = torch.full(pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_pos

        conds['pred_logits'] = pred_logits
        conds['gt_classes'] = gt_classes

        # preprocess for card loss
        target_len = torch.as_tensor([len(iter) for iter in gt_dicts['gt_classes']], device=pred_logits.device)
        conds['target_len'] = target_len

        num_boxes = sum([len(iter) for iter in gt_dicts['gt_classes']])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        conds['num_boxes'] = num_boxes

        return conds

    def loss_boxes(self, gt_boxes, pred_boxes, num_boxes, **kwargs):
        loss_bbox = 0.0
        loss_bbox_loc = []
        for pred, gt in zip(pred_boxes, gt_boxes):
            tmp = self.reg_loss(pred, gt.squeeze().long()) / num_boxes
            loss_bbox_loc.append(tmp)
            loss_bbox = loss_bbox + tmp
        loss_bbox_loc = torch.stack(loss_bbox_loc).view(-1).detach().clone()

        losses = {
            'loss_bbox': loss_bbox,
            'loc_loss_elem': loss_bbox_loc
        }
        return losses

    def loss_labels(self, pred_logits, gt_classes, num_boxes, **kwargs):
        """
        Classification loss (NLL)
        """
        target_one_hot = label_to_one_hot(gt_classes, pred_logits)
        loss_ce = self.focal_loss(pred_logits, target_one_hot) / num_boxes

        losses = {'loss_ce': loss_ce}
        return losses

    def get_loss(self, loss, conds):
        assert loss in self.loss_map, f'do you really want to compute {loss} loss?'
        return self.loss_map[loss](**conds)

    def forward(self, pred_dicts, gt_dicts, curr_epoch):
        conds = self._preprocess(pred_dicts, gt_dicts, curr_epoch)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, conds))

        total = sum([losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict])
        losses['loss'] = total
        return losses
