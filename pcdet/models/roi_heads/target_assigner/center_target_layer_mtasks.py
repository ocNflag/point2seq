import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils

class CenterTargetLayerMTasks(nn.Module):
    def __init__(self, roi_sampler_cfg, task_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg
        self.tasks = task_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M * ntask, 7 + C)
                gt_of_rois: (B, M * ntask, 7 + C)
                gt_iou_of_rois: (B, M * ntask)
                roi_scores: (B, M * ntask)
                roi_labels: (B, M * ntask)
                reg_valid_mask: (B, M * ntask)
                rcnn_cls_labels: (B, M * ntask)
        """
        rois_list, gt_of_rois_list, gt_dist_of_rois_list, roi_scores_list, roi_labels_list, reg_valid_mask_list, rcnn_cls_labels_list = [],[],[],[],[],[],[]
        for task_id, task in enumerate(self.tasks):
            batch_rois, batch_gt_of_rois, batch_roi_dist, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
                batch_dict = batch_dict, task = task
            )
            # regression valid mask
            reg_valid_mask = (batch_roi_dist <= self.roi_sampler_cfg.REG_FG_DIST).long()

            # classification label
            assert self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_dist'
            dist_bg_thresh = self.roi_sampler_cfg.CLS_BG_DIST
            dist_fg_thresh = self.roi_sampler_cfg.CLS_FG_DIST
            fg_mask = batch_roi_dist <= dist_fg_thresh
            bg_mask = batch_roi_dist > dist_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            # inverse scores !!!
            batch_cls_labels[interval_mask] = (dist_bg_thresh - batch_roi_dist[interval_mask]) / (dist_bg_thresh - dist_fg_thresh)

            rois_list.append(batch_rois)
            gt_of_rois_list.append(batch_gt_of_rois)
            gt_dist_of_rois_list.append(batch_roi_dist)
            roi_scores_list.append(batch_roi_scores)
            roi_labels_list.append(batch_roi_labels)
            reg_valid_mask_list.append(reg_valid_mask)
            rcnn_cls_labels_list.append(batch_cls_labels)

        roi_labels_list = torch.cat(roi_labels_list, dim=1)
        task_mask = (roi_labels_list > 0).float()

        targets_dict = {'rois': torch.cat(rois_list, dim = 1), 'gt_of_rois': torch.cat(gt_of_rois_list, dim = 1),
                        'gt_dist_of_rois': torch.cat(gt_dist_of_rois_list, dim = 1),
                        'roi_scores': torch.cat(roi_scores_list, dim = 1), 'roi_labels': roi_labels_list,
                        'reg_valid_mask': torch.cat(reg_valid_mask_list, dim = 1),
                        'rcnn_cls_labels': torch.cat(rcnn_cls_labels_list, dim = 1),
                        'task_mask': task_mask
                        }

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict, task):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']

        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_dist = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            roi_flag = (cur_roi[:, 3:6].sum(dim = 1) != 0) # wlh != 0 valid roi
            cur_roi = cur_roi[roi_flag]
            cur_roi_scores = cur_roi_scores[roi_flag]
            cur_roi_labels = cur_roi_labels[roi_flag]

            # select rois and gts for this task
            roi_mask = cur_roi_labels.new_zeros(cur_roi_labels.shape, dtype=bool)
            for cls_id in task['class_ids']:
                roi_mask |= (cur_roi_labels == cls_id)
            gt_mask = cur_gt.new_zeros(cur_gt.shape[0], dtype=bool)
            for cls_id in task['class_ids']:
                gt_mask |= (cur_gt[:, -1] == cls_id)

            # task does not exist, skip it
            if roi_mask.sum().item() == 0 or gt_mask.sum().item() == 0:
                continue

            cur_gt = cur_gt[gt_mask]
            cur_roi = cur_roi[roi_mask]
            cur_roi_scores = cur_roi_scores[roi_mask]
            cur_roi_labels = cur_roi_labels[roi_mask]

            cdist = iou3d_nms_utils.boxes_dist_torch(cur_roi[:, 0:7], cur_gt[:, 0:7])  # (M, N)
            min_dist, gt_assignment = torch.min(cdist, dim=1)

            sampled_inds = self.subsample_rois(min_dist = min_dist)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_dist[index] = min_dist[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_dist, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, min_dist):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_dist = max(self.roi_sampler_cfg.REG_FG_DIST, self.roi_sampler_cfg.CLS_FG_DIST)
        bg_dist = self.roi_sampler_cfg.CLS_BG_DIST_LO

        fg_inds = (min_dist <= fg_dist).nonzero().view(-1)
        easy_bg_inds = (min_dist > bg_dist).nonzero().view(-1)
        hard_bg_inds = ((min_dist > fg_dist) & (min_dist <= bg_dist)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(min_dist).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(min_dist).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds.new_zeros(0)

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('min distance:(min=%f, max=%f)' % (min_dist.min().item(), min_dist.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds
