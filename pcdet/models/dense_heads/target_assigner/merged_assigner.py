import torch
import math
from ....ops.center_ops import center_ops_cuda
import numpy as np
from pcdet.utils.common_utils import limit_period_torch

class MergedAssigner(object):
    def __init__(self, assigner_cfg, num_classes, no_log, grid_size,
                 pc_range, voxel_size):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.out_size_factor = assigner_cfg.out_size_factor
        self.num_classes = num_classes
        self.tasks = assigner_cfg.tasks
        self.dense_reg = assigner_cfg.dense_reg
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self.fg_enlarge_ratio = assigner_cfg.fg_enlarge_ratio
        # self.num_dir_bins = num_dir_bins
        # self.use_dir_classifier = use_dir_classifier
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.class_to_idx = assigner_cfg.mapping
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.no_log = no_log
        self.period = 2 * np.pi

        # if self.use_dir_classifier and (self.num_dir_bins >= 1):
        #     self.period = self.period / self.num_dir_bins


    def limit_period_wrapper(self, input, offset=0, dim=6):
        prev, r, rem = input[..., :dim], input[..., dim:dim + 1], input[..., dim + 1:]
        r = limit_period_torch(r, offset=offset, period=self.period)
        return torch.cat([prev, r, rem], dim=-1)


    def get_direction_target(self, gt_boxes, one_hot=False):
        if not self.use_dir_classifier:
            return None

        gt_rot = gt_boxes[..., 6]
        offset_rot = limit_period_torch(gt_rot, 0, 2 * np.pi)

        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / self.num_dir_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=self.num_dir_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), self.num_dir_bins, dtype=gt_boxes.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets

        return dir_cls_targets


    @staticmethod
    def get_corners(gt_boxes):
        """
        This function return bev corner info in world coord
        M stands for max objs in this instance
        Args:
            gt_boxes: (B, M, C + cls) (x, y, z, w, l, h, theta, (velox, veloy))

        Returns:
            gt_corners (B, M, 4, 2)
        """

        def rotz(t):
            """
            Args:
                t: a tensor with size (B, M, 1)
            Returns:
                rlt a tensor with size (B, M, 2, 2)
                [c, -s]
                [s,  c]
            """
            c = torch.cos(t)
            s = torch.sin(t)
            b, q, *_ = t.size()
            rlt = torch.cat([c, -s, s, c], dim=-1)
            rlt = rlt.view((b, q, 2, 2))
            return rlt

        x, y, z, w, l, h, theta, *_ = torch.split(gt_boxes, 1, dim=-1)
        rot_mat = rotz(theta)
        corner_x = torch.cat([-w / 2, -w / 2, w / 2, w / 2], dim=-1)
        corner_y = torch.cat([l / 2, -l / 2, l / 2, -l / 2], dim=-1)
        local_corner = torch.stack([corner_x, corner_y], dim=-1)  # (B, M, 4, 2)
        rot_corner = torch.einsum('bmhw,bmcw->bmch', rot_mat, local_corner)  # (B, M, 4, 2)
        center = torch.stack([x, y], dim=-1)  # (B, M, 1, 2)
        corner = rot_corner + center

        return corner.contiguous()

    def gt_process(self, gt_boxes):
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
            # gt_dicts[task_id]['gt_dirs'] = []

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
                # dir_targets_of_tasks = []
                classes_of_tasks = []
                class_offset = 0

                for class_name in task.class_names:
                    class_idx = self.class_to_idx[class_name]
                    class_mask = (iter_gt_classes == class_idx)
                    _boxes = iter_box[class_mask]
                    # _dir_targets = self.get_direction_target(_boxes)
                    _boxes = self.limit_period_wrapper(_boxes)
                    _class = _boxes.new_full((_boxes.shape[0],), class_offset).long()
                    boxes_of_tasks.append(_boxes)
                    # dir_targets_of_tasks.append(_dir_targets)
                    classes_of_tasks.append(_class)
                    class_offset += 1

                task_boxes = torch.cat(boxes_of_tasks, dim=0)
                task_classes = torch.cat(classes_of_tasks, dim=0)
                gt_dicts[task_id]['gt_boxes'].append(task_boxes)
                gt_dicts[task_id]['gt_classes'].append(task_classes)

                # if self.use_dir_classifier:
                #     task_dirs = torch.cat(dir_targets_of_tasks, dim=0)
                #     gt_dicts[task_id]['gt_dirs'].append(task_dirs)

        return gt_dicts

    def assign_targets_cuda(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)
        Returns:
        """
        gt_dicts = self.gt_process(gt_boxes)
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = self.grid_size[:2] // self.out_size_factor  # grid_size WxHxD feature_map_size WxH
        batch_size = gt_boxes.shape[0]
        code_size = gt_boxes.shape[2]  # cls -> sin/cos
        num_classes = self.num_classes
        assert gt_boxes[:, :, -1].max().item() <= num_classes, "labels must match, found {}".format(
            gt_boxes[:, :, -1].max().item())

        center_maps = {}
        corner_maps = {}
        foreground_maps = {}

        gt_corners = self.get_corners(gt_boxes)

        center_map = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]),
                                 dtype=torch.float32).to(gt_boxes.device)
        corner_map = torch.zeros((batch_size, num_classes, 4, feature_map_size[1], feature_map_size[0]),
                                 dtype=torch.float32).to(gt_boxes.device)
        foreground_map = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]),
                                     dtype=torch.float32).to(gt_boxes.device)

        gt_ind = torch.zeros((batch_size, num_classes, max_objs), dtype=torch.int32).to(gt_boxes.device)
        # unfolded index range from 0 to H * W of center in feature map
        gt_mask = torch.zeros((batch_size, num_classes, max_objs), dtype=torch.int32).to(gt_boxes.device)
        gt_cat = torch.zeros((batch_size, num_classes, max_objs), dtype=torch.int32).to(gt_boxes.device)
        gt_cnt = torch.zeros((batch_size, num_classes), dtype=torch.int32).to(gt_boxes.device)
        gt_box_encoding = torch.zeros((batch_size, num_classes, max_objs, code_size),
                                      dtype=torch.float32).to(gt_boxes.device)

        center_ops_cuda.draw_all_gpu(gt_boxes, gt_corners, center_map, corner_map, foreground_map, gt_ind, gt_mask,
                                     gt_cat, gt_box_encoding, gt_cnt, self._min_radius, self.voxel_size[0],
                                     self.voxel_size[1], self.pc_range[0], self.pc_range[1], self.out_size_factor,
                                     self.gaussian_overlap, self.fg_enlarge_ratio)

        offset = 0
        for task_id, task in enumerate(self.tasks):
            end = offset + len(task.class_names)

            center_map_of_task = center_map[:, offset:end]
            corner_map_of_task = corner_map[:, offset:end]
            foreground_map_of_task = foreground_map[:, offset:end]
            offset = end

            center_maps[task_id] = center_map_of_task
            corner_maps[task_id] = corner_map_of_task
            foreground_maps[task_id] = foreground_map_of_task

        target_dict = {
            'center_map': center_maps,
            'corner_map': corner_maps,
            'foreground_map': foreground_maps
        }
        target_dict.update(gt_dicts)
        return target_dict
