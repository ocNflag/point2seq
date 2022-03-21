import torch
import math
from ....ops.center_ops import center_ops_cuda


class BundleAssigner(object):
    def __init__(self, assigner_cfg, num_classes, no_log, grid_size, pc_range, voxel_size):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.out_size_factor = assigner_cfg.out_size_factor
        self.num_classes = num_classes
        self.tasks = assigner_cfg.tasks
        self.dense_reg = assigner_cfg.dense_reg
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self.fg_enlarge_ratio = assigner_cfg.fg_enlarge_ratio
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.class_to_idx = assigner_cfg.mapping
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.no_log = no_log

    def limit_period(self, val, offset=0.5, period=math.pi):
        return val - math.floor(val / period + offset) * period

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

    def assign_targets_cuda(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)
        Returns:
        """
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
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}
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
            gt_ind_of_task = gt_ind[:, offset:end, :].reshape(batch_size, -1)
            gt_mask_of_task = gt_mask[:, offset:end, :].reshape(batch_size, -1)
            gt_cat_of_task = gt_cat[:, offset:end, :].reshape(batch_size, -1) - (offset + 1)  # cat begin from 1
            gt_box_encoding_of_task = gt_box_encoding[:, offset:end, :, :].reshape(batch_size, -1, code_size)
            gt_ind_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_box_encoding_merged = torch.zeros((batch_size, max_objs, code_size), dtype=torch.float32).to(gt_boxes.device)
            offset = end

            for i in range(batch_size):
                mask = gt_mask_of_task[i] == 1
                mask_range = mask.sum().item()
                assert mask_range <= max_objs
                gt_mask_merged[i, :mask_range] = gt_mask_of_task[i, mask]
                gt_ind_merged[i, :mask_range] = gt_ind_of_task[i, mask]
                gt_cat_merged[i, :mask_range] = gt_cat_of_task[i, mask]
                gt_box_encoding_merged[i, :mask_range, :] = gt_box_encoding_of_task[i, mask, :]
                # only perform log on valid gt_box_encoding
                if not self.no_log:
                    gt_box_encoding_merged[i, :mask_range, 3:6] = torch.log(gt_box_encoding_merged[i, :mask_range, 3:6])  # log(wlh)

            center_maps[task_id] = center_map_of_task
            corner_maps[task_id] = corner_map_of_task
            foreground_maps[task_id] = foreground_map_of_task
            gt_inds[task_id] = gt_ind_merged.long()
            gt_masks[task_id] = gt_mask_merged.bool()
            gt_cats[task_id] = gt_cat_merged.long()
            gt_box_encodings[task_id] = gt_box_encoding_merged

        target_dict = {
            'center_map': center_maps,
            'corner_map': corner_maps,
            'foreground_map': foreground_maps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }
        return target_dict
