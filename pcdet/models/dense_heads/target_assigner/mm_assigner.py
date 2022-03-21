import torch
import math
from ....ops.center_ops import center_ops_cuda

class MMAssigner(object):
    def __init__(self, assigner_cfg, num_classes, no_log, grid_size, pc_range, voxel_size):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.out_size_factor = assigner_cfg.out_size_factor
        self.num_classes = num_classes
        self.tasks = assigner_cfg.tasks
        self.dense_reg = assigner_cfg.dense_reg
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.class_to_idx = assigner_cfg.mapping
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.no_log = no_log

    def limit_period(self, val, offset=0.5, period=math.pi):
        return val - math.floor(val / period + offset) * period

    def anchor_encode(self, anchor_list, boxes_dim, cats):
        """
        Args:
            boxes_dim: (N, 3)
            cats: (N)
        Returns:
            encode_dim: (N, 3)
        """
        anchors_temp = torch.tensor(anchor_list).to(boxes_dim.device)
        anchors = anchors_temp[cats]
        encode_boxes_dim = torch.log(boxes_dim / anchors)
        return encode_boxes_dim

    def z_encode(self, z_list, z_dim, cats):
        z_temp = torch.tensor(z_list).to(z_dim.device).squeeze(-1)
        zs = z_temp[cats]
        encode_z = z_dim - zs
        return encode_z

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)
        Returns:
        """
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = self.grid_size[:2] // self.out_size_factor # grid_size WxHxD feature_map_size WxH
        batch_size = gt_boxes.shape[0]
        code_size = gt_boxes.shape[2] #cls -> sin/cos
        num_classes = self.num_classes
        assert gt_boxes[:, :, -1].max().item() <= num_classes, "labels must match, found {}".format(gt_boxes[:, :, -1].max().item())

        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}

        heatmap = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)
        gt_ind = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_mask = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cat = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cnt = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)
        gt_box_encoding = torch.zeros((batch_size, num_classes, max_objs, code_size), dtype = torch.float32).to(gt_boxes.device)

        center_ops_cuda.draw_center_gpu(gt_boxes, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)

        offset = 0
        for task_id, task in enumerate(self.tasks):
            end = offset + len(task.class_names)
            heatmap_of_task = heatmap[:, offset:end, :, :]
            gt_ind_of_task = gt_ind[:, offset:end, :].reshape(batch_size, -1)
            gt_mask_of_task = gt_mask[:, offset:end, :].reshape(batch_size, -1)
            gt_cat_of_task = gt_cat[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_encoding_of_task = gt_box_encoding[:, offset:end, :, :].reshape(batch_size, -1, code_size)
            gt_ind_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
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
                # anchor regression
                cat_selected = gt_cat_merged[i, :mask_range].long()
                gt_box_encoding_merged[i, :mask_range, 3:6] = self.anchor_encode(
                    task.anchors,
                    gt_box_encoding_merged[i, :mask_range, 3:6],
                    cat_selected
                )
                gt_box_encoding_merged[i, :mask_range, 2] = self.z_encode(
                    task.z,
                    gt_box_encoding_merged[i, :mask_range, 2],
                    cat_selected
                )

            heatmaps[task_id] = heatmap_of_task
            gt_inds[task_id] = gt_ind_merged.long()
            gt_masks[task_id] = gt_mask_merged.bool()
            gt_cats[task_id] = gt_cat_merged.long()
            gt_box_encodings[task_id] = gt_box_encoding_merged

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }
        return target_dict