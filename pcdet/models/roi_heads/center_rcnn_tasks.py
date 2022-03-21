import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils, loss_utils
from .roi_head_template import RoIHeadTemplate
from .target_assigner.center_target_layer_mtasks import CenterTargetLayerMTasks

class CenterRCNNTasks(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.tasks = self.model_cfg.TASKS
        self.num_tasks = len(self.tasks)
        self.num_rois = self.model_cfg.TARGET_CONFIG.ROI_PER_IMAGE

        self.proposal_target_layer = CenterTargetLayerMTasks(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG, task_cfg=self.model_cfg.TASKS)

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        c_out = sum([x[-1] for x in mlps])


        self.roi_grid_pool_layers = nn.ModuleList()
        self.shared_fc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()

        for task in self.tasks:
            self.roi_grid_pool_layers.append(
                pointnet2_stack_modules.StackSAModuleMSG(
                    radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
                    nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
                    mlps=copy.deepcopy(mlps),
                    use_xyz=True,
                    pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
                )
            )

            shared_fc_list = []
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            self.shared_fc_layers.append(nn.Sequential(*shared_fc_list))
            self.cls_layers.append(
                self.make_fc_layers(
                    input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
                )
            )
            self.reg_layers.append(
                self.make_fc_layers(
                    input_channels=pre_channel,
                    output_channels=self.box_coder.code_size * self.num_class,
                    fc_list=self.model_cfg.REG_FC
                )
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
        for i in range(len(self.tasks)):
            nn.init.normal_(self.reg_layers[i][-1].weight, mean=0, std=0.001)

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.reshape(-1, rois.shape[-1])
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

    def forward(self, batch_dict, targets_dict):
        if self.training:
            batch_dict = self.train_rcnn(batch_dict, targets_dict)
        else:
            batch_dict = self.test_rcnn(batch_dict, targets_dict)
        return batch_dict

    def train_rcnn(self, batch_dict, targets_dict):
        rois = batch_dict['rois'] # (B, ntask * N, code_size)
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)
        batch_size, _, code_size = rois.shape
        num_grid_points = self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3
        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        rois = rois.reshape(batch_size, self.num_tasks, self.num_rois, code_size)

        rcnn_cls = []
        rcnn_reg = []
        for task_id, task in enumerate(self.tasks):
            roi_single = rois[:, task_id, :, :]

            global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
                roi_single, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            )  # (BxN, 6x6x6, 3)
            global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)
            new_xyz = global_roi_grid_points.view(-1, 3)
            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
            pooled_points, pooled_features = self.roi_grid_pool_layers[task_id](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features.contiguous(),
            )  # (M1 + M2 ..., C)
            num_features = pooled_features.shape[-1]
            pooled_features = pooled_features.reshape(-1, num_grid_points, num_features)  # (BN, 6x6x6, C)
            pooled_features = pooled_features.transpose(1, 2).contiguous()
            pooled_features = pooled_features.reshape(-1, num_features * num_grid_points, 1)

            shared_features = self.shared_fc_layers[task_id](pooled_features)
            rcnn_cls_single = self.cls_layers[task_id](shared_features).transpose(1, 2).contiguous().squeeze(
                dim=1)  # (BN, 1 or 2)
            rcnn_reg_single = self.reg_layers[task_id](shared_features).transpose(1, 2).contiguous().squeeze(
                dim=1)  # (BN, C)

            rcnn_cls_single = rcnn_cls_single.reshape(batch_size, self.num_rois, 1)
            rcnn_reg_single = rcnn_reg_single.reshape(batch_size, self.num_rois, rcnn_reg_single.shape[-1])
            rcnn_cls.append(rcnn_cls_single)
            rcnn_reg.append(rcnn_reg_single)

        rcnn_cls = torch.cat(rcnn_cls, dim = 1)
        rcnn_reg = torch.cat(rcnn_reg, dim = 1)
        targets_dict['rcnn_cls'] = rcnn_cls
        targets_dict['rcnn_reg'] = rcnn_reg

        self.forward_ret_dict = targets_dict

        return batch_dict

    def test_rcnn(self, batch_dict, targets_dict):
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'] # (B, N, code_size)
        roi_labels = batch_dict['roi_labels'] # (B, N)
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)
        num_rois = rois.shape[1]
        num_grid_points = self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        rcnn_cls = rois.new_zeros((batch_size * num_rois, 1))
        rcnn_reg = rois.new_zeros((batch_size * num_rois, rois.shape[-1]))
        rcnn_dummy_cls, rcnn_dummy_reg = [], []

        for task_id, task in enumerate(self.tasks):
            mask = roi_labels.new_zeros(roi_labels.shape, dtype=bool)
            for cls_id in task['class_ids']:
                mask |= (roi_labels == cls_id)
            roi_cnt = mask.sum(1) # (B) num_rois selected in each batch
            mask = mask.reshape(-1) # (BxN)

            if mask.sum().item() == 0: # this task does not exist, set dummy to avoid bugs
                new_xyz = torch.zeros((batch_size * num_grid_points, 3)).to(mask.device)
                new_xyz_batch_cnt = torch.ones(batch_size).to(mask.device) * num_grid_points
                pooled_points, pooled_features = self.roi_grid_pool_layers[task_id](
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=new_xyz,
                    new_xyz_batch_cnt=new_xyz_batch_cnt.int(),
                    features=point_features.contiguous(),
                )  # (M1 + M2 ..., C)
                num_features = pooled_features.shape[-1]
                pooled_features = pooled_features.reshape(-1, num_grid_points, num_features)  # (M, 6x6x6, C)
                pooled_features = pooled_features.transpose(1, 2).contiguous()
                pooled_features = pooled_features.reshape(-1, num_features * num_grid_points, 1)

                shared_features = self.shared_fc_layers[task_id](pooled_features)
                rcnn_cls_single = self.cls_layers[task_id](shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
                rcnn_reg_single = self.reg_layers[task_id](shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

                rcnn_dummy_cls.append(rcnn_cls_single)
                rcnn_dummy_reg.append(rcnn_reg_single)

            else:
                assert global_roi_grid_points.shape[0] == batch_size * num_rois
                new_xyz = global_roi_grid_points[mask].reshape(-1, 3)
                new_xyz_batch_cnt = roi_cnt.int() * num_grid_points
                pooled_points, pooled_features = self.roi_grid_pool_layers[task_id](
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=new_xyz,
                    new_xyz_batch_cnt=new_xyz_batch_cnt,
                    features=point_features.contiguous(),
                )  # (M1 + M2 ..., C)

                num_features = pooled_features.shape[-1]
                pooled_features = pooled_features.reshape(-1, num_grid_points, num_features)  # (M, 6x6x6, C)
                pooled_features = pooled_features.transpose(1, 2).contiguous()
                pooled_features = pooled_features.reshape(-1, num_features * num_grid_points, 1)

                shared_features = self.shared_fc_layers[task_id](pooled_features)
                rcnn_cls_single = self.cls_layers[task_id](shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (M, 1 or 2)
                rcnn_reg_single = self.reg_layers[task_id](shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (M, C)

                rcnn_cls[mask] = rcnn_cls_single
                rcnn_reg[mask] = rcnn_reg_single

        dummy_flag = rcnn_dummy_cls or rcnn_dummy_reg
        if dummy_flag:
            rcnn_dummy_cls = torch.cat(rcnn_dummy_cls, dim = 0)
            rcnn_dummy_reg = torch.cat(rcnn_dummy_reg, dim = 0)

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
        )
        batch_dict['batch_cls_preds'] = batch_cls_preds
        batch_dict['batch_box_preds'] = batch_box_preds
        batch_dict['cls_preds_normalized'] = False

        return batch_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        task_mask = forward_ret_dict['task_mask'].view(-1)
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        # skip empty task
        reg_valid_mask = reg_valid_mask * task_mask

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']

            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        task_mask = forward_ret_dict['task_mask'].view(-1)

        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()

            cls_valid_mask = cls_valid_mask * task_mask

            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()

            cls_valid_mask = cls_valid_mask * task_mask

            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']

        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict
