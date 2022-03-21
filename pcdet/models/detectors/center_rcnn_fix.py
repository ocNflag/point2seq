import torch
from collections import OrderedDict

from .detector3d_template import Detector3DTemplate

class CenterRCNNFix(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # Load first-stage parameters
        # self.checkpoints = torch.load('/home/xuhang/maojiageng/PVDet/output/nuscenes_models/pretrain/checkpoint_epoch_20.pth')['model_state']
        # self.checkpoints = torch.load('/home/work/user-job-dir/PCDet/checkpoints/checkpoint_epoch_20.pth')['model_state']
        self.checkpoints = torch.load(model_cfg.PRE_PATH)['model_state']
        model_dict = self.state_dict()
        new_dict = {k:v for k,v in self.checkpoints.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        # Freeze first-stage parameters
        self.vfe.requires_grad = False
        self.backbone_3d.requires_grad = False
        self.map_to_bev_module.requires_grad = False
        self.backbone_2d.requires_grad = False
        self.dense_head.requires_grad = False

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        if self.training:
            batch_dict = self.roi_head(batch_dict, targets_dict)
        else:
            batch_dict = self.roi_head(batch_dict, {})

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
