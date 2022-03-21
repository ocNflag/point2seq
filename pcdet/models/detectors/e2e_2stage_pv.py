import os

import torch
import torch.nn as nn
from .detector3d_template import Detector3DTemplate


class E2E2StagePV(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.freeze_layer = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'backbone_2d',
            'dense_head'
        ]
        self.module_list = self.build_networks()
        self.pretrained_path = model_cfg.PRETRAINED_CKPT
        self.load_params()
        self.freeze()

    def freeze(self):
        for name in self.freeze_layer:
            sub_module = getattr(self, name, None)
            if sub_module:
                for para in sub_module.parameters():
                    para.requires_grad = False

                for m in sub_module.modules():
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                        m.eval()

    def forward(self, batch_dict):
        batch_dict['cur_epoch'] = self.cur_epoch

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

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

        # loss_rpn, tb_dict = self.dense_head.get_loss(self.cur_epoch)
        loss_point, tb_dict = self.point_head.get_loss({})
        loss_rcnn, tb_dict = self.roi_head.get_loss(self.cur_epoch)

        tb_dict = {
            'loss_point': loss_point.item(),
            'loss_rcnn': loss_rcnn.item(),
            **tb_dict
        }

        loss = loss_rcnn  + loss_point
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        pred_dicts = batch_dict['pred_dicts']

        recall_dict = {}
        batch_size = batch_dict['batch_size']
        for index in range(batch_size):
            recall_dict = self.generate_recall_record(
                box_preds=pred_dicts[index]['pred_boxes'],
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=[0.3, 0.5, 0.7]
            )
        return pred_dicts, recall_dict

    def load_params(self, to_cpu=False):
        if not os.path.isfile(self.pretrained_path):
            raise FileNotFoundError

        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(self.pretrained_path, map_location=loc_type)

        model_dict = self.state_dict()
        pretrained_dict = checkpoint['model_state']
        pretrained_dict = { k: v for k, v in pretrained_dict.items() if k in model_dict }
        model_dict.update(pretrained_dict)

        self.load_state_dict(model_dict)

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])

        return
