import torch
import torch.nn as nn
import copy
import numpy as np
import numba
from pcdet.models.dense_heads.fusion_modules import DCNFusionHead, NaiveFusionHead
from pcdet.models.dense_heads.utils import _sigmoid
from ...ops.iou3d_nms import iou3d_nms_cuda
from ...ops.center_ops import center_ops_cuda

from ..model_utils import centernet_box_utils
from ...utils import loss_utils
from .target_assigner.bundle_assigner import BundleAssigner


class FusionHead(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):

        super().__init__()
        self.model_cfg = model_cfg
        self.post_cfg = model_cfg.TEST_CONFIG
        self.in_channels = input_channels
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training

        self.num_classes = [len(t["class_names"]) for t in model_cfg.TASKS]
        self.class_names = [t["class_names"] for t in model_cfg.TASKS]

        self.weight_dict = model_cfg.LOSS_CONFIG.weight_dict
        self.code_weights = model_cfg.LOSS_CONFIG.code_weights

        self.dataset = model_cfg.DATASET
        self.box_n_dim = 9 if self.dataset == 'nuscenes' else 7

        self.encode_background_as_zeros = True
        self.use_sigmoid_score = True
        self.no_log = False
        self.use_direction_classifier = False
        self.bev_only = True if model_cfg.MODE == "bev" else False

        share_conv_channel = model_cfg.PARAMETERS.share_conv_channel
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.common_heads = model_cfg.PARAMETERS.common_heads
        self.init_bias = model_cfg.PARAMETERS.init_bias
        self.tasks = nn.ModuleList()
        self.use_dcn = model_cfg.USE_DCN

        for num_cls in self.num_classes:
            heads = copy.deepcopy(self.common_heads)
            if self.use_dcn in ['V1', 'V2']:
                self.tasks.append(
                    DCNFusionHead(share_conv_channel, num_cls, heads, dcn_version=self.use_dcn, bn=True,
                                  init_bias=self.init_bias, final_kernel=3)
                )

            else:
                self.tasks.append(
                    NaiveFusionHead(share_conv_channel, num_cls, heads, bn=True, init_bias=self.init_bias, final_kernel=3)
                )

        self.target_assigner = BundleAssigner(model_cfg.TARGET_ASSIGNER_CONFIG, num_classes=sum(self.num_classes),
                                              no_log=self.no_log, grid_size=grid_size, pc_range=point_cloud_range,
                                              voxel_size=voxel_size)

        self.forward_ret_dict = {}
        self.build_losses()

    def assign_targets(self, gt_boxes):
        """
        note that cornermap will be a (B, num_cls, 4, H, W) feat map
        """
        targets_dict = self.target_assigner.assign_targets_cuda(gt_boxes)
        return targets_dict

    def forward(self, data_dict):
        multi_head_features = []
        spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = self.shared_conv(spatial_features_2d)
        for task in self.tasks:
            multi_head_features.append(task(spatial_features_2d))

        self.forward_ret_dict['multi_head_features'] = multi_head_features

        if self.training:
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            data_dict = self.generate_predicted_boxes(data_dict)

        return data_dict

    def build_losses(self):
        self.add_module('crit_fusion', loss_utils.CenterNetFocalLoss())
        self.add_module('crit_center', loss_utils.CenterNetFocalLoss())
        self.add_module('crit_corner', loss_utils.CenterNetFocalLoss())
        self.add_module('crit_foreground', loss_utils.ForegroundFocalLoss())
        self.add_module('crit_reg', loss_utils.CenterNetRegLoss())
        return

    def get_pred_box_encoding(self, pred_dict):
        if self.dataset == 'nuscenes':
            pred_box_encoding = torch.cat([pred_dict['reg'], pred_dict['height'], pred_dict['dim'], pred_dict['rot'],
                                           pred_dict['vel']], dim=1).contiguous()
        else:
            pred_box_encoding = torch.cat([pred_dict['reg'], pred_dict['height'], pred_dict['dim'], pred_dict['rot']],
                                          dim=1).contiguous()
        return pred_box_encoding

    def _reshape_corner_map(self, corner_map):
        bs, c, h, w = corner_map.size()
        return corner_map.view(bs, c // 4, 4, h, w)

    def get_loss(self):
        tb_dict = {}
        pred_dicts = self.forward_ret_dict['multi_head_features']
        center_loss = []
        self.forward_ret_dict['pred_box_encoding'] = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            # process map
            loss_dict = {}

            pred_dict['fusion_map'] = _sigmoid(pred_dict['fusion_map'])
            pred_dict['center_map'] = _sigmoid(pred_dict['center_map'])
            pred_dict['corner_map'] = _sigmoid(self._reshape_corner_map(pred_dict['corner_map']))
            pred_dict['foreground_map'] = pred_dict['foreground_map']

            # get cls loss
            loss_dict['crit_fusion'] = self.crit_fusion(pred_dict['fusion_map'],
                                                        self.forward_ret_dict['center_map'][task_id])
            loss_dict['crit_center'] = self.crit_center(pred_dict['center_map'],
                                                        self.forward_ret_dict['center_map'][task_id])
            loss_dict['crit_corner'] = self.crit_corner(pred_dict['corner_map'],
                                                        self.forward_ret_dict['corner_map'][task_id])
            loss_dict['crit_foreground'] = self.crit_foreground(pred_dict['foreground_map'],
                                                                self.forward_ret_dict['foreground_map'][task_id])

            target_box_encoding = self.forward_ret_dict['box_encoding'][task_id]
            # nuscense encoding format [x, y, z, w, l, h, sinr, cosr, vx, vy]
            pred_box_encoding = self.get_pred_box_encoding(pred_dict)  # (B, 10, H, W)

            self.forward_ret_dict['pred_box_encoding'][task_id] = pred_box_encoding

            box_loss = self.crit_reg(pred_box_encoding, self.forward_ret_dict['mask'][task_id],
                                     self.forward_ret_dict['ind'][task_id], target_box_encoding)

            loss_dict['crit_reg'] = (box_loss * box_loss.new_tensor(self.code_weights)).sum()

            loss = 0.0
            for k in self.weight_dict:
                loss = loss + (self.weight_dict[k] * loss_dict[k])

            tb_key = 'task_' + str(task_id) + '/'
            tb_dict.update({
                tb_key + 'loss': loss.item(),
                tb_key + 'fusion_loss': loss_dict['crit_fusion'].item(),
                tb_key + 'center_loss': loss_dict['crit_center'].item(),
                tb_key + 'corner_loss': loss_dict['crit_corner'].item(),
                tb_key + 'foreground_loss': loss_dict['crit_foreground'].item(),
                tb_key + 'reg_loss': loss_dict['crit_reg'].item(),
                tb_key + 'x_loss': box_loss[0].item(),
                tb_key + 'y_loss': box_loss[1].item(),
                tb_key + 'z_loss': box_loss[2].item(),
                tb_key + 'w_loss': box_loss[3].item(),
                tb_key + 'l_loss': box_loss[4].item(),
                tb_key + 'h_loss': box_loss[5].item(),
                tb_key + 'sin_r_loss': box_loss[6].item(),
                tb_key + 'cos_r_loss': box_loss[7].item(),
                tb_key + 'num_positive': self.forward_ret_dict['mask'][task_id].float().sum(),
            })
            center_loss.append(loss)

        return sum(center_loss), tb_dict

    def _double_flip_process(self, pred_dict, batch_size):
        rlt = {}
        for k in pred_dict.keys():
            # transform the prediction map back to their original coordinate befor flipping
            # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
            # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is
            # X and Y flip pointcloud(x=-x, y=-y).
            # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
            # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
            # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
            _, C, H, W = pred_dict[k].shape
            pred_dict[k] = pred_dict[k].reshape(int(batch_size), 4, C, H, W)
            pred_dict[k][:, 1] = torch.flip(pred_dict[k][:, 1], dims=[2])
            pred_dict[k][:, 2] = torch.flip(pred_dict[k][:, 2], dims=[3])
            pred_dict[k][:, 3] = torch.flip(pred_dict[k][:, 3], dims=[2, 3])

        # batch_hm = pred_dict['hm'].sigmoid_() inplace may cause errors
        batch_center = pred_dict['fusion_map'].sigmoid()
        batch_reg = pred_dict['reg']
        batch_height = pred_dict['height']
        if not self.no_log:
            batch_dim = torch.exp(pred_dict['dim'])
        else:
            batch_dim = pred_dict['dim']

        batch_center = batch_center.mean(dim=1)
        batch_height = batch_height.mean(dim=1)
        batch_dim = batch_dim.mean(dim=1)

        # y = -y reg_y = 1-reg_y
        batch_reg[:, 1, 1] = 1 - batch_reg[:, 1, 1]
        batch_reg[:, 2, 0] = 1 - batch_reg[:, 2, 0]
        batch_reg[:, 3, 0] = 1 - batch_reg[:, 3, 0]
        batch_reg[:, 3, 1] = 1 - batch_reg[:, 3, 1]
        batch_reg = batch_reg.mean(dim=1)

        batch_sin = pred_dict['rot'][:, :, 0:1]
        batch_cos = pred_dict['rot'][:, :, 1:2]

        batch_sin[:, 1] = -batch_sin[:, 1]
        batch_cos[:, 2] = -batch_cos[:, 2]
        batch_sin[:, 3] = -batch_sin[:, 3]
        batch_cos[:, 3] = -batch_cos[:, 3]

        batch_cos = batch_cos.mean(dim=1)
        batch_sin = batch_sin.mean(dim=1)

        rlt['center'] = batch_center
        # rlt['corner'] = batch_corner
        # rlt['foreground'] = batch_foreground
        rlt['sinr'] = batch_sin
        rlt['cosr'] = batch_cos
        rlt['height'] = batch_height
        rlt['dim'] = batch_dim
        rlt['reg'] = batch_reg
        rlt['dir_preds'] = [None] * batch_size


        if self.dataset == 'nuscenes':
            batch_vel = pred_dict['vel']
            # flip vy
            batch_vel[:, 1, 1] = - batch_vel[:, 1, 1]
            # flip vx
            batch_vel[:, 2, 0] = - batch_vel[:, 2, 0]
            batch_vel[:, 3] = - batch_vel[:, 3]
            batch_vel = batch_vel.mean(dim=1)
            rlt['vel'] = batch_vel

        return rlt

    # collect topk feats
    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    @numba.jit(nopython=True)
    def circle_nms(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        scores = dets[:, 2]
        order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
        ndets = dets.shape[0]
        suppressed = np.zeros((ndets), dtype=np.int32)
        keep = []
        for _i in range(ndets):
            i = order[_i]  # start with highest score box
            if suppressed[i] == 1:  # if any box have enough iou with this, remove it
                continue
            keep.append(i)
            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                # calculate center distance between i and j box
                dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

                # ovr = inter / areas[j]
                if dist <= thresh:
                    suppressed[j] = 1
        return keep

    def _circle_nms(self, boxes, min_radius, post_max_size=83):
        """
        NMS according to center distance
        """
        keep = np.array(self.circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
        keep = torch.from_numpy(keep).long().to(boxes.device)
        return keep

    def _rotate_nms(self, boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
        """
        :param boxes: (N, 5) [x1, y1, x2, y2, ry]
        :param scores: (N)
        :param thresh:
        :return:
        """
        # areas = (x2 - x1) * (y2 - y1)
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]

        boxes = boxes[order].contiguous()

        keep = torch.LongTensor(boxes.size(0))
        num_out = center_ops_cuda.center_rotate_nms_gpu(boxes, keep, thresh)
        selected = order[keep[:num_out].cuda()].contiguous()

        if post_max_size is not None:
            selected = selected[:post_max_size]

        return selected

    def _nms_gpu_3d(self, boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
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

    def _boxes3d_to_bevboxes_lidar_torch(self, boxes3d):
        """
        :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :return:
            boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
        """
        boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

        cu, cv = boxes3d[:, 0], boxes3d[:, 1]

        half_w, half_l = boxes3d[:, 3] / 2, boxes3d[:, 4] / 2
        boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l
        boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l
        boxes_bev[:, 4] = boxes3d[:, -1]
        return boxes_bev

    @torch.no_grad()
    def proposal_layer(self, pred_dict, post_center_range=None, score_threshold=None, cfg=None, raw_rot=False,
                       task_id=-1):

        assert self.encode_background_as_zeros is True
        assert self.use_sigmoid_score is True

        batch, cat, _, _ = pred_dict['center'].size()

        nms_cfg = cfg.nms.train if self.training else cfg.nms.test
        K = nms_cfg.nms_pre_max_size  # topK selected
        maxpool = nms_cfg.get('max_pool_nms', False) or \
                  (nms_cfg.get('circle_nms', False) and (nms_cfg.min_radius[task_id] == -1))
        use_circle_nms = nms_cfg.get('circle_nms', False) and (nms_cfg.min_radius[task_id] != -1)

        center = pred_dict['center']
        if maxpool:
            center = self._nms(center)
        scores, inds, clses, ys, xs = self._topk(center, K=K)

        assert pred_dict['reg'] is not None
        reg = self._transpose_and_gather_feat(pred_dict['reg'], inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        assert raw_rot is False
        sinr = self._transpose_and_gather_feat(pred_dict['sinr'], inds)
        sinr = sinr.view(batch, K, 1)
        cosr = self._transpose_and_gather_feat(pred_dict['cosr'], inds)
        cosr = cosr.view(batch, K, 1)
        rot = torch.atan2(sinr, cosr)

        # height in the bev
        height = self._transpose_and_gather_feat(pred_dict['height'], inds)
        height = height.view(batch, K, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(pred_dict['dim'], inds)
        dim = dim.view(batch, K, 3)

        # class label
        clses = clses.view(batch, K).float()
        scores = scores.view(batch, K)

        # center location
        pc_range = cfg.pc_range
        xs = xs.view(batch, K, 1) * cfg.out_size_factor * cfg.voxel_size[0] + pc_range[0]
        ys = ys.view(batch, K, 1) * cfg.out_size_factor * cfg.voxel_size[1] + pc_range[1]

        final_box_preds_lst = [xs, ys, height, dim, rot]

        if self.dataset == 'nuscenes':
            vel = self._transpose_and_gather_feat(pred_dict['vel'], inds)
            vel = vel.view(batch, K, 2)
            final_box_preds_lst.append(vel)

        final_box_preds = torch.cat(final_box_preds_lst, dim=2)
        final_scores = scores
        final_preds = clses

        # restrict center range
        assert post_center_range is not None
        post_center_range = torch.tensor(post_center_range).to(final_box_preds.device)
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        # use score threshold
        assert score_threshold is not None
        thresh_mask = final_scores > score_threshold
        mask &= thresh_mask

        predictions_dicts = []
        for i in range(batch):
            cmask = mask[i, :]
            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]

            # circle nms
            if use_circle_nms:
                centers = boxes3d[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                keep = self._circle_nms(boxes, min_radius=nms_cfg.min_radius[task_id],
                                        post_max_size=nms_cfg.post_max_size)

                boxes3d = boxes3d[keep]
                scores = scores[keep]
                labels = labels[keep]

            # rotate nms
            elif nms_cfg.get('use_rotate_nms', False):
                assert not maxpool
                top_scores = scores
                if top_scores.shape[0] != 0:
                    boxes_for_nms = self._boxes3d_to_bevboxes_lidar_torch(boxes3d)
                    selected = self._rotate_nms(boxes_for_nms, top_scores,
                                                thresh=nms_cfg.nms_iou_threshold,
                                                pre_maxsize=nms_cfg.nms_pre_max_size,
                                                post_max_size=nms_cfg.nms_post_max_size
                                                )
                else:
                    selected = []
                boxes3d = boxes3d[selected]
                labels = labels[selected]
                scores = scores[selected]

            # iou 3d nms
            elif nms_cfg.get('use_iou_3d_nms', False):
                assert not maxpool
                top_scores = scores
                if top_scores.shape[0] != 0:
                    selected = self._nms_gpu_3d(boxes3d[:, :7], top_scores,
                                                thresh=nms_cfg.nms_iou_threshold,
                                                pre_maxsize=nms_cfg.nms_pre_max_size,
                                                post_max_size=nms_cfg.nms_post_max_size
                                                )
                else:
                    selected = []
                boxes3d = boxes3d[selected]
                labels = labels[selected]
                scores = scores[selected]

            predictions_dict = {
                "boxes": boxes3d,
                "scores": scores,
                "labels": labels.long()
            }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts


    def _predict_boxes_preprocess(self, pred_dict, batch_size):
        rlt = {}

        batch_center = pred_dict['fusion_map'].sigmoid()
        # batch_corner = pred_dict['corner_map'].sigmoid()
        # batch_foreground = pred_dict['foreground_map']

        batch_reg = pred_dict['reg']
        batch_height = pred_dict['height']

        if not self.no_log:
            batch_dim = torch.exp(pred_dict['dim'])
            # add clamp for good init, otherwise we will get inf with exp
            batch_dim = torch.clamp(batch_dim, min=0.001, max=30)
        else:
            batch_dim = pred_dict['dim']
        batch_sin = pred_dict['rot'][:, 0:1]
        batch_cos = pred_dict['rot'][:, 1:2]

        rlt['center'] = batch_center
        # rlt['corner'] = batch_corner
        # rlt['foreground'] = batch_foreground
        rlt['sinr'] = batch_sin
        rlt['cosr'] = batch_cos
        rlt['height'] = batch_height
        rlt['dim'] = batch_dim
        rlt['reg'] = batch_reg
        rlt['dir_preds'] = [None] * batch_size

        if self.dataset == 'nuscenes':
            rlt['vel'] = pred_dict['vel']

        return rlt


    @torch.no_grad()
    def generate_predicted_boxes(self, data_dict):
        """
        Generate box predictions with decode, topk and circular_nms
        For single-stage-detector, another post-processing (nms) is needed
        For two-stage-detector, no need for proposal layer in roi_head
        Returns:
        """
        double_flip = not self.training and self.post_cfg.get('double_flip', False)
        post_center_range = self.post_cfg.post_center_limit_range
        pred_dicts = self.forward_ret_dict['multi_head_features']

        task_box_preds = {}
        task_score_preds = {}
        task_label_preds = {}

        for task_id, pred_dict in enumerate(pred_dicts):
            batch_size = pred_dict['center_map'].shape[0]
            if double_flip:
                assert batch_size % 4 == 0, print('cannot be divided by 4, batch_size=' + batch_size)
                batch_size = int(batch_size / 4)
                processed_dict = self._double_flip_process(pred_dict, batch_size)
                # convert data_dict format
                data_dict['batch_size'] = batch_size
            else:
                # batch_hm = pred_dict['hm'].sigmoid_() inplace may cause errors
                processed_dict = self._predict_boxes_preprocess(pred_dict, batch_size)

            # decode
            boxes = self.proposal_layer(processed_dict, post_center_range=post_center_range,
                                        score_threshold=self.post_cfg.score_threshold, cfg=self.post_cfg,
                                        task_id=task_id)

            task_box_preds[task_id] = [box['boxes'] for box in boxes]
            task_score_preds[task_id] = [box['scores'] for box in boxes]
            task_label_preds[task_id] = [box['labels'] for box in boxes]  # labels are local here

        pred_dicts = []
        batch_size = len(task_box_preds[0])
        rois, roi_scores, roi_labels = [], [], []
        nms_cfg = self.post_cfg.nms.train if self.training else self.post_cfg.nms.test
        num_rois = nms_cfg.nms_post_max_size * len(self.class_names)
        for batch_idx in range(batch_size):
            if self.dataset == 'nuscenes':
                offset = 1  # class label start from 1
            else:
                offset = 0
            final_boxes, final_scores, final_labels = [], [], []
            for task_id, class_name in enumerate(self.class_names):
                final_boxes.append(task_box_preds[task_id][batch_idx])
                final_scores.append(task_score_preds[task_id][batch_idx])
                # convert to global labels
                final_global_label = task_label_preds[task_id][batch_idx] + offset
                offset += len(class_name)
                final_labels.append(final_global_label)

            final_boxes = torch.cat(final_boxes)
            final_scores = torch.cat(final_scores)
            final_labels = torch.cat(final_labels)

            roi = final_boxes.new_zeros(num_rois, final_boxes.shape[-1])
            roi_score = final_scores.new_zeros(num_rois)
            roi_label = final_labels.new_zeros(num_rois)
            num_boxes = final_boxes.shape[0]
            roi[:num_boxes] = final_boxes
            roi_score[:num_boxes] = final_scores
            roi_label[:num_boxes] = final_labels
            rois.append(roi)
            roi_scores.append(roi_score)
            roi_labels.append(roi_label)

            record_dict = {
                "pred_boxes": final_boxes,
                "pred_scores": final_scores,
                "pred_labels": final_labels
            }
            pred_dicts.append(record_dict)

        data_dict['pred_dicts'] = pred_dicts
        data_dict['rois'] = torch.stack(rois, dim=0)
        data_dict['roi_scores'] = torch.stack(roi_scores, dim=0)
        data_dict['roi_labels'] = torch.stack(roi_labels, dim=0)
        data_dict['has_class_labels'] = True  # Force to be true
        data_dict.pop('batch_index', None)
        return data_dict


    # def get_center_from_idx(self, idx, center_map):
    #     bs, cls, h, w = center_map.size()
    #     xs = (idx % w).int().float()
    #     ys = (idx / w).int().float()
    #     return xs, ys
    #
    # def get_corners(self, ind, anchor):
    #     """
    #     This function return bev corner info in world coord
    #     M stands for max objs in this instance
    #     Args:
    #         ind: (B, num_cls, max_objs)
    #         anchors: (B, cls, n, 4) (w, l, h, theta) n=2 in most situations
    #
    #     Returns:
    #         raw_corners (B, num_cls, 4, 2)
    #     """
    #
    #
    # @staticmethod
    # def get_grids(rois, batch_size_rcnn, grid_size):
    #
    #     faked_features = rois.new_ones((grid_size, grid_size))
    #     dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
    #     dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)
    #     local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:5]
    #     roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) - (
    #                 local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
    #     return roi_grid_points