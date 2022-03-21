from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]],
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features

class StackSAModulePyramid(nn.Module):

    def __init__(self, *, mlps: List[List[int]], nsamples, use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return new_features

class StackSAModuleMSGDeform(nn.Module):
    """
    Set abstraction with single radius prediction for each roi
    """

    def __init__(self, *, temperatures: List[float], div_coefs: List[float], radii: List[float],
                 nsamples: List[int], predict_nsamples: List[int],
                 mlps: List[List[int]], pmlps: List[List[int]], pfcs: List[List[int]],
                 grid_size: int, use_xyz: bool = True):
        """
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.grid_size = grid_size
        self.MIN_R = 0.01

        self.radii_list = radii
        self.div_coef_list = div_coefs

        self.norm_groupers = nn.ModuleList()
        self.deform_groupers = nn.ModuleList()

        self.feat_mlps = nn.ModuleList()

        self.predict_mlps = nn.ModuleList()
        self.predict_fcs = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            predict_nsample = predict_nsamples[i]
            temperature = temperatures[i]

            self.norm_groupers.append(
                pointnet2_utils.QueryAndGroup(radius, predict_nsample, use_xyz=use_xyz)
            )
            self.deform_groupers.append(
                pointnet2_utils.QueryAndGroupDeform(temperature, nsample, use_xyz=use_xyz)
            )

            mlp_spec = mlps[i]
            predict_mlp_spec = pmlps[i]
            if use_xyz:
                mlp_spec[0] += 3
                predict_mlp_spec[0] += 3

            self.feat_mlps.append(self._make_mlp_layer(mlp_spec))

            self.predict_mlps.append(self._make_mlp_layer(predict_mlp_spec))

            fc_spec = pfcs[i]
            self.predict_fcs.append(self._make_fc_layer(fc_spec))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_mlp_layer(self, mlp_spec):
        mlps = []
        for i in range(len(mlp_spec) - 1):
            mlps.extend([
                nn.Conv2d(mlp_spec[i], mlp_spec[i + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp_spec[i + 1]),
                nn.ReLU()
            ])
        return nn.Sequential(*mlps)

    def _make_fc_layer(self, fc_spec):
        assert len(fc_spec) == 2
        return nn.Linear(fc_spec[0], fc_spec[1], bias = True)

    def forward(self, xyz, xyz_batch_cnt, rois, roi_features, features=None, temperature_decay=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param rois: (B, num_rois, grid_size^3, 3) roi grid points
        :param roi_features: (B, num_rois, C) roi features
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        batch_size = rois.shape[0]
        num_rois = rois.shape[1]
        new_xyz = rois.view(batch_size, -1, 3).contiguous()
        new_xyz_batch_cnt = new_xyz.new_full((batch_size), new_xyz.shape[1]).int()
        new_xyz = new_xyz.view(-1, 3).contiguous()
        new_features_list = []

        for k in range(len(self.norm_groupers)):
            # radius prediction
            predicted_features, ball_idxs = self.norm_groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M, C, nsample)
            predicted_features = predicted_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M, nsample)
            predicted_features = self.predict_mlps[k](predicted_features)  # (1, C, M, nsample)

            predicted_features = F.max_pool2d(
                predicted_features, kernel_size=[1, predicted_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M)

            # M = batch_size * num_rois * grid_size^3
            predicted_features = predicted_features.squeeze(0).permute(0, 1).contiguous() # (M, C)
            num_predicted_features = predicted_features.shape[1]
            predicted_features = predicted_features.view(batch_size, num_rois, self.grid_size ** 3, num_predicted_features)
            predicted_features = predicted_features.view(batch_size, num_rois, -1).contiguous()

            predicted_residual_r = self.predict_fcs[k](torch.cat([predicted_features, roi_features], dim = 2))  # (batch_size, num_rois, C -> 1)

            new_xyz_r = predicted_residual_r / self.div_coef_list[k] + self.radii_list[k]
            # constrain predicted radius above MIN_R
            new_xyz_r = torch.clamp(new_xyz_r, min = self.MIN_R)

            new_xyz_r = new_xyz_r.unsqueeze(2).repeat(1, 1, self.grid_size ** 3, 1) # (batch_size, num_rois, grid_size^3, 1)
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()

            # feature extraction
            # new_features (M, C, nsample) weights (M, nsample)
            new_features, new_weights, ball_idxs = self.deform_groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features, temperature_decay
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M, nsample)
            new_features = self.feat_mlps[k](new_features)  # (1, C, M, nsample)

            # multiply after mlps
            new_weights = new_weights.unsqueeze(0).unsqueeze(0) # (1, 1, M, nsample)
            new_features = new_weights * new_features
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features

class StackPointnetFPModule(nn.Module):
    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (N1 + N2 ..., C2 + C1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
        new_features = self.mlp(new_features)

        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features
