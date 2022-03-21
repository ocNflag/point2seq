import torch
import torch.nn as nn
import spconv
from functools import partial
import time

from ...ops.point_voxel_ops.pv_modules import VoxelModuleMSG
from ...ops.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

def get_v_coords(p_range, v_size, v_indices):
    """
    Args:
        p_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        v_size: [vx, vy, vz]
        v_indices : [M, 4] -> [bs, z_i, y_i, x_i]
    Returns:
        v_coords: [M, 4] -> [bs, x, y, z]
    """
    with torch.no_grad():
        v_size = torch.tensor(v_size).unsqueeze(0).to(v_indices.device)
        min_range = torch.tensor(p_range[0:3]).unsqueeze(0).to(v_indices.device)
        v_xyz_idx = v_indices[:, [3, 2, 1]]
        v_bs = v_indices[:, [0]].float()
        v_xyz = (v_indices[:, [3, 2, 1]].float() + 0.5) * v_size + min_range
        v_coords = torch.cat([v_bs, v_xyz], dim = 1)
    return v_coords

class P_TO_V_Module(nn.Module):
    """
    A wrapper for gathering point features for voxels
    """
    def __init__(self, radii, nsamples, mlps, v_size, v_range, point_cloud_range):
        super().__init__()
        self.gather_block = StackSAModuleMSG(radii = radii, nsamples = nsamples, mlps = mlps)
        self.v_size = v_size
        self.v_range = v_range
        self.point_cloud_range = point_cloud_range

    def forward(self, p_coords, p_features, sp_tensor):
        """
        Args:
            p_coords: [N, 4]
            p_features: [N, C]
        """
        batch_size = sp_tensor.batch_size
        v_indices_range = sp_tensor.spatial_shape
        assert v_indices_range[0] == self.v_range[0], \
            'voxels indices range must be equal in Z dim, input: {}, cfgs: {}'.format(v_indices_range[0], self.v_range[0])
        assert v_indices_range[1] == self.v_range[1], \
            'voxels indices range must be equal in Y dim, input: {}, cfgs: {}'.format(v_indices_range[1], self.v_range[1])
        assert v_indices_range[2] == self.v_range[2], \
            'voxels indices range must be equal in X dim, input: {}, cfgs: {}'.format(v_indices_range[2], self.v_range[2])
        v_indices = sp_tensor.indices
        v_coords = get_v_coords(self.point_cloud_range, self.v_size, v_indices)

        p_batch_cnt = p_coords.new_zeros(batch_size).int()
        batch_idx = p_coords[:, 0]
        for k in range(batch_size):
            p_batch_cnt[k] = (batch_idx == k).sum()
        v_batch_cnt = v_coords.new_zeros(batch_size).int()
        batch_idx = v_coords[:, 0]
        for k in range(batch_size):
            v_batch_cnt[k] = (batch_idx == k).sum()
        
        xyz = p_coords[:, 1:4].contiguous()
        new_xyz = v_coords[:, 1:4].contiguous()
        _, p_to_v_features = self.gather_block(xyz, p_batch_cnt, new_xyz, v_batch_cnt, features = p_features)
        return p_to_v_features

class PointVoxelBackBone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.v_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.v_to_p_block1 = VoxelModuleMSG(
            steps = [[16, 16, 8]], 
            dilated_rates = [[2, 2, 2]], 
            bins = [[2, 2, 2]], 
            nsamples = [32], 
            mlps = [[16, 16, 16]], 
            v_size = [0.05, 0.05, 0.1], 
            v_range = [41, 1600, 1408], 
            point_cloud_range = point_cloud_range, 
            max_hash_size = 60000
        )

        self.p_to_v_block1 = P_TO_V_Module(
            radii = self.model_cfg.PV_CFG.x_conv1.RADIUS, 
            nsamples = self.model_cfg.PV_CFG.x_conv1.NSAMPLE, 
            mlps = self.model_cfg.PV_CFG.x_conv1.MLPS, 
            v_size = [0.05, 0.05, 0.1], 
            v_range = [41, 1600, 1408], 
            point_cloud_range = point_cloud_range
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.v_to_p_block2 = VoxelModuleMSG(
            steps = [[12, 12, 6]], 
            dilated_rates = [[2, 2, 2]], 
            bins = [[2, 2, 2]], 
            nsamples = [32], 
            mlps = [[32, 32, 32]], 
            v_size = [0.1, 0.1, 0.2], 
            v_range = [21, 800, 704], 
            point_cloud_range = point_cloud_range, 
            max_hash_size = 60000
        )

        self.p_to_v_block2 = P_TO_V_Module(
            radii = self.model_cfg.PV_CFG.x_conv2.RADIUS, 
            nsamples = self.model_cfg.PV_CFG.x_conv2.NSAMPLE, 
            mlps = self.model_cfg.PV_CFG.x_conv2.MLPS, 
            v_size = [0.1, 0.1, 0.2], 
            v_range = [21, 800, 704], 
            point_cloud_range = point_cloud_range
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.v_to_p_block3 = VoxelModuleMSG(
            steps = [[12, 12, 6]], 
            dilated_rates = [[2, 2, 2]], 
            bins = [[2, 2, 2]], 
            nsamples = [32], 
            mlps = [[64, 64, 64]], 
            v_size = [0.2, 0.2, 0.4], 
            v_range = [11, 400, 352], 
            point_cloud_range = point_cloud_range, 
            max_hash_size = 60000
        )

        self.p_to_v_block3 = P_TO_V_Module(
            radii = self.model_cfg.PV_CFG.x_conv3.RADIUS, 
            nsamples = self.model_cfg.PV_CFG.x_conv3.NSAMPLE, 
            mlps = self.model_cfg.PV_CFG.x_conv3.MLPS, 
            v_size = [0.2, 0.2, 0.4], 
            v_range = [11, 400, 352], 
            point_cloud_range = point_cloud_range
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(128, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.v_to_p_block4 = VoxelModuleMSG(
            steps = [[12, 12, 6]], 
            dilated_rates = [[2, 2, 2]], 
            bins = [[2, 2, 2]], 
            nsamples = [32], 
            mlps = [[64, 64, 64]], 
            v_size = [0.4, 0.4, 0.8], 
            v_range = [5, 200, 176],
            point_cloud_range = point_cloud_range, 
            max_hash_size = 60000
        )

        self.p_to_v_block4 = P_TO_V_Module(
            radii = self.model_cfg.PV_CFG.x_conv4.RADIUS, 
            nsamples = self.model_cfg.PV_CFG.x_conv4.NSAMPLE, 
            mlps = self.model_cfg.PV_CFG.x_conv4.MLPS, 
            v_size = [0.4, 0.4, 0.8], 
            v_range = [5, 200, 176], 
            point_cloud_range = point_cloud_range
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']

        src_points = batch_dict['points'][:, 1:4]
        batch_indices = batch_dict['points'][:, 0].long()

        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)

            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
            ).long()

            if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            bs_indices = keypoints.new_full((1, keypoints.shape[1], 1), bs_idx)
            keypoints = torch.cat([bs_indices, keypoints], dim = 2)

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 4)
        return keypoints


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        keypoints = self.get_sampled_points(batch_dict)
        p_coords = keypoints.view(-1, 4).contiguous()
        batch_dict['point_coords'] = p_coords


        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        #a1 = time.time()
        x_conv1 = self.conv1(x)
        #a2 = time.time()
        p_feats1 = self.v_to_p_block1(x_conv1, p_coords)
        #a3 = time.time()
        v_feats1 = self.p_to_v_block1(p_coords, p_feats1, x_conv1)
        #a4 = time.time()
        x_conv1.features = torch.cat([x_conv1.features, v_feats1], dim = 1)
        #print('----profiling-----')
        #print('----layer_one-----')
        #print('sp_conv: {}'.format(a2-a1))
        #print('voxel_to_point: {}'.format(a3-a2))
        #print('point_to_voxel: {}'.format(a4-a3))

        #a1 = time.time()
        x_conv2 = self.conv2(x_conv1)
        #a2 = time.time()
        p_feats2 = self.v_to_p_block2(x_conv2, p_coords)
        #a3 = time.time()
        v_feats2 = self.p_to_v_block2(p_coords, p_feats2, x_conv2)
        #a4 = time.time()
        x_conv2.features = torch.cat([x_conv2.features, v_feats2], dim = 1)
        #print('----profiling-----')
        #print('----layer_two-----')
        #print('sp_conv: {}'.format(a2-a1))
        #print('voxel_to_point: {}'.format(a3-a2))
        #print('point_to_voxel: {}'.format(a4-a3))

        #a1 = time.time()
        x_conv3 = self.conv3(x_conv2)
        #a2 = time.time()
        p_feats3 = self.v_to_p_block3(x_conv3, p_coords)
        #a3 = time.time()
        v_feats3 = self.p_to_v_block3(p_coords, p_feats3, x_conv3)
        #a4 = time.time()
        x_conv3.features = torch.cat([x_conv3.features, v_feats3], dim = 1)
        #print('----profiling-----')
        #print('----layer_three-----')
        #print('sp_conv: {}'.format(a2-a1))
        #print('voxel_to_point: {}'.format(a3-a2))
        #print('point_to_voxel: {}'.format(a4-a3))

        #a1 = time.time()
        x_conv4 = self.conv4(x_conv3)
        #a2 = time.time()
        p_feats4 = self.v_to_p_block4(x_conv4, p_coords)
        #a3 = time.time()
        v_feats4 = self.p_to_v_block4(p_coords, p_feats4, x_conv4)
        #a4 = time.time()
        x_conv4.features = torch.cat([x_conv4.features, v_feats4], dim = 1)
        #print('----profiling-----')
        #print('----layer_four-----')
        #print('sp_conv: {}'.format(a2-a1))
        #print('voxel_to_point: {}'.format(a3-a2))
        #print('point_to_voxel: {}'.format(a4-a3))

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_point_features': {
                'x_conv1': p_feats1,
                'x_conv2': p_feats2,
                'x_conv3': p_feats3,
                'x_conv4': p_feats4,
            }
        })

        return batch_dict


class PointVoxelBackBoneLarge(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.v_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.v_to_p_block1 = VoxelModuleMSG(
            steps = [[16, 16, 8]], 
            dilated_rates = [[2, 2, 2]], 
            bins = [[2, 2, 2]], 
            nsamples = [32], 
            mlps = [[16, 16, 16]], 
            v_size = [0.06, 0.06, 0.1], 
            v_range = [41, 1600, 1408], 
            point_cloud_range = point_cloud_range, 
            max_hash_size = 60000
        )

        self.p_to_v_block1 = P_TO_V_Module(
            radii = self.model_cfg.PV_CFG.x_conv1.RADIUS, 
            nsamples = self.model_cfg.PV_CFG.x_conv1.NSAMPLE, 
            mlps = self.model_cfg.PV_CFG.x_conv1.MLPS, 
            v_size = [0.06, 0.06, 0.1], 
            v_range = [41, 1600, 1408], 
            point_cloud_range = point_cloud_range
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.v_to_p_block2 = VoxelModuleMSG(
            steps = [[12, 12, 6]], 
            dilated_rates = [[2, 2, 2]], 
            bins = [[2, 2, 2]], 
            nsamples = [32], 
            mlps = [[32, 32, 32]], 
            v_size = [0.12, 0.12, 0.2], 
            v_range = [21, 800, 704], 
            point_cloud_range = point_cloud_range, 
            max_hash_size = 60000
        )

        self.p_to_v_block2 = P_TO_V_Module(
            radii = self.model_cfg.PV_CFG.x_conv2.RADIUS, 
            nsamples = self.model_cfg.PV_CFG.x_conv2.NSAMPLE, 
            mlps = self.model_cfg.PV_CFG.x_conv2.MLPS, 
            v_size = [0.12, 0.12, 0.2], 
            v_range = [21, 800, 704], 
            point_cloud_range = point_cloud_range
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.v_to_p_block3 = VoxelModuleMSG(
            steps = [[12, 12, 6]], 
            dilated_rates = [[2, 2, 2]], 
            bins = [[2, 2, 2]], 
            nsamples = [32], 
            mlps = [[64, 64, 64]], 
            v_size = [0.24, 0.24, 0.4], 
            v_range = [11, 400, 352], 
            point_cloud_range = point_cloud_range, 
            max_hash_size = 60000
        )

        self.p_to_v_block3 = P_TO_V_Module(
            radii = self.model_cfg.PV_CFG.x_conv3.RADIUS, 
            nsamples = self.model_cfg.PV_CFG.x_conv3.NSAMPLE, 
            mlps = self.model_cfg.PV_CFG.x_conv3.MLPS, 
            v_size = [0.24, 0.24, 0.4], 
            v_range = [11, 400, 352], 
            point_cloud_range = point_cloud_range
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(128, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.v_to_p_block4 = VoxelModuleMSG(
            steps = [[12, 12, 6]], 
            dilated_rates = [[2, 2, 2]], 
            bins = [[2, 2, 2]], 
            nsamples = [32], 
            mlps = [[64, 64, 64]], 
            v_size = [0.48, 0.48, 0.8], 
            v_range = [5, 200, 176],
            point_cloud_range = point_cloud_range, 
            max_hash_size = 60000
        )

        self.p_to_v_block4 = P_TO_V_Module(
            radii = self.model_cfg.PV_CFG.x_conv4.RADIUS, 
            nsamples = self.model_cfg.PV_CFG.x_conv4.NSAMPLE, 
            mlps = self.model_cfg.PV_CFG.x_conv4.MLPS, 
            v_size = [0.48, 0.48, 0.8], 
            v_range = [5, 200, 176], 
            point_cloud_range = point_cloud_range
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']

        src_points = batch_dict['points'][:, 1:4]
        batch_indices = batch_dict['points'][:, 0].long()

        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)

            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
            ).long()

            if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            bs_indices = keypoints.new_full((1, keypoints.shape[1], 1), bs_idx)
            keypoints = torch.cat([bs_indices, keypoints], dim = 2)

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 4)
        return keypoints


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        keypoints = self.get_sampled_points(batch_dict)
        p_coords = keypoints.view(-1, 4).contiguous()
        batch_dict['point_coords'] = p_coords


        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        #a1 = time.time()
        x_conv1 = self.conv1(x)
        #a2 = time.time()
        p_feats1 = self.v_to_p_block1(x_conv1, p_coords)
        #a3 = time.time()
        v_feats1 = self.p_to_v_block1(p_coords, p_feats1, x_conv1)
        #a4 = time.time()
        x_conv1.features = torch.cat([x_conv1.features, v_feats1], dim = 1)
        #print('----profiling-----')
        #print('----layer_one-----')
        #print('sp_conv: {}'.format(a2-a1))
        #print('voxel_to_point: {}'.format(a3-a2))
        #print('point_to_voxel: {}'.format(a4-a3))

        #a1 = time.time()
        x_conv2 = self.conv2(x_conv1)
        #a2 = time.time()
        p_feats2 = self.v_to_p_block2(x_conv2, p_coords)
        #a3 = time.time()
        v_feats2 = self.p_to_v_block2(p_coords, p_feats2, x_conv2)
        #a4 = time.time()
        x_conv2.features = torch.cat([x_conv2.features, v_feats2], dim = 1)
        #print('----profiling-----')
        #print('----layer_two-----')
        #print('sp_conv: {}'.format(a2-a1))
        #print('voxel_to_point: {}'.format(a3-a2))
        #print('point_to_voxel: {}'.format(a4-a3))

        #a1 = time.time()
        x_conv3 = self.conv3(x_conv2)
        #a2 = time.time()
        p_feats3 = self.v_to_p_block3(x_conv3, p_coords)
        #a3 = time.time()
        v_feats3 = self.p_to_v_block3(p_coords, p_feats3, x_conv3)
        #a4 = time.time()
        x_conv3.features = torch.cat([x_conv3.features, v_feats3], dim = 1)
        #print('----profiling-----')
        #print('----layer_three-----')
        #print('sp_conv: {}'.format(a2-a1))
        #print('voxel_to_point: {}'.format(a3-a2))
        #print('point_to_voxel: {}'.format(a4-a3))

        #a1 = time.time()
        x_conv4 = self.conv4(x_conv3)
        #a2 = time.time()
        p_feats4 = self.v_to_p_block4(x_conv4, p_coords)
        #a3 = time.time()
        v_feats4 = self.p_to_v_block4(p_coords, p_feats4, x_conv4)
        #a4 = time.time()
        x_conv4.features = torch.cat([x_conv4.features, v_feats4], dim = 1)
        #print('----profiling-----')
        #print('----layer_four-----')
        #print('sp_conv: {}'.format(a2-a1))
        #print('voxel_to_point: {}'.format(a3-a2))
        #print('point_to_voxel: {}'.format(a4-a3))

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_point_features': {
                'x_conv1': p_feats1,
                'x_conv2': p_feats2,
                'x_conv3': p_feats3,
                'x_conv4': p_feats4,
            }
        })

        return batch_dict
