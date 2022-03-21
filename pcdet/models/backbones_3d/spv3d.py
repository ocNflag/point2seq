import torch
import torch.nn as nn
import spconv
from functools import partial
import time

from ...ops.point_voxel_ops import pv_utils

def sp_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
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

def mlp_block(channel_list):
    """
    Input: [B, C, N]
    """
    mlps = []
    pre_channel = channel_list[0]
    for i in range(1, len(channel_list)):
        post_channel = channel_list[i]
        mlps.append(nn.Conv1d(pre_channel, post_channel, 1, bias=False))
        mlps.append(nn.BatchNorm1d(post_channel))
        mlps.append(nn.ReLU())
        pre_channel = post_channel

    m = nn.Sequential(
        *mlps
    )

    return m

def fc_block(channel_list):
    """
    Input : [N, C]
    """
    fcs = []
    pre_channel = channel_list[0]
    for i in range(1, len(channel_list)):
        post_channel = channel_list[i]
        fcs.append(nn.Linear(pre_channel, post_channel, bias=False))
        fcs.append(nn.BatchNorm1d(post_channel))
        fcs.append(nn.ReLU())
        pre_channel = post_channel

    m = nn.Sequential(
        *fcs
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

class PosPool(nn.Module):
    def __init__(self, proj_specs, kernel, v_size):
        super().__init__()
        self.kernel = kernel
        self.v_size = v_size
        if len(proj_specs) == 0:
            self.proj_layer = nn.Sequential()
        else:
            if self.kernel == 'max':
                proj_specs[0] += 4
            self.proj_layer = fc_block(proj_specs)

    def forward(self, features, coords, centers, mask):
        """
        Args:
            features: [N, max, C]
            coords: [N, max, 3]
            centers: [N, 4] -> bs, x, y, z
            mask: [N, max]
        Returns:
            features: [N, C]
        """
        with torch.no_grad():
            centers = centers[:, 1:4]
            relative_coords = coords - centers.unsqueeze(1)
            relative_dist = (relative_coords ** 2).sum(2, keepdim = True)
            if self.kernel == 'xyz':
                normalizer = torch.tensor([self.v_size[0], self.v_size[1], self.v_size[2], self.v_size[0] ** 2 + self.v_size[1] ** 2 + self.v_size[2] ** 2])
                normalizer = normalizer.view(1, 1, 4).contiguous().to(centers.device)
                geo_features = torch.cat([relative_coords, relative_dist], dim = 2) # [N, max, 4]
                geo_features = geo_features / normalizer
            elif self.kernel == 'exp':
                # ERROR
                normalizer = torch.tensor([self.v_size[0], self.v_size[1], self.v_size[2], self.v_size[0] ** 2 + self.v_size[1] ** 2 + self.v_size[2] ** 2])
                normalizer = normalizer.view(1, 1, 4).contiguous().to(centers.device)
                geo_features = torch.cat([relative_coords, relative_dist], dim = 2) # [N, max, 4]
                geo_features = (-1) * geo_features.abs() / normalizer
                geo_features = torch.exp(geo_features)
                #print(geo_features.max())
            elif self.kernel == 'max':
                geo_features = torch.cat([relative_coords, relative_dist], dim = 2) # [N, max, 4]
            else:
                raise NotImplementedError

        if self.kernel == 'max':
            features = torch.cat([geo_features, features], dim = 2)
            N, ns, C = features.shape
            features = self.proj_layer(features.view(-1, C).contiguous())
            features = features.view(N, ns, -1).contiguous()
            features = features.max(1)[0]
        else:
            N, ns, C = features.shape
            features = self.proj_layer(features.view(-1, C).contiguous())
            features = features.view(N, ns, -1).contiguous()
            num_features = features.shape[2]
            assert num_features % 4 == 0, 'feature dimensions divided by 4 (xyzd)'
            features = features * geo_features.repeat(1, 1, num_features // 4)
            features = torch.sum(features * mask.unsqueeze(-1), dim = 1) / torch.clamp(mask.sum(1, keepdim = True), min = 1)
        return features

def gather_stack(batch_size, indices, coords, features):
    """
    Args:
        indices: [N1+N2, 1 + nsample] -> bs_idx, n1, n2 ...
        coords: [M1+M2, 4] -> bs_idx, x, y, z
        features: [M1+M2, C] -> c1, c2, c3 ... w/o leading bs_idx
    Returns:
        grouped_coords: [N1+N2, nsample, 3]
        grouped_features: [N1+N2, nsample, C]
    """
    # TODO: Convert to parallel format
    grouped_coords = []
    grouped_features = []
    for i in range(batch_size):
        indices_one = indices[indices[:, 0] == i, 1:] # [N, nsample]
        coords_one = coords[coords[:, 0] == i, 1:] # [M, 3]
        features_one = features[coords[:, 0] == i, :] # [M, C]
        grouped_features.append(features_one[indices_one, :]) # [N, nsample, C]
        grouped_coords.append(coords_one[indices_one, :])
    grouped_coords = torch.cat(grouped_coords, dim = 0)
    grouped_features = torch.cat(grouped_features, dim = 0)
    return grouped_coords, grouped_features

class P2VModule(nn.Module):
    def __init__(self, num_samples, v_size, v_range, point_cloud_range, fusion_mode, proj_specs, kernel, max_hash_size):
        super().__init__()
        self.num_samples = num_samples
        self.v_size = v_size
        self.v_range = v_range
        self.z_max, self.y_max, self.x_max = v_range
        self.x_size, self.y_size, self.z_size = v_size
        self.fusion_mode = fusion_mode
        self.point_cloud_range = point_cloud_range
        self.pos_pool = PosPool(proj_specs, kernel, v_size)
        self.max_hash_size = max_hash_size
        self.use_stack = True

    def get_p_to_v_map_stack(self, batch_size, p_coords, v_indices):
        """
        Returns:
            v_maps: [M1+M2, 1 + nsamples]
            v_masks: [M1+M2, nsamples]
        """
        with torch.no_grad():
            p_coords = p_coords.contiguous()
            v_bs_cnt = torch.zeros((batch_size)).int().to(v_indices.device)
            p_bs_cnt = torch.zeros((batch_size)).int().to(p_coords.device)
            for i in range(batch_size):
                v_bs_cnt[i] = (v_indices[:, 0] == i).sum()
                p_bs_cnt[i] = (p_coords[:, 0] == i).sum()
            xyz_to_vidx = torch.cuda.IntTensor(batch_size, self.max_hash_size, 2).fill_(-1)           
            xyz_to_vidx = pv_utils.fill_dense_stack(v_indices, xyz_to_vidx, v_bs_cnt, self.v_range)
            v_map, v_mask = pv_utils.point_voxel_query_stack(p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, 
                                                                self.v_size, self.v_range, self.num_samples)
            v_map_mask = (v_map >= 0).float()
            v_map[v_map < 0] = 0
            # for gather stack [..., 1+nsamples]
            v_map = torch.cat([v_indices[:,[0]].int(), v_map], dim = 1)
        return v_map.long(), v_map_mask        

    def get_p_to_v_map(self, batch_size, p_coords, v_indices):
        """
        Returns:
            v_maps: [M1+M2, 1 + nsamples]
            v_masks: [M1+M2, nsamples]
        """
        with torch.no_grad():
            v_maps = []
            v_masks = []
            for i in range(batch_size):
                p_coords_one = p_coords[p_coords[:,0] == i, 1:4]
                v_indices_one = v_indices[v_indices[:,0] == i, 1:4].int() # convert to Int
                xyz_to_vidx = torch.cuda.IntTensor(self.x_max, self.y_max, self.z_max).fill_(-1)
                xyz_to_vidx = pv_utils.fill_dense(v_indices_one, xyz_to_vidx, self.v_range)
                v_map, v_mask = pv_utils.point_voxel_query(p_coords_one, v_indices_one, xyz_to_vidx, self.v_size, self.v_range, self.num_samples)
                del xyz_to_vidx
                torch.cuda.empty_cache()
                v_map_mask = (v_map >= 0).float()
                v_map[v_map < 0] = 0
                v_map = torch.cat([torch.cuda.IntTensor(v_map.shape[0], 1).fill_(i), v_map], dim = 1)
                v_maps.append(v_map)
                v_masks.append(v_map_mask)
            v_maps = torch.cat(v_maps, dim = 0).long() # convert to Long
            v_masks = torch.cat(v_masks, dim = 0)
        return v_maps, v_masks  

    def forward(self, p_coords, p_features, sp_tensor):
        """
        Args:
            p_coords: [N1+N2, 4]
            p_features: [N1+N2, C]
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
        v_features = sp_tensor.features
        v_coords = get_v_coords(self.point_cloud_range, self.v_size, v_indices)

        #a1 = time.time()

        # v_map [M1+M2, 1 + max_points] index array, v_mask [M1+M2, max_points] 0-1
        if self.use_stack:
            v_map, v_mask = self.get_p_to_v_map_stack(batch_size, p_coords, v_indices)
        else:
            v_map, v_mask = self.get_p_to_v_map(batch_size, p_coords, v_indices)
        assert v_map.shape[0] == v_indices.shape[0], \
            'the number of non-empty voxels should be equal, v_map: {}, v_indices: {}'.format(v_map.shape[0], v_indices.shape[0])

        #a2 = time.time()
        p_to_v_coords, p_to_v_features = gather_stack(batch_size, v_map, p_coords, p_features)
        #a3 = time.time()
        p_to_v_features = self.pos_pool(p_to_v_features, p_to_v_coords, v_coords, v_mask)
        #a4 = time.time()

        #print('------profiling--p2v-------')
        #print('get index: {}'.format(a2 - a1))
        #print('gather: {}'.format(a3 - a2))
        #print('pos pool: {}'.format(a4 - a3))

        if self.fusion_mode == 'sum':
            assert v_features.shape[1] == p_to_v_features.shape[1], \
                'feature dimentions have to be the same, v_features: {}, p_to_v_features: {}'.format(v_features.shape[1], p_to_v_features.shape[1])
            v_features = v_features + p_to_v_features
        elif self.fusion_mode == 'cat':
            v_features = torch.cat([v_features, p_to_v_features], dim = 1)
        else:
            raise NotImplementedError

        output_sp_tensor = spconv.SparseConvTensor(
            features=v_features,
            indices=v_indices,
            spatial_shape=v_indices_range,
            batch_size=batch_size
        )
        return output_sp_tensor

class V2PModule(nn.Module):
    def __init__(self, num_samples, v_size, v_range, point_cloud_range, fusion_mode, proj_specs, kernel, max_hash_size, step):
        super().__init__()
        self.num_samples = num_samples
        self.v_size = v_size
        self.v_range = v_range
        self.z_max, self.y_max, self.x_max = v_range
        self.x_size, self.y_size, self.z_size = v_size
        self.fusion_mode = fusion_mode
        self.point_cloud_range = point_cloud_range
        self.pos_pool = PosPool(proj_specs, kernel, v_size)
        self.max_hash_size = max_hash_size
        self.step = step
        self.use_stack = True

    def get_v_to_p_map_stack(self, batch_size, p_coords, v_indices):
        """
        Returns:
            p_maps: [N1+N2, 1 + nsamples]
            p_masks: [N1+N2, nsamples]
        """
        with torch.no_grad():
            p_coords = p_coords.contiguous()
            v_bs_cnt = torch.zeros((batch_size)).int().to(v_indices.device)
            p_bs_cnt = torch.zeros((batch_size)).int().to(p_coords.device)
            for i in range(batch_size):
                v_bs_cnt[i] = (v_indices[:, 0] == i).sum()
                p_bs_cnt[i] = (p_coords[:, 0] == i).sum()
            xyz_to_vidx = torch.cuda.IntTensor(batch_size, self.max_hash_size, 2).fill_(-1)           
            xyz_to_vidx = pv_utils.fill_dense_stack(v_indices, xyz_to_vidx, v_bs_cnt, self.v_range)
            p_map, p_mask = pv_utils.voxel_point_query_stack(p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, 
                                                                self.v_size, self.v_range, self.num_samples, self.step)
            p_map_mask = (p_map >= 0).float()
            p_map[p_map < 0] = 0
            # for gather stack [..., 1+nsamples]
            p_map = torch.cat([p_coords[:,[0]].int(), p_map], dim = 1)
        return p_map.long(), p_map_mask 

    def get_v_to_p_map(self, batch_size, p_coords, v_indices):
        """
        Returns:
            p_maps: [N1+N2, 1 + nsamples]
            p_masks: [N1+N2, nsamples]
        """
        with torch.no_grad():
            p_maps = []
            p_masks = []
            for i in range(batch_size):
                p_coords_one = p_coords[p_coords[:,0] == i, 1:4]
                v_indices_one = v_indices[v_indices[:,0] == i, 1:4].int() # convert to Int
                xyz_to_vidx = torch.cuda.IntTensor(self.x_max, self.y_max, self.z_max).fill_(-1)
                xyz_to_vidx = pv_utils.fill_dense(v_indices_one, xyz_to_vidx, self.v_range)
                p_map, p_mask = pv_utils.voxel_point_query(p_coords_one, v_indices_one, xyz_to_vidx, self.v_size, self.v_range, self.num_samples)
                del xyz_to_vidx
                torch.cuda.empty_cache()
                p_map_mask = (p_map >= 0).float()
                p_map[p_map < 0] = 0
                p_map = torch.cat([torch.cuda.IntTensor(p_map.shape[0], 1).fill_(i), p_map], dim = 1)
                p_maps.append(p_map)
                p_masks.append(p_map_mask)
            p_maps = torch.cat(p_maps, dim = 0).long()
            p_masks = torch.cat(p_masks, dim = 0)
        return p_maps, p_masks        

    def forward(self, sp_tensor, p_coords, p_features):
        """
        Args:
            p_coords: [N1+N2, 4]
            p_features: [N1+N2, C]
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
        v_features = sp_tensor.features
        v_coords = get_v_coords(self.point_cloud_range, self.v_size, v_indices)

        #a1 = time.time()
        with torch.no_grad():
            # p_map [N1+N2, 1 + max_neighboring_voxel_centers] index array, v_mask [N1+N2, max_neighboring_voxel_centers] 0-1
            if self.use_stack:
                p_map, p_mask = self.get_v_to_p_map_stack(batch_size, p_coords, v_indices)
            else:
                p_map, p_mask = self.get_v_to_p_map(batch_size, p_coords, v_indices)
        assert p_map.shape[0] == p_coords.shape[0], \
            'the number of points should be equal, p_map: {}, p_coords: {}'.format(p_map.shape[0], p_coords.shape[0])

        #a2 = time.time()
        v_to_p_coords, v_to_p_features = gather_stack(batch_size, p_map, v_coords, v_features)
        #a3 = time.time()
        v_to_p_features = self.pos_pool(v_to_p_features, v_to_p_coords, p_coords, p_mask)
        #a4 = time.time()
        #print('------profiling--v2p-------')
        #print('get index: {}'.format(a2 - a1))
        #print('gather: {}'.format(a3 - a2))
        #print('pos pool: {}'.format(a4 - a3))

        if self.fusion_mode == 'sum':
            assert p_features.shape[1] == v_to_p_features.shape[1], \
                'feature dimentions have to be the same, p_features: {}, v_to_p_features: {}'.format(p_features.shape[1], v_to_p_features.shape[1])
            p_features = p_features + v_to_p_features
        elif self.fusion_mode == 'cat':
            p_features = torch.cat([p_features, v_to_p_features], dim = 1)
        else:
            raise NotImplementedError

        return p_features

class V2VModule(nn.Module):
    def __init__(self, sp_specs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        sp_conv_list = []
        for sp_spec in sp_specs:
            in_channels = sp_spec[0]
            out_channels = sp_spec[1]
            kernel_size = self.check_is_tuple(sp_spec[2])
            stride = self.check_is_tuple(sp_spec[3])
            padding = self.check_is_tuple(sp_spec[4])
            indice_key = sp_spec[5]
            conv_type = sp_spec[6]
            sp_conv_list.append(
                sp_block(in_channels, out_channels, kernel_size, norm_fn=norm_fn, stride=stride, padding=padding, indice_key=indice_key, conv_type=conv_type)
                )

        self.sp_conv = spconv.SparseSequential(
            *sp_conv_list
        )

    def check_is_tuple(self, v):
        if isinstance(v, list):
            return tuple(v)
        elif isinstance(v, int):
            return v
        else:
            raise TypeError 

    def forward(self, input_sp_tensor):
        return self.sp_conv(input_sp_tensor)

class P2PModule(nn.Module):
    def __init__(self, specs, mode = 'FC'):
        super().__init__()
        if mode == 'FC':
            self.p2p_layer = fc_block(specs)
        else:
            raise NotImplementedError

    def forward(self, p_features):
        p_features = self.p2p_layer(p_features)
        return p_features

class PVBlock(nn.Module):
    def __init__(self, cfg, point_cloud_range):
        super().__init__()
        self.p2v_module = P2VModule(num_samples = cfg.P_TO_V.N_SAMPLES, v_size = cfg.P_TO_V.V_SIZE, v_range = cfg.P_TO_V.V_RANGE,
                                    point_cloud_range = point_cloud_range, fusion_mode = cfg.P_TO_V.FUSION, proj_specs = cfg.P_TO_V.PROJ, 
                                    kernel = cfg.P_TO_V.KERNEL, max_hash_size = cfg.P_TO_V.HASH_SIZE)
        self.v2p_module = V2PModule(num_samples = cfg.V_TO_P.N_SAMPLES, v_size = cfg.V_TO_P.V_SIZE, v_range = cfg.V_TO_P.V_RANGE,
                                    point_cloud_range = point_cloud_range, fusion_mode = cfg.V_TO_P.FUSION, proj_specs = cfg.V_TO_P.PROJ, 
                                    kernel = cfg.V_TO_P.KERNEL, max_hash_size = cfg.V_TO_P.HASH_SIZE, step = cfg.V_TO_P.STEP)
        self.v2v_module = V2VModule(sp_specs = cfg.V_TO_V.SP_SPECS)
        self.p2p_module = P2PModule(specs = cfg.P_TO_P.SPECS)

    def forward(self, sp_tensor, p_coords, p_features):
        #a1 = time.time()
        sp_tensor = self.p2v_module(p_coords, p_features, sp_tensor)
        #a2 = time.time()
        sp_tensor = self.v2v_module(sp_tensor)
        #a3 = time.time()
        p_features = self.p2p_module(p_features)
        #a4 = time.time()
        p_features = self.v2p_module(sp_tensor, p_coords, p_features)
        #a5 = time.time()
        #print('----profiling-----')
        #print('p_to_v: {}'.format(a2-a1))
        #print('v_to_v: {}'.format(a3-a2))
        #print('p_to_p: {}'.format(a4-a3))
        #print('v_to_p: {}'.format(a5-a4))
        return sp_tensor, p_coords, p_features

class PVBlock2(nn.Module):
    """
    similar to pv-rcnn or pv-cnn
    """
    def __init__(self, cfg, point_cloud_range):
        super().__init__()
        self.v2p_module = V2PModule(num_samples = cfg.V_TO_P.N_SAMPLES, v_size = cfg.V_TO_P.V_SIZE, v_range = cfg.V_TO_P.V_RANGE,
                                    point_cloud_range = point_cloud_range, fusion_mode = cfg.V_TO_P.FUSION, proj_specs = cfg.V_TO_P.PROJ, 
                                    kernel = cfg.V_TO_P.KERNEL, max_hash_size = cfg.V_TO_P.HASH_SIZE, step = cfg.V_TO_P.STEP)
        self.v2v_module = V2VModule(sp_specs = cfg.V_TO_V.SP_SPECS)
        self.p2p_module = P2PModule(specs = cfg.P_TO_P.SPECS)

    def forward(self, sp_tensor, p_coords, p_features):
        #a1 = time.time()
        #a2 = time.time()
        sp_tensor = self.v2v_module(sp_tensor)
        #a3 = time.time()
        p_features = self.p2p_module(p_features)
        #a4 = time.time()
        p_features = self.v2p_module(sp_tensor, p_coords, p_features)
        #a5 = time.time()
        #print('----profiling-----')
        #print('p_to_v: {}'.format(a2-a1))
        #print('v_to_v: {}'.format(a3-a2))
        #print('p_to_p: {}'.format(a4-a3))
        #print('v_to_p: {}'.format(a5-a4))
        return sp_tensor, p_coords, p_features

class PVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.pv_blocks = nn.ModuleList()
        for pv_cfg in model_cfg.CONFIG:
            self.pv_blocks.append(PVBlock(pv_cfg, point_cloud_range))
        self.v_conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.v_conv_output = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.p_mlp_input = nn.Sequential(
            nn.Conv1d(input_channels, 16, 1, bias = False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
    
    def forward(self, batch_dict):
        v_features, v_indices = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=v_features,
            indices=v_indices.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        p_coords = batch_dict['points'][:,0:4]
        p_features = batch_dict['points'][:, 1:5] # xyzr
        p_features = self.p_mlp_input(p_features.unsqueeze(0).transpose(1,2))
        p_features = p_features.squeeze(0).transpose(0,1) # [N, C]
        sp_tensor = self.v_conv_input(input_sp_tensor)

        intermediate_data_dict = {}
        intermediate_data_dict.update({'p_features_0': p_features})
        for i, pv_block in enumerate(self.pv_blocks):
            sp_tensor, p_coords, p_features = pv_block(sp_tensor, p_coords, p_features)
            p_features_name = 'p_features_{}'.format(i+1)
            intermediate_data_dict.update({p_features_name: p_features})
        
        out = self.v_conv_output(sp_tensor)
        batch_dict.update({'multi_scale_point_features': intermediate_data_dict})

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        return batch_dict

class PVBackbone2(nn.Module):
    """
    only consider voxel-to-point ops
    """
    def __init__(self, model_cfg, input_channels, grid_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = sp_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.pv_blocks = nn.ModuleList()
        for pv_cfg in model_cfg.CONFIG:
            self.pv_blocks.append(PVBlock2(pv_cfg, point_cloud_range))
        
        self.num_point_features = 128
    
    def forward(self, batch_dict):
        v_features, v_indices = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        sp_tensor = spconv.SparseConvTensor(
            features=v_features,
            indices=v_indices.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        sp_tensor = self.conv_input(sp_tensor)
        sp_tensor = self.conv1(sp_tensor)

        p_coords = batch_dict['points'][:,0:4]
        p_features = batch_dict['points'][:, 1:5] # xyzr

        intermediate_data_dict = {}
        intermediate_data_dict.update({'p_features_0': p_features})
        for i, pv_block in enumerate(self.pv_blocks):
            sp_tensor, p_coords, p_features = pv_block(sp_tensor, p_coords, p_features)
            p_features_name = 'p_features_{}'.format(i+1)
            intermediate_data_dict.update({p_features_name: p_features})
        
        batch_dict.update({'multi_scale_point_features': intermediate_data_dict})

        batch_dict.update({
            'encoded_spconv_tensor': sp_tensor,
            'encoded_spconv_tensor_stride': 8
        })
        return batch_dict
