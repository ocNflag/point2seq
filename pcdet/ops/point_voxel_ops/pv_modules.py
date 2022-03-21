import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pv_utils
from ..pointnet2.pointnet2_stack import pointnet2_utils

class VoxelModuleMSG(nn.Module):
    def __init__(self, steps, dilated_rates, bins, nsamples, mlps, v_size, v_range, point_cloud_range, max_hash_size):
        super().__init__()
        self.v_size = v_size
        self.v_range = v_range
        self.z_max, self.y_max, self.x_max = v_range
        self.x_size, self.y_size, self.z_size = v_size
        self.point_cloud_range = point_cloud_range
        self.max_hash_size = max_hash_size

        assert len(steps) == len(nsamples) == len(mlps) == len(dilated_rates)
        self.steps = steps
        self.dilated_rates = dilated_rates
        self.bins = bins
        self.nsamples = nsamples
        self.mlps = nn.ModuleList()
        for mlp in mlps:
            mlp[0] += 3 # xyz
            shared_mlps = []
            for k in range(len(mlp) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
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

    def get_v_to_p_map_stack(self, batch_size, p_coords, v_indices, v_bs_cnt, p_bs_cnt):
        """
        Returns:
            p_maps: [N1+N2, nsamples]
            p_masks: [N1+N2, nsamples]
        """
        min_range = [0] + list(self.point_cloud_range[0:3]) # [0, min_x, min_y, min_z]
        min_range = torch.tensor(min_range).unsqueeze(0).to(v_indices.device)
        p_coords = p_coords.contiguous()
        # IMPORTANT !!!!!
        # normalized p_coords
        p_coords = p_coords - min_range

        xyz_to_vidx = torch.cuda.IntTensor(batch_size, self.max_hash_size, 2).fill_(-1)           
        xyz_to_vidx = pv_utils.fill_dense_stack(v_indices, xyz_to_vidx, v_bs_cnt, self.v_range)
        #xyz_to_vidx = pv_utils.fill_dense_stack_simple(v_indices, xyz_to_vidx, v_bs_cnt, self.v_range)

        """
        import h5py
        f = h5py.File('xyz_to_vidx.h5', 'w')
        f.create_dataset('hash', data = xyz_to_vidx.detach().cpu().numpy())
        f.create_dataset('v', data = v_indices.detach().cpu().numpy())
        f.create_dataset('p', data = p_coords.detach().cpu().numpy())
        f.close()
        import pdb; pdb.set_trace()
        """

        p_maps = []
        empty_masks = []
        for i in range(len(self.steps)):
            # p_map, p_mask = pv_utils.voxel_point_query_stack(p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, 
            #                                                     self.v_size, self.v_range, self.nsamples[i], self.steps[i], self.dilated_rates[i], self.bins[i])
            p_map, p_mask, p_bin = pv_utils.voxel_point_query_bin(p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, 
                                                                self.v_size, self.v_range, self.nsamples[i], self.steps[i], self.dilated_rates[i], self.bins[i])
            # p_map, p_mask, p_bin = pv_utils.voxel_point_query_bin_simple(p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, 
            #                                                                 self.v_size, self.v_range, self.nsamples[i], self.steps[i], self.dilated_rates[i], self.bins[i])       
            empty_mask = (p_mask[:, 0] == 0) # nsample = 0
            # p_map[p_map < 0] = 0 # p_map init with 0
            p_maps.append(p_map)
            empty_masks.append(empty_mask)

        """
        import h5py
        f = h5py.File('p_map.h5', 'w')
        f.create_dataset('p_map', data = p_maps[0].detach().cpu().numpy())
        f.create_dataset('p_mask', data = p_map_masks[0].detach().cpu().numpy())
        f.close()
        import pdb; pdb.set_trace()
        """

        del xyz_to_vidx
        del p_bin
        torch.cuda.empty_cache()
        return p_maps, empty_masks 

    def get_v_coords_3(self, p_range, v_size, v_indices):
        """
        Args:
            p_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            v_size: [vx, vy, vz]
            v_indices : [M, 4] -> [bs, z_i, y_i, x_i]
        Returns:
            v_coords: [M, 3] -> [x, y, z]
        """
        v_size = torch.tensor(v_size).unsqueeze(0).to(v_indices.device)
        min_range = torch.tensor(p_range[0:3]).unsqueeze(0).to(v_indices.device)
        v_xyz_idx = v_indices[:, [3, 2, 1]]
        v_bs = v_indices[:, [0]].float()
        v_xyz = (v_indices[:, [3, 2, 1]].float() + 0.5) * v_size + min_range
        #v_coords = torch.cat([v_bs, v_xyz], dim = 1)
        v_coords = v_xyz
        return v_coords

    def forward(self, sp_tensor, p_coords):
        """
        Args:
            p_coords: [N1+N2, 4]
        Returns:
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
        v_coords = self.get_v_coords_3(self.point_cloud_range, self.v_size, v_indices)

        v_bs_cnt = torch.zeros((batch_size)).int().to(v_indices.device)
        p_bs_cnt = torch.zeros((batch_size)).int().to(p_coords.device)
        for i in range(batch_size):
            v_bs_cnt[i] = (v_indices[:, 0] == i).sum()
            p_bs_cnt[i] = (p_coords[:, 0] == i).sum()

        with torch.no_grad():
            p_maps, empty_masks = self.get_v_to_p_map_stack(batch_size, p_coords, v_indices, v_bs_cnt, p_bs_cnt)
        assert p_maps[0].shape[0] == p_coords.shape[0], \
            'the number of points should be equal, p_map: {}, p_coords: {}'.format(p_maps[0].shape[0], p_coords.shape[0])

        v_to_p_features_list = []
        for i in range(len(self.steps)):
            v_to_p_features = pointnet2_utils.grouping_operation(v_features, v_bs_cnt, p_maps[i], p_bs_cnt) # [N1+N2, C, ns]
            v_to_p_coords = pointnet2_utils.grouping_operation(v_coords, v_bs_cnt, p_maps[i], p_bs_cnt) # [N1+N1, 3, ns]
            """
            import h5py
            f = h5py.File('pv.h5', 'w')
            f.create_dataset('v_coords', data = v_to_p_coords.detach().cpu().numpy())
            f.create_dataset('p_coords', data = p_coords.detach().cpu().numpy())
            f.close()
            import pdb; pdb.set_trace()
            """
            v_to_p_coords = v_to_p_coords - p_coords[:,1:].unsqueeze(-1)
            v_to_p_features = torch.cat([v_to_p_coords, v_to_p_features], dim = 1)
            empty_mask = empty_masks[i]
            v_to_p_features[empty_mask] = 0

            v_to_p_features = v_to_p_features.permute(1, 0, 2).unsqueeze(dim=0)
            v_to_p_features = self.mlps[i](v_to_p_features) # [1, C, N1+N2, ns]
            # mask out empty
            #p_mask = p_masks[i].unsqueeze(0).unsqueeze(0)
            #p_mask = p_mask.detach()
            #v_to_p_features = v_to_p_features * p_mask
            # maxpool
            v_to_p_features = F.max_pool2d(
                v_to_p_features, kernel_size=[1, v_to_p_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, N1+N2)
            v_to_p_features = v_to_p_features.squeeze(dim=0).permute(1, 0) # [N1+N2 , C]
            v_to_p_features_list.append(v_to_p_features)
        v_to_p_features_cat = torch.cat(v_to_p_features_list, dim=1)
        return v_to_p_features_cat
