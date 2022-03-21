import torch
import copy
import torch.nn as nn
import numpy as np

from operator import sub
from pcdet.ops.votr_ops import votr_utils
from pcdet.models.backbones_2d.swin_helpers import GELU
from pcdet.models.backbones_3d.vfe.pillar_vfe import PFNLayer


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape).contiguous()
    return ret


class MLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_ch, mid_ch=None, out_ch=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = mid_ch or in_ch
        self.fc1 = nn.Linear(in_ch, mid_ch)
        self.act = act_layer()
        self.fc2 = nn.Linear(mid_ch, out_ch)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GridToBEV(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            bev_ch,
            strides,
            hash_size,
            max_point,
            num_ds_voxels,
            point_cloud_range,
            rf_ratio = 1.0,
            use_attn=False,
            attn_setting=None,
            use_positional_embedding=False,
            use_relative_coords=True,
            **kwargs
    ):
        super(GridToBEV, self).__init__()
        self.use_attn = use_attn
        self.attn_setting = attn_setting
        self.use_positional_embedding = use_positional_embedding
        self.use_relative_coords = use_relative_coords
        assert len(strides) == 2, 'bev stride only has 2 dim'
        self.strides = strides
        self.hash_size = hash_size
        self.max_point = max_point
        self.num_ds_voxels = num_ds_voxels
        self.point_cloud_range = point_cloud_range
        self.rf_ratio = rf_ratio

        self.in_ch = in_ch
        self.mid_ch = in_ch
        self.out_ch = out_ch
        self.bev_ch = bev_ch

        self.k_pos_proj = None

        if self.use_positional_embedding:
            self.k_pos_proj = nn.Sequential(
                nn.Conv1d(3, in_ch, 1),
                GELU(),
            )
        else:
            self.mid_ch += 3

        self.pointnet = nn.Sequential(
            PFNLayer(self.mid_ch, self.mid_ch, True, False),
            PFNLayer(self.mid_ch, self.out_ch, True, not self.use_attn),
        )

        if self.use_attn:
            self.bev_proj = nn.Sequential(
                nn.Linear(self.bev_ch, self.out_ch),
                GELU(),
            )
            self.mhead_attn = nn.MultiheadAttention(
                embed_dim=self.out_ch,
                num_heads=self.attn_setting['num_head'],
                dropout=self.attn_setting['drop_rate'],
            )
            self.mlp = MLP(
                in_ch=self.out_ch
            )
            self.norm_q = nn.LayerNorm(self.out_ch)
            self.norm_k = nn.LayerNorm(self.out_ch)
            self.norm_f = nn.LayerNorm(self.out_ch)


    def build_map_table(self, x):
        bs_cnt = torch.zeros(x.batch_size).int()
        for i in range(x.batch_size):
            bs_cnt[i] = (x.indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(x.indices.device)
        map_table = votr_utils.build_hash_table(
            x.batch_size,
            self.hash_size,
            x.spatial_shape,
            x.indices,
            bs_cnt,
        )
        return map_table

    @torch.no_grad()
    def downsample(self, sp_tensor):
        x_shape = sp_tensor.spatial_shape[0] // self.strides[0]
        y_shape = sp_tensor.spatial_shape[1] // self.strides[1]
        z_shape = 1
        new_spatial_shape = [x_shape, y_shape, z_shape]
        new_indices, new_map_table = votr_utils.hash_table_down_sample(
            self.strides + [sp_tensor.spatial_shape[2], ],
            self.num_ds_voxels,
            sp_tensor.batch_size,
            self.hash_size,
            new_spatial_shape,
            sp_tensor.indices
        )
        return new_spatial_shape, new_indices, new_map_table

    @torch.no_grad()
    def create_gather_dict(self, map_table, voxel_indices, spatial_shape):
        range = [
            [
                0, int(self.strides[0] // 2 * self.rf_ratio), 1,
                0, int(self.strides[1] // 2 * self.rf_ratio), 1,
                0, spatial_shape[2] // 2 + 1, 1
            ],
        ]
        strides = [self.strides[0], self.strides[1], spatial_shape[2]]
        gather_indices = votr_utils.bev_sparse_strided_attention_hash_indices(
            spatial_shape,
            self.max_point,
            range,
            strides,
            map_table,
            voxel_indices
        )
        gather_mask = (gather_indices < 0)
        return gather_indices, gather_mask

    @torch.no_grad()
    def with_bs_cnt(self, indices, batch_size):
        bs_cnt = torch.zeros(batch_size).int()
        for i in range(batch_size):
            bs_cnt[i] = (indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(indices.device)
        return bs_cnt

    @torch.no_grad()
    def with_coords(self, indices, point_cloud_range, spatial_shape):
        pc_size = torch.tensor(list(map(sub, point_cloud_range[3:], point_cloud_range[:3])))
        voxel_size = pc_size / torch.tensor(spatial_shape)
        voxel_size = voxel_size.unsqueeze(0).to(indices.device)
        min_range = torch.tensor(point_cloud_range[0:3]).unsqueeze(0).to(indices.device)
        coords = (indices[:, [3, 2, 1]].float() + 0.5) * voxel_size + min_range
        return coords

    def densify(self, spatial_shape, batch_size, ch, indices, features):
        reverse_spatial_shape = spatial_shape[::-1]  # (ZYX)
        output_shape = [batch_size] + list(reverse_spatial_shape) + [ch,]
        res = scatter_nd(
            indices.to(features.device).long(),
            features,
            output_shape
        )
        ndim = len(reverse_spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        res = res.permute(*trans_params).contiguous() # b c z y x
        return res.squeeze(2)

    def forward(self, sp_tensor, bev=None):
        bev = bev.permute(0, 2, 3, 1).contiguous()
        sp_tensor.spatial_shape = sp_tensor.spatial_shape[::-1].copy()
        map_table = self.build_map_table(sp_tensor)
        new_spatial_shape, new_indices, new_map_table = self.downsample(sp_tensor)
        key_indices, key_mask = self.create_gather_dict(map_table, new_indices, sp_tensor.spatial_shape)

        voxel_features = sp_tensor.features
        v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size)
        k_bs_cnt = self.with_bs_cnt(new_indices, sp_tensor.batch_size)

        key_feats = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices, k_bs_cnt)
        sp_coords = self.with_coords(sp_tensor.indices, self.point_cloud_range, sp_tensor.spatial_shape)
        key_coords = votr_utils.grouping_operation(sp_coords, v_bs_cnt, key_indices, k_bs_cnt)
        que_coords = self.with_coords(new_indices, self.point_cloud_range, new_spatial_shape)

        bev_indices = new_indices[:, [0, 2, 3]].long()
        bev_feats = bev[bev_indices[:, 0], bev_indices[:, 1], bev_indices[:, 2]].contiguous()

        if self.use_relative_coords:
            key_coords = key_coords - que_coords.unsqueeze(-1)

        if self.use_positional_embedding:
            key_pos_emb = self.k_pos_proj(key_coords)
            key_feats = key_feats + key_pos_emb
        else:
            key_feats = torch.cat([key_feats, key_coords], dim=-1)

        if self.use_attn:
            key_feats = key_feats.permute(2, 0, 1).squeeze().contiguous()
            shortcut = self.bev_proj(bev_feats)

            bev_feats = self.norm_q(shortcut).unsqueeze(0)
            key_feats = self.norm_k(key_feats)

            attend_feats, attend_weights = self.mhead_attn(
                query=bev_feats,
                key=key_feats,
                value=key_feats,
                key_padding_mask=key_mask
            )

            key_feats = attend_feats + shortcut
            key_feats = self.mlp(self.norm_f(key_feats)) + key_feats
            key_feats = key_feats.squeeze(0)
        else:
            key_feats = key_feats * (~key_mask.unsqueeze(1))
            key_feats = self.pointnet(key_feats.permute(0, 2, 1).contiguous())
            key_feats = key_feats.permute(1, 0, 2).squeeze().contiguous()

        mid_bev = self.densify(new_spatial_shape, sp_tensor.batch_size, self.out_ch, new_indices, key_feats)
        # 这里要不要再搞几层卷积?
        return mid_bev


class CrossFPN(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        self.grid_input_settings = self.model_cfg.get('GRID_INPUTS_SETTINGS', None)
        if self.grid_input_settings is not None:
            self.grid_base_settings = self.model_cfg.GRID_BASE_SETTINGS

        grid_chs = []
        self.grid_to_bev_layers = nn.ModuleDict()
        for iter in self.grid_input_settings:
            name = iter.pop('name')
            grid_chs.append(iter['out_ch'])
            tmp = copy.deepcopy(self.grid_base_settings)
            tmp.update(iter)
            tmp_grid_layer = GridToBEV(**tmp)
            self.grid_to_bev_layers[name] = tmp_grid_layer

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                GELU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    GELU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        GELU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        GELU()
                    ))

        c_in = sum(num_upsample_filters + grid_chs)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                GELU()
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        multi_scale_3d_features = data_dict['multi_scale_3d_features']
        mid_bevs = []
        for iter in self.grid_to_bev_layers.keys():
            mid_bevs.append(self.grid_to_bev_layers[iter](multi_scale_3d_features[iter], x))
        mid_bev = torch.cat(mid_bevs, dim=1)
        x = torch.cat([x, mid_bev], dim=1)

        data_dict['spatial_features_2d'] = x


        return data_dict