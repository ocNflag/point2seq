import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.dense_heads.utils import NaiveSepHead, kaiming_init


class HydraFusionHead(nn.Module):
    def __init__(self, num_cls, heads, in_ch, head_ch=64, final_kernel=1, bn=False, init_bias=-2.19, key_emb_dim=16,
                 value_emb_dim=16, voxel_size=None, point_cloud_range=None, out_size_factor=None, sampler_params=None,
                 **kwargs):
        super(HydraFusionHead, self).__init__()
        self.head_configs = heads

        # heatmap prediction head
        self.center_conv = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.center_head.bias.data.fill_(init_bias)

        # corner_map prediction head
        self.corner_conv = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.corner_head.bias.data.fill_(init_bias)

        self.fg_conv = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(head_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.fg_head.bias.data.fill_(init_bias)

        self.sample_configs = None
        self.base_sampler = BaseSampler(
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            sampler_params=sampler_params
        )

        self.fusion_module = HydraFusion(
            in_ch=in_ch,
            num_cls=num_cls,
            head_ch=head_ch,
            key_emb_dim=key_emb_dim,
            value_emb_dim=value_emb_dim,
            head_configs=self.head_configs,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=out_size_factor,
            init_bias=init_bias
        )

    def forward(self, base_feats):

        center_feat = self.center_conv(base_feats)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(base_feats)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(base_feats)
        fg_map = self.fg_head(fg_feat)

        sample_grids = self.base_sampler(center_map)
        sample_feats = {
            'center': [center_feat, center_map],
            'corner': [corner_feat, corner_map],
            'foreground': [fg_feat, fg_map],
        }

        ret = self.fusion_module(sample_feats, sample_grids, base_feats)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        return ret


class HydraFusion(nn.Module):
    def __init__(self, in_ch, num_cls, head_ch, key_emb_dim, value_emb_dim, head_configs,
                 voxel_size, point_cloud_range, out_size_factor, init_bias, **kwargs):
        super(HydraFusion, self).__init__()
        tmp = torch.FloatTensor(voxel_size) * out_size_factor
        self.register_buffer('voxel_size', tmp)
        self.point_cloud_range = point_cloud_range
        self.vx, self.vy, *_ = self.voxel_size
        self.x_min, self.y_min, *_ = point_cloud_range

        self.head_configs = head_configs
        self.base_querie_modules = nn.ModuleDict()
        self.center_key_modules = nn.ModuleDict()
        self.center_value_modules = nn.ModuleDict()
        self.corner_key_modules = nn.ModuleDict()
        self.corner_value_modules = nn.ModuleDict()
        self.fg_key_modules = nn.ModuleDict()
        self.fg_value_modules = nn.ModuleDict()
        self.predict_modules = nn.ModuleDict()

        for head_key in self.head_configs:
            out_ch, *_ = self.head_configs[head_key]
            self.base_querie_modules[head_key] = nn.Sequential(
                nn.Conv2d(in_ch, key_emb_dim, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(key_emb_dim),
                nn.ReLU(),
            )
            self.center_key_modules[head_key] = nn.Sequential(
                nn.Conv2d((head_ch + 2 + num_cls) * 9, key_emb_dim, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(key_emb_dim),
                nn.ReLU(),
            )
            self.center_value_modules[head_key] = nn.Sequential(
                nn.Conv2d((head_ch + 2 + num_cls) * 9, value_emb_dim, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(value_emb_dim),
                nn.ReLU(),
            )
            self.corner_key_modules[head_key] = nn.Sequential(
                nn.Conv2d((head_ch + 2 + (num_cls * 4)) * 8, key_emb_dim, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(key_emb_dim),
                nn.ReLU(),
            )
            self.corner_value_modules[head_key] = nn.Sequential(
                nn.Conv2d((head_ch + 2 + (num_cls * 4)) * 8, value_emb_dim, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(value_emb_dim),
                nn.ReLU(),
            )
            self.fg_key_modules[head_key] = nn.Sequential(
                nn.Conv2d((head_ch + 2 + num_cls) * 9, key_emb_dim, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(key_emb_dim),
                nn.ReLU(),
            )
            self.fg_value_modules[head_key] = nn.Sequential(
                nn.Conv2d((head_ch + 2 + num_cls) * 9, value_emb_dim, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(value_emb_dim),
                nn.ReLU(),
            )
            self.predict_modules[head_key] = nn.Sequential(
                nn.Conv2d(value_emb_dim + in_ch, head_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(head_ch),
                nn.ReLU(),
                nn.Conv2d(head_ch, out_ch, kernel_size=1, padding=0, bias=True)
            )
            self.predict_modules[head_key][-1].bias.data.fill_(init_bias)


    def recover_coords(self, grids, h, w):
        grid_x = grids[:, [0], :, :, :]
        grid_y = grids[:, [1], :, :, :]
        grid_x = (grid_x) / w
        grid_y = (grid_y) / h
        return torch.cat([grid_x, grid_y], dim=1)

    def sample_feats(self, feats, grids):
        # grids (B, 2, n, H, W)
        bs, _, n, h, w = grids.shape
        geo_feats = self.recover_coords(grids, h, w)
        grids = grids.permute(0, 2, 3, 4, 1).contiguous().view(bs, n * h, w, 2)
        # Normalize to [-1, 1], x->W, y->H
        grid_x = (grids[..., [0]] - w / 2) / (w / 2)
        grid_y = (grids[..., [1]] - h / 2) / (h / 2)

        sample_grids = torch.cat([grid_x, grid_y], dim=-1).contiguous()
        feats = torch.cat(feats, dim=1)
        sample_feats = F.grid_sample(feats, sample_grids)

        _, ch, _, _ = sample_feats.shape
        sample_feats = sample_feats.view(bs, ch, n, h, w)
        sample_feats = torch.cat([sample_feats, geo_feats], dim=1)
        return sample_feats.view(bs, (ch + 2) * n, h, w)

    def forward(self, sample_feats, sample_grids, base_feats):
        center_sampled_feats = self.sample_feats(sample_feats['center'], sample_grids['center'])
        corner_sampled_feats = self.sample_feats(sample_feats['corner'], sample_grids['corner'])
        fg_sampled_feats = self.sample_feats(sample_feats['foreground'], sample_grids['foreground'])

        ret = {}
        for head_key in self.head_configs:
            base_q = self.base_querie_modules[head_key](base_feats)

            center_k = self.center_key_modules[head_key](center_sampled_feats)
            center_v = self.center_value_modules[head_key](center_sampled_feats)
            center_a = (base_q * center_k).sum(1, keepdim=True)  # (B, 1, W, H)

            corner_k = self.corner_key_modules[head_key](corner_sampled_feats)
            corner_v = self.corner_value_modules[head_key](corner_sampled_feats)
            corner_a = (base_q * corner_k).sum(1, keepdim=True)  # (B, 1, W, H)

            fg_k = self.fg_key_modules[head_key](fg_sampled_feats)
            fg_v = self.fg_value_modules[head_key](fg_sampled_feats)
            fg_a = (base_q * fg_k).sum(1, keepdim=True)  # (B, 1, W, H)

            attention_weights = F.softmax(torch.cat([center_a, corner_a, fg_a], dim=1), dim=1).unsqueeze(2)  # (B, 3, 1, H, W)
            attention_feats = torch.stack([center_v, corner_v, fg_v], dim=1)  # (B, 3, C, H, W)
            attention_feats = (attention_weights * attention_feats).sum(1)  # (B, C, H, W)
            # attention_feats = center_v + corner_v + fg_v
            final_feats = torch.cat([base_feats, attention_feats], dim=1)
            pred_feats = self.predict_modules[head_key](final_feats)
            ret[head_key] = pred_feats

        return ret


class BaseSampler(nn.Module):
    def __init__(self, voxel_size, out_size_factor, sampler_params):
        super(BaseSampler, self).__init__()
        tmp = torch.FloatTensor(voxel_size[:2]).view(1, -1) * out_size_factor
        self.register_buffer('voxel_size', tmp)
        self.center_stride = stride = sampler_params['center_stride']
        self.dim_w = w = sampler_params['dim_w']
        self.dim_l = l = sampler_params['dim_l']

        corner_x = torch.FloatTensor([-w / 2, -w / 2, w / 2, w / 2, l / 2, -l / 2, l / 2, -l / 2])
        corner_y = torch.FloatTensor([l / 2, -l / 2, l / 2, -l / 2, -w / 2, -w / 2, w / 2, w / 2])

        corner_points = torch.stack([corner_x, corner_y], dim=1) / tmp
        corner_points = corner_points.view(1, 2, 8, 1, 1).contiguous()
        self.register_buffer('corner_points', corner_points)

        r = max(w, l)
        fg_x = torch.FloatTensor([-r / 3, -r / 3, -r / 3, 0, 0, 0, r / 3, r / 3, r / 3])
        fg_y = torch.FloatTensor([-r / 3, 0, r / 3, -r / 3, 0, r / 3, -r / 3, 0, r / 3])
        fg_points = torch.stack([fg_x, fg_y], dim=1) / tmp
        fg_points = fg_points.view(1, 2, 9, 1, 1).contiguous()
        self.register_buffer('fg_points', fg_points)

        grid_y, grid_x = torch.meshgrid([torch.arange(-stride, stride + 1, stride), torch.arange(-stride, stride + 1, stride)])
        center_points = torch.stack([grid_x, grid_y], dim=0).contiguous().view(1, 2, 9, 1, 1)
        self.register_buffer('center_points', center_points)

    def get_raw_grid(self, center_map):
        bs, _, h, w = center_map.shape
        grid_y, grid_x = torch.meshgrid([torch.arange(0, h), torch.arange(0, w)])
        grids = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(bs, 1, 1, 1).contiguous() + 0.5
        return grids.to(center_map)

    def _rot(self, x, y, sin_r, cos_r):
        nx = x * cos_r - y * sin_r
        ny = y * cos_r + x * sin_r
        return nx, ny


    def get_sample_grids(self, raw_grid):
        center_grids = raw_grid.unsqueeze(2) + self.center_points
        corner_grids = raw_grid.unsqueeze(2) + self.corner_points
        fg_grids = raw_grid.unsqueeze(2) + self.fg_points
        return center_grids, corner_grids, fg_grids

    def forward(self, center_map):
        raw_grid = self.get_raw_grid(center_map)
        center_grids, corner_grids, fg_grids = self.get_sample_grids(raw_grid)
        grid_dict = {
            'center': center_grids,
            'foreground': fg_grids,
            'corner': corner_grids,
        }
        return grid_dict
