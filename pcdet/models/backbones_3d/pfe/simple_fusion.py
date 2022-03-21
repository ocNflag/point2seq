import torch
import torch.nn as nn
from ....utils import common_utils

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

class SimpleFusion(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        c_in = 16 + 32 + 64 + 64 + num_bev_features

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features_stack(self, points, bev_features, batch_size, bev_stride):
        """
        Args:
            points: [N1+N2, 4]
        Returns:
            point_bev_features: [N1+N2, num_bev_features]
        """
        x_idxs = (points[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (points[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[points[:, 0] == k]
            cur_y_idxs = y_idxs[points[:, 0] == k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1+N2, C0)
        return point_bev_features

    def forward(self, batch_dict):
        point_coords = batch_dict['point_coords']

        point_features_list = []

        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name == 'bev':
                point_bev_features = self.interpolate_from_bev_features_stack(
                    point_coords, batch_dict['spatial_features'], batch_dict['batch_size'],
                    bev_stride=batch_dict['spatial_features_stride']
                )
                point_features_list.append(point_bev_features)
            else:
                point_features_list.append(batch_dict['multi_scale_point_features'][src_name])

        point_features = torch.cat(point_features_list, dim=1)

        batch_dict['point_features_before_fusion'] = point_features
        point_features = self.vsa_point_feature_fusion(point_features)

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict

