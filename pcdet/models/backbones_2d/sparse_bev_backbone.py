import numpy as np
import torch
import torch.nn as nn
import spconv
from functools import partial


class SparseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

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

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        indice_key_temp = 'fpn_{0}_{1}_{2}'

        for idx in range(num_levels):
            cur_layers = [
                spconv.SparseConv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=1,
                    bias=False,
                    indice_key = indice_key_temp.format('block', idx, 0)
                ),
                norm_fn(num_filters[idx]),
                nn.ReLU()
            ]

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    spconv.SparseConv2d(
                        num_filters[idx],
                        num_filters[idx],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        indice_key=indice_key_temp.format('block', idx, k + 1)
                    ),
                    norm_fn(num_filters[idx]),
                    nn.ReLU()
                ])

            self.blocks.append(spconv.SparseSequential(*cur_layers))

            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        spconv.SparseSequential(
                            spconv.SparseConvTranspose2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx],
                                bias=False,
                                indice_key=indice_key_temp.format('deblock', idx, 0)
                            ),
                            norm_fn(num_upsample_filters[idx]),
                            nn.ReLU()
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(
                        spconv.SparseSequential(
                            spconv.SparseConv2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                stride,
                                stride=stride,
                                bias=False,
                                indice_key=indice_key_temp.format('deblock', idx, 0)
                            ),
                            norm_fn(num_upsample_filters[idx]),
                            nn.ReLU()
                        )
                    )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    spconv.SparseConvTranspose2d(
                        c_in,
                        c_in,
                        upsample_strides[-1],
                        stride=upsample_strides[-1],
                        bias=False,
                        indice_key='deblock_final'
                    ),
                    norm_fn(c_in),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        x = data_dict['spatial_features']
        ups = []
        # x = spconv.SparseConvTensor.from_dense(spatial_features.permute(0, 2, 3, 1).contiguous())

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x).dense())
            else:
                ups.append(x.dense())

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
