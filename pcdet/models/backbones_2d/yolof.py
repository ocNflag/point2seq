import torch
import torch.nn as nn

class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self, model_cfg, input_channels):
        super(DilatedEncoder, self).__init__()
        self.input_channels = input_channels
        self.encoder_channels = model_cfg.ENCODER_CHANNEL
        self.mid_channels = model_cfg.MID_CHANNEL
        self.block_dilations = model_cfg.BLOCK_DILATIONS
        self.block_replications = model_cfg.BLOCK_REPLICATIONS
        self.num_bev_features = self.encoder_channels
        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.input_channels, self.encoder_channels, kernel_size=1)
        self.lateral_norm = nn.BatchNorm2d(self.encoder_channels, eps=1e-3, momentum=0.01)

        self.fpn_conv = nn.Conv2d(self.encoder_channels, self.encoder_channels, kernel_size=3, padding=1)
        self.fpn_norm = nn.BatchNorm2d(self.encoder_channels, eps=1e-3, momentum=0.01)
        encoder_blocks = []
        for dilation, rep in zip(self.block_dilations, self.block_replications):
            for i in range(rep):
                encoder_blocks.append(
                    Bottleneck(
                        self.encoder_channels,
                        self.mid_channels,
                        dilation=dilation,
                    )
                )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weight(self):
        xavier_fill(self.lateral_conv)
        xavier_fill(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        out = self.lateral_norm(self.lateral_conv(spatial_features))
        out = self.fpn_norm(self.fpn_conv(out))
        rlt = self.dilated_encoder_blocks(out)

        data_dict['spatial_features_2d'] = rlt
        return data_dict


class Bottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, dilation):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out

def xavier_fill(module: nn.Module):
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
